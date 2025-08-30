

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from utils.feature_extraction import extract_text_features
import clip
import torch.nn.init as init
from sklearn.metrics.pairwise import cosine_similarity
    
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough positon encoding matrix
        self.P = torch.zeros((1, max_len, num_hiddens)).cuda()
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

class SR(nn.Module):
    def __init__(self, vid_embed_size, hsize, rnn_enc, text_model,
                 text_feature_extractor='clip', tokenizer=None, context_encoder=None):
        super(SR, self).__init__()
        # k = 4 (frame + 3 bounding boxes per frame)
        # csv : k = 1
        self.vid_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                   rnn_type="lstm")
        self.text_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                    rnn_type="lstm")
        # self.text_ctx_rnn = rnn_enc(768, hsize, bidirectional=True, dropout_p=0, n_layers=1,
        #                             rnn_type="lstm")
        self.text_change = nn.Linear(vid_embed_size,hsize*2)
        self.text_model = text_model
        self.text_model.eval()
        self.text_feature_extractor = text_feature_extractor
        self.tokenizer = tokenizer 
        
        # transformer
        d_model = 300
        dropout = 0.5
        self.pos_encoder_coarse = PositionalEncoding(d_model, dropout)
        self.pos_encoder_fine = PositionalEncoding(d_model, dropout)
        self.transformer_coarse = nn.Transformer(d_model, nhead=6, num_encoder_layers=6)
        self.transformer_fine = nn.Transformer(d_model, nhead=6, num_encoder_layers=6)
        self.fc = nn.Linear(d_model*2, 1) 
        self.fc_coarse = nn.Linear(d_model, 1)
        self.fc_fine = nn.Linear(d_model, 1)
    
        self.sigmoid = nn.Sigmoid()  
        self.sigmoid_coarse = nn.Sigmoid()
        self.sigmoid_fine = nn.Sigmoid()

        self.context_encoder = context_encoder
        if self.context_encoder == 'mha':  # multi-head attention
            # positional encoding
            self.positional_encode = PositionalEncoding(num_hiddens=2*hsize, dropout=0.5)
            # multi-headed attention
            self.multihead_attn = nn.MultiheadAttention(embed_dim=2*hsize,
                                                        num_heads=10,
                                                        dropout=0.5,
                                                        batch_first=True)
        elif self.context_encoder == 'bilstm':
            self.bilstm = nn.LSTM(input_size=2*hsize,
                                  hidden_size=hsize,
                                  batch_first=True,
                                  bidirectional=True)
            
        self.vid_adapter = nn.Sequential(nn.Linear(2 * hsize, 2*hsize),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(2 * hsize, 2 * hsize))
        
        self.relation_query = nn.Sequential(nn.Linear(4 * hsize, hsize),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(hsize, 1))
        # for coarse
        self.global_token = nn.Parameter(torch.randn(1, d_model))

        # learnable node for binary classification
        self.learnable_node = nn.Parameter(torch.randn(1, d_model))

        self.apply(weights_init)
        self.fc1= nn.Linear(600, 512) 
        self.fc2 = nn.Linear(512, 1) 
 
        
        
        
        


    def query(self, seg_text_feats, vid_feature):
        """
        given segment video features and query features
        output a logit for querying the segment with the text features
        """
        # vid_feature = self.vid_adapter(vid_feature+seg_text_feats)
        return self.relation_query(torch.cat((seg_text_feats, vid_feature), dim=-1))[0]
        


    def dp_align(self, hypo_list, vid_feature):
        """
        optimized recurse: stores values of sub-problems in N*S array
        # nodes = N
        # segments = S
        vid_features: [1,num_segments,512 ? hsize*2=300]
        """
        vid_feature = vid_feature
        max_arr = [] # keeps track of max for each sorted sequence
        parent_dict = [] # keeps track of the best path for each sorted sequence
        num_segments = len(vid_feature)
        logits_arr = []  # keeps track of logits matrix (for each cell)

        # 初始化sorted_nodes
        sorted_nodes = [] # 子句的个数
        for i in range(len(hypo_list)):
            sorted_nodes.append(i)

        if len(vid_feature) < len(sorted_nodes):
            raise ValueError('lenghth error')
        
        with torch.no_grad():
            nodes_feat = extract_text_features(hypotheses=hypo_list,
                                                    model=self.text_model,
                                                    feature_extractor=self.text_feature_extractor,
                                                    tokenizer=self.tokenizer) 
        seg_text_feats, seg_text_lens = nodes_feat#[num_nodes, 20, 512],[num_nodes]
        seg_text_lens = seg_text_lens.to('cuda')
                
        # seg_text_feats = torch.cat(seg_text_feats, dim=0)
        _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)  # [num_nodes, 2*hsize]

        num_nodes = len(sorted_nodes)
        parent_dict= {k1: {k2: tuple() for k2 in range(num_segments)} for k1 in sorted_nodes}
        # array keeps track of cumulative max logprob for each cell
        arr = torch.full((num_nodes, num_segments), torch.tensor(-1000.)).cuda()
        logits_arr= torch.zeros((num_nodes, num_segments)).cuda()

        # setting the start & end indices of each node on segments
        start_ind = dict(zip(sorted_nodes, np.arange(0, num_nodes, 1)))
        end_ind = dict(zip(sorted_nodes, [num_segments - num_nodes + i for i in range(num_nodes)]))

        # starting outer loop from the last node
        for node_ind, node in zip(np.arange(num_nodes - 1, -1, -1), reversed(sorted_nodes)): # Start calculation from the last text node
            # starting inner loop from the last segment
            for segment_ind in range(end_ind[node], start_ind[node] - 1, -1): # Current text node may correspond to video node end->start


                # setting the value of the last column 
                if segment_ind == num_segments - 1:
                    logit = self.query(
                                        seg_text_feats[node_ind, :],
                                        vid_feature[segment_ind, :])
                    arr[node_ind][segment_ind] = F.logsigmoid(logit)
                    logits_arr[node_ind][segment_ind] = logit
                    parent_dict[node][segment_ind] = (segment_ind,)
                    continue

                logit = self.query(
                                seg_text_feats[node_ind, :],
                                vid_feature[segment_ind, :])

                # setting the values of the last row (except last cell in the row)
                if node_ind == num_nodes - 1:
                    V_opt_curr = F.logsigmoid(logit)    
                    V_opt_next = arr[node_ind][segment_ind + 1]
                    if V_opt_curr >= V_opt_next:
                        arr[node_ind][segment_ind] = V_opt_curr
                        parent_dict[node][segment_ind] =  (segment_ind,)
                    else:
                        arr[node_ind][segment_ind] = V_opt_next
                        parent_dict[node][segment_ind] = \
                            parent_dict[sorted_nodes[node_ind]][segment_ind + 1]

                # calculating the values of the remaining cells
                # dp[i][j] = max(query(i,j) + dp[i+1][j], dp[i][j+1])
                else:
                    # V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind]  # relaxation added
                    V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind + 1]  # no relaxation
                    V_opt_next = arr[node_ind][segment_ind + 1]
                    if V_opt_curr >= V_opt_next:
                        arr[node_ind][segment_ind] = V_opt_curr
                        parent_dict[node][segment_ind] = \
                                (segment_ind,) + parent_dict[sorted_nodes[node_ind + 1]][segment_ind + 1]
                    else:
                        arr[node_ind][segment_ind] = V_opt_next
                        parent_dict[node][segment_ind] = \
                            parent_dict[sorted_nodes[node_ind]][segment_ind + 1]
                logits_arr[node_ind][segment_ind] = logit

        # TODO: could be more than one optimum paths
        best_alignment = parent_dict[sorted_nodes[0]][0]
        # if best_alignment is None:
        #     raise ValueError()
        aggregated_logits = torch.tensor(0.).cuda()
        # aggregated_logits = []
        # length normalized aggregation of logits
        for i, j in zip(np.arange(num_nodes), best_alignment):
            aggregated_logits +=  logits_arr[i][j] / len(sorted_nodes)
            # aggregated_logits.append(logits_arr[max_sort_ind][i][j])
        adj_matrix = torch.zeros((num_nodes + num_segments, num_nodes + num_segments)).cuda()
        for i in range(num_nodes):# Establish connections between text nodes and video nodes
            adj_matrix[num_segments + i][best_alignment[i]] = 1
        return  adj_matrix, aggregated_logits,seg_text_feats



    
    def func(self,hypo,vid_feat):
        num_seg, vid_len, _ = vid_feat.shape
        vid_lens = torch.full((num_seg,), vid_len).cuda() #(size,full_value)，
        _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # aggregate [num_segments,300]
        

        # coarse alignment 
        with torch.no_grad():
            agg_hypo = ','.join(hypo)
            dis_feat = extract_text_features(hypotheses=agg_hypo,
                                                        model=self.text_model,
                                                        feature_extractor=self.text_feature_extractor,
                                                        tokenizer=self.tokenizer)
        agg_text_feats, agg_text_lens = dis_feat #[1,20,512],20
        agg_text_lens = agg_text_lens.to('cuda')
        _, agg_text_feats = self.text_ctx_rnn(agg_text_feats, agg_text_lens)  # [1, 2*hsize]

        combined_features = torch.cat((vid_feat, agg_text_feats), dim=0) 
        combined_features = torch.cat((combined_features, self.global_token), dim=0) # [x+1, 300]
        input = self.pos_encoder_coarse(combined_features.unsqueeze(0)).squeeze(0)#[x+1, 300]

        dummy_tgt = torch.zeros_like(input).to('cuda')  
        output_coarse = self.transformer_coarse(input,dummy_tgt)
        output_coarse = output_coarse[-1]


        # fine
        best_alignment_adj_matrix, aligned_aggregated,text_feat= self.dp_align(hypo, vid_feat)
        aligned_aggregated = torch.sigmoid(aligned_aggregated)
        
        num_vid = num_seg
        num_text = len(hypo)

        
        # Get adjacency matrix, convert to dense matrix, as attention mask for transformer
        adj_tensor = torch.zeros(num_vid+num_text,num_vid+num_text).to('cuda')
        for i in range(num_vid-1):
            adj_tensor[i][i+1] = 1
        for i in range(num_text-1):
            adj_tensor[i+num_vid][i+num_vid+1] = 1
        
        idx_edges = torch.nonzero(best_alignment_adj_matrix>0)
        for idx in idx_edges:
            idx_front = max(0, idx[1]-1)
            idx_back = min(num_vid, idx[1]+1)
            best_alignment_adj_matrix[idx[0], idx_front] = 1
            best_alignment_adj_matrix[idx[0], idx_back] = 1
        adj_tensor = best_alignment_adj_matrix+adj_tensor

        input = vid_feat.unsqueeze(0)
        # input = self.pos_encoder_fine(input)
        input = torch.cat((input,text_feat.unsqueeze(0)),dim=1)

        learnable_node_expanded = self.learnable_node.expand(1, 1, -1)
        input = torch.cat([input, learnable_node_expanded], dim=1)# .squeeze(0) # [m,300]
        input = self.pos_encoder_fine(input).squeeze(0)


        adj_tensor = torch.cat([adj_tensor, torch.zeros((1, adj_tensor.size(1)), device=adj_tensor.device)], dim=0)
        adj_tensor = torch.cat([adj_tensor, torch.zeros((adj_tensor.size(0), 1), device=adj_tensor.device)], dim=1)
        adj_tensor[num_vid+num_text-1][num_vid+num_text] = 1
        adj_tensor = torch.where(adj_tensor == 0, -1e9, 0.0).to(dtype=input.dtype) 

        dummy_tgt = torch.zeros_like(input).to('cuda') 
        output_fine = self.transformer_fine(input,dummy_tgt, src_mask=adj_tensor)
        output_fine = output_fine[-1]

        output=torch.cat((output_coarse, output_fine), dim=0)
        output1 = self.fc1(output)  #  [512]
        return output1, aligned_aggregated
    def get_closest_prototype(self, feature_vector, prototypes):
        """
        Calculate the similarity between a feature vector and a set of prototypes, return the index of the most similar prototype
        Args:
            feature_vector: single feature vector [feature_dim]
            prototypes: set of prototypes [n_clusters, feature_dim]
        Returns:
            best_cluster: Index of the most similar prototype
            similarity: Cosine similarity with the best prototype
        """
        if prototypes is None:
            return None, None
            
        # Ensure input is in the correct shape
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.unsqueeze(0)  # [1, feature_dim]
        
        # Convert to numpy array for cosine similarity calculation
        feature_np = feature_vector.detach().cpu().numpy()
        prototypes_np = prototypes.detach().cpu().numpy() if isinstance(prototypes, torch.Tensor) else prototypes
        
        # Calculate cosine similarity
        similarities = cosine_similarity(feature_np, prototypes_np)[0]
        
        # Find the most similar prototype
        best_cluster_idx = np.argmax(similarities)
        max_similarity = similarities[best_cluster_idx]
        
        return best_cluster_idx, max_similarity
    def calculate_cluster_alignment_loss(self, features, labels, prototypes, temperature=0.1):
        """Optimized cluster alignment loss calculation"""
        if not labels or prototypes is None:
            return torch.tensor(0.0).to(features.device).float()
            
        # Calculate similarity between features and all prototypes
        similarities = F.cosine_similarity(
            features.unsqueeze(1),  # [B, 1, D]
            prototypes.unsqueeze(0),  # [1, N, D]
            dim=2
        )  # [B, N]
        
        # Calculate contrastive loss
        labels = torch.tensor(labels).long().to(features.device)
        pos_mask = F.one_hot(labels, num_classes=prototypes.shape[0]).float()
        neg_mask = 1 - pos_mask
        
        # Positive sample loss
        pos_sim = similarities * pos_mask
        pos_loss = -torch.sum(pos_sim, dim=1) / (torch.sum(pos_mask, dim=1) + 1e-6)
        
        # Negative sample loss (using margin)
        margin = 0.5
        neg_sim = similarities * neg_mask
        neg_loss = torch.sum(F.relu(neg_sim - pos_sim.unsqueeze(1) + margin) * neg_mask, dim=1) / (torch.sum(neg_mask, dim=1) + 1e-6)
        
        # Entropy regularization
        probs = F.softmax(similarities / temperature, dim=1)
        entropy_reg = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        return pos_loss.mean() + 0.5 * neg_loss.mean() - 0.1 * entropy_reg.mean()
    def calculate_domain_losses(self, epoch,total_epochs,sim_outputs, real_outputs, sim_cluster_labels, 
                          sim_cluster_preds, real_cluster_labels, real_cluster_preds,
                          sim_prototypes, real_prototypes, device):

        
        # float32
        sim_prototypes = sim_prototypes.float()
        real_prototypes = real_prototypes.float()
        
        cross_domain_entropy_losses = []
        
        for sim_output, real_output in zip(sim_outputs, real_outputs):
            cross_domain_entropy = torch.tensor(0.0).to(device).float()
            
            # 1. Calculate entropy loss between simulated domain features and real domain prototypes
            if sim_output is not None and real_prototypes is not None:
    
                sim_output = sim_output.float()
                sim_to_real_sim = F.cosine_similarity(sim_output.unsqueeze(0).unsqueeze(0), real_prototypes.unsqueeze(0), dim=2).squeeze(0)  # [n_prototypes]
            
                # Apply softmax to get probability distribution
                temperature = max(0.1, 1.0 - epoch/total_epochs)
                sim_to_real_prob = F.softmax(sim_to_real_sim /temperature, dim=0)
                
                # Calculate entropy
                sim_to_real_entropy = -torch.sum(sim_to_real_prob * torch.log(sim_to_real_prob + 1e-10))
                cross_domain_entropy += sim_to_real_entropy
            
            # 2. Calculate entropy loss between real domain features and simulated domain prototypes
            if real_output is not None and sim_prototypes is not None:
              
                real_output = real_output.float()
                real_to_sim_sim = F.cosine_similarity(real_output.unsqueeze(0).unsqueeze(0), sim_prototypes.unsqueeze(0), dim=2).squeeze(0)  # [n_prototypes]
                
                # Apply softmax to get probability distribution
                temperature = max(0.1, 1.0 - epoch/total_epochs)
                real_to_sim_prob = F.softmax(real_to_sim_sim / temperature, dim=0)
                
                # Calculate entropy
                real_to_sim_entropy = -torch.sum(real_to_sim_prob * torch.log(real_to_sim_prob + 1e-10))
                cross_domain_entropy += real_to_sim_entropy
            
            cross_domain_entropy_losses.append(cross_domain_entropy)
        
        # 3. Calculate the mean cross-domain entropy loss
        cross_domain_entropy_loss = torch.stack(cross_domain_entropy_losses).mean() if cross_domain_entropy_losses else torch.tensor(0.0).to(device).float()

        # 4. Calculate cluster alignment loss
        cluster_alignment_loss = torch.tensor(0.0).to(device).float()
    
        if sim_cluster_labels and sim_prototypes is not None:
            sim_loss = self.calculate_cluster_alignment_loss(
                torch.stack(sim_outputs),
                sim_cluster_labels,
                sim_prototypes,
                temperature=max(0.1, 1.0 - epoch/total_epochs)
            )
            cluster_alignment_loss += sim_loss
            
        if real_cluster_labels and real_prototypes is not None:
            real_loss = self.calculate_cluster_alignment_loss(
                torch.stack(real_outputs),
                real_cluster_labels,
                real_prototypes,
                temperature=max(0.1, 1.0 - epoch/total_epochs)
            )
            cluster_alignment_loss += real_loss
            
        return cross_domain_entropy_loss + cluster_alignment_loss, cross_domain_entropy_loss, cluster_alignment_loss
    def forward(self, file_name, hypothese, vid_feat, real_name, real_hypothese, realv_feat, 
           sim_prototypes, real_prototypes, video_ass,epoch,total_epochs):
        """
        Forward propagation for a single sample
        Args:
            file_name: Name of the simulated video file
            hypothese: Hypothesis text for the simulated video
            vid_feat: Features of the simulated video
            real_name: Name of the real video file
            real_hypothese: Hypothesis text for the real video
            realv_feat: Features of the real video
            sim_prototypes: Prototypes of the simulated domain
            real_prototypes: Prototypes of the real domain
            video_ass: Mapping dictionary from video to cluster
        """
        # 1. Process simulated domain video
        sim_output, sim_aligned_aggregated = self.func(hypothese, vid_feat)
        
        # 2. Retrieve simulated domain cluster information
        sim_cluster_label = None
        sim_cluster_pred = None
        if video_ass is not None and file_name in video_ass:
            sim_cluster_label = video_ass[file_name]
            sim_cluster_pred, _ = self.get_closest_prototype(sim_output, sim_prototypes)
        
        # 3. Process real domain video
        real_output = None
        real_cluster_label = None
        real_cluster_pred = None
        if real_hypothese is not None and realv_feat is not None:
            real_output, _ = self.func(real_hypothese, realv_feat)
            
            # Retrieve real domain cluster information
            if video_ass is not None and real_name in video_ass:
                real_cluster_label = video_ass[real_name]
                real_cluster_pred, _ = self.get_closest_prototype(real_output, real_prototypes)
        
        # 4. Calculate domain adaptation loss
        if real_name is not None:
            
            domain_loss, cross_domain_entropy_loss, cluster_alignment_loss = self.calculate_domain_losses(
                epoch,
                total_epochs,
                sim_outputs=[sim_output],
                real_outputs=[real_output],
                sim_cluster_labels=[sim_cluster_label] if sim_cluster_label is not None else [],
                sim_cluster_preds=[sim_cluster_pred] if sim_cluster_pred is not None else [],
                real_cluster_labels=[real_cluster_label] if real_cluster_label is not None else [],
                real_cluster_preds=[real_cluster_pred] if real_cluster_pred is not None else [],
                sim_prototypes=sim_prototypes,
                real_prototypes=real_prototypes,
                device=vid_feat.device,
            )
        else:
            cross_domain_entropy_loss = torch.tensor(0.0).to(vid_feat.device).float()
            cluster_alignment_loss = torch.tensor(0.0).to(vid_feat.device).float()
            domain_loss = cross_domain_entropy_loss + cluster_alignment_loss
            
        sim_output = self.fc2(sim_output)
        # sim_output = self.sigmoid(sim_output)
        
        return sim_output, domain_loss