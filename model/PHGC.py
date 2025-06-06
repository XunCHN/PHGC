import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.feat_extraction import extract_text_features




class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens)).cuda()
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



class PHGC(nn.Module):
    def __init__(self, vid_embed_size, hsize, rnn_enc, text_model):
        """
        Initialize the PHGC model with video and text processing components.

        Args:
            vid_embed_size (int): Size of video feature embeddings.
            hsize (int): Hidden size for the RNN encoders.
            rnn_enc (nn.Module): RNN encoder class (e.g., LSTM).
            text_model (PreTrainedModel): Pre-trained text model.
            text_feature_extractor (str): Text feature extractor type ('clip' by default).
            tokenizer (Tokenizer): Tokenizer for text processing.
        """
        super(PHGC, self).__init__()

        # Video context RNN
        self.vid_ctx_rnn = rnn_enc(4 * vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        # Text context RNN
        self.text_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        # Text processing layers
        self.text_model = text_model
        self.text_model.eval()
        
        # Transformer Encoder
        d_model = hsize * 2
        dropout = 0.5
        self.pos_encoder_coarse = PositionalEncoding(d_model, dropout)
        self.pos_encoder_fine = PositionalEncoding(d_model, dropout)

        # Transformer encoder layers
        self.transformer_coarse = TransformerEncoder(TransformerEncoderLayer(d_model, nhead=6), num_layers=6)
        self.transformer_fine = TransformerEncoder(TransformerEncoderLayer(d_model, nhead=6), num_layers=6)

        # Output layers
        self.fc = nn.Linear(d_model * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # State and relation queries
        self.state_query = nn.Sequential(
            nn.Linear(4 * hsize, hsize),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hsize, 1)
        )
        
        self.relation_query = nn.Sequential(
            nn.Linear(4 * hsize, hsize),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hsize, 1)
        )
        
        # Learnable node and global token
        self.global_token = nn.Parameter(torch.randn(1, d_model))
        self.learnable_node = nn.Parameter(torch.randn(1, d_model))



    def query(self, query_type, seg_text_feats, vid_feature):
        """
        given segment video features and query features
        output a logit for querying the segment with the text features
        """
        if query_type == 'StateQuery':
            return self.state_query(torch.cat((seg_text_feats, vid_feature), dim=-1))[0]
        elif query_type == 'RelationQuery':
            return self.relation_query(torch.cat((seg_text_feats, vid_feature), dim=-1))[0]
        
    def process_nodes(self, nodes):
        """
        nodes: nodes of graph in DSL
        returns: list of node args in str format
        """
        pred_args = []
        queries = []
        for node in nodes:
            # 'Step 1 StateQuery(apple,heat)'
            node = re.sub('Step \d+ ', '', node)
            node = re.sub(r"[()]", " ", node).strip().split(" ")
            query_type, pred_arg = node[0], ','.join(node[1:])
            split_text = [pred_arg.split(',')[0], pred_arg.split(',')[-1]]
            pred_args.append(' '.join(split_text))
            queries.append(query_type)
        return pred_args, queries



    def dp_align(self, all_sorts, vid_feature):

        """
        Optimized dynamic programming alignment function.
        Computes the best alignment of video segments to the sorted textual descriptions using DP.
        
        Parameters:
        - all_sorts: List of sorted textual nodes.
        - vid_feature: Tensor of video features with shape (1, num_segments, hsize).
        
        Returns:
        - max_sort_ind: The index of the optimal sorted sequence.
        - max_arr[max_sort_ind]: The maximum cumulative log-probability for the best alignment.
        - best_alignment: The best alignment between video segments and textual queries.
        - aggregated_logits: Aggregated log-probabilities of the best alignment.
        - all_seg_text_feats: The textual features used for alignment.
        """
        
        vid_feature = vid_feature.unsqueeze(0)  # Shape: (1, num_segments, hsize)
        
        # Initialize tracking lists for results
        max_arr = [[]] * len(all_sorts)  # Stores max values for each sorted sequence
        parent_dict = [[]] * len(all_sorts)  # Tracks the best path for each sorted sequence
        logits_arr = [[]] * len(all_sorts)  # Stores logits matrix (each cell contains logit values)
        all_seg_text_feats = []  # Stores the text features for each sorted sequence

        num_segments = len(vid_feature[0])  # Number of video segments

        # Process each sorted sequence
        for ind, sorted_nodes in enumerate(all_sorts):
            nodes = all_sorts[ind]
            pred_args, queries = self.process_nodes(nodes)  # Process nodes to get arguments and queries

            seg_text_feats = []  # List to hold text features for segments

            with torch.no_grad():
                # Extract text features for each query node using the text model
                nodes_feat = extract_text_features(hypotheses=pred_args,model=self.text_model)
                seg_text_feats, seg_text_lens = nodes_feat
                seg_text_lens = seg_text_lens.to('cuda')

            # Pass text features through the RNN to get context-aware embeddings
            _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)
            seg_text_feats = seg_text_feats.unsqueeze(0)  # Add batch dimension
            all_seg_text_feats.append(seg_text_feats)

            num_nodes = len(sorted_nodes)  # Number of query nodes
            parent_dict[ind] = {k1: {k2: tuple() for k2 in range(num_segments)} for k1 in sorted_nodes}

            # Initialize DP array with very low log-probabilities (negative infinity)
            arr = torch.full((num_nodes, num_segments), torch.tensor(-100.0)).cuda()
            logits_arr[ind] = torch.zeros((num_nodes, num_segments)).cuda()

            # Start and end indices for each node in the sequence
            start_ind = dict(zip(sorted_nodes, np.arange(0, num_nodes, 1)))
            end_ind = dict(zip(sorted_nodes, [num_segments - num_nodes + i for i in range(num_nodes)]))

            # DP computation: Loop over nodes and segments in reverse order
            for node_ind, node in zip(np.arange(num_nodes - 1, -1, -1), reversed(sorted_nodes)):
                for segment_ind in range(end_ind[node], start_ind[node] - 1, -1):

                    # Compute the logit value for the last column (base case)
                    if segment_ind == num_segments - 1:
                        logit = self.query(queries[node_ind], seg_text_feats[:, node_ind, :], vid_feature[:, segment_ind, :])
                        arr[node_ind][segment_ind] = F.logsigmoid(logit)  # Log-sigmoid for log-probability
                        logits_arr[ind][node_ind][segment_ind] = logit
                        parent_dict[ind][node][segment_ind] = (segment_ind,)
                        continue

                    # Compute logit for the current cell
                    logit = self.query(queries[node_ind], seg_text_feats[:, node_ind, :], vid_feature[:, segment_ind, :])

                    # For the last row, use the DP recurrence to fill the values
                    if node_ind == num_nodes - 1:
                        V_opt_curr = F.logsigmoid(logit)
                        V_opt_next = arr[node_ind][segment_ind + 1]
                        if V_opt_curr >= V_opt_next:
                            arr[node_ind][segment_ind] = V_opt_curr
                            parent_dict[ind][node][segment_ind] = (segment_ind,)
                        else:
                            arr[node_ind][segment_ind] = V_opt_next
                            parent_dict[ind][node][segment_ind] = parent_dict[ind][sorted_nodes[node_ind]][segment_ind + 1]
                    else:
                        # Standard DP update for each cell
                        V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind + 1]  # No relaxation
                        V_opt_next = arr[node_ind][segment_ind + 1]
                        if V_opt_curr >= V_opt_next:
                            arr[node_ind][segment_ind] = V_opt_curr
                            parent_dict[ind][node][segment_ind] = (segment_ind,) + parent_dict[ind][sorted_nodes[node_ind + 1]][segment_ind + 1]
                        else:
                            arr[node_ind][segment_ind] = V_opt_next
                            parent_dict[ind][node][segment_ind] = parent_dict[ind][sorted_nodes[node_ind]][segment_ind + 1]

                    logits_arr[ind][node_ind][segment_ind] = logit

            # Store the maximum log-probability for the current sequence and normalize
            max_arr[ind] = arr[0][0] / len(sorted_nodes)

        # Find the best sequence based on the maximum log-probability
        max_sort_ind = torch.tensor(max_arr).argmax()  # Index of the best sequence
        best_alignment = parent_dict[max_sort_ind][all_sorts[max_sort_ind][0]][0]  # Best alignment sequence

        # Aggregate logits (normalized by sequence length)
        aggregated_logits = torch.tensor(0.).cuda()
        for i, j in zip(np.arange(num_nodes), best_alignment):
            aggregated_logits += logits_arr[max_sort_ind][i][j] / len(all_sorts[max_sort_ind])

        adj_matrix = torch.zeros((num_segments + num_nodes, num_segments + num_nodes)).cuda()
        for i in range(num_nodes):
            adj_matrix[num_segments + i][best_alignment[i]] = 1

        return max_sort_ind, adj_matrix, aggregated_logits, all_seg_text_feats[max_sort_ind].squeeze(0)




    def forward(self, file_names, vid_feats, text_graphs, hypotheses, labels):
        """
        Forward pass for the PHGC model.

        Args:
            file_names (list): List of file names.
            vid_feats (list): List of video features.
            text_graphs (list): List of text graphs.
            hypotheses (list): List of text hypotheses.
            labels (list): List of labels.

        Returns:
            torch.Tensor: The predicted labels.
            torch.Tensor: The ground truth labels.
        """
        labels_ls = []
        dp_preds = []
        map_preds = []
        

        for index, (file_name, vid_feat, text_graph, hypo, label) in enumerate(zip(file_names, vid_feats, text_graphs, hypotheses, labels)):
            b, vid_len, _ = vid_feat.shape
            vid_lens = torch.full((b,), vid_len).cuda()

            # Aggregate video features
            _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # Aggregate [num_segments, 2*hsize]

            # Coarse-grained text feature extraction and RNN processing
            with torch.no_grad():
                dis_feat = extract_text_features(hypotheses=hypo, model=self.text_model)
                seg_text_feats, seg_text_lens = dis_feat
                seg_text_lens = seg_text_lens.to('cuda')

                _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)  # [num_nodes, 2*hsize]

            # Combine video and text features
            combined_features = torch.cat((vid_feat, seg_text_feats), dim=0)  # [num_segments+1, 2*hsize]
            combined_features = torch.cat((combined_features, self.global_token), dim=0) # [num_segments+2, 2*hsize]

            # Apply positional encoding
            input = self.pos_encoder_coarse(combined_features.unsqueeze(0)).squeeze(0)

            # Coarse-grained transformer output
            output_coarse = self.transformer_coarse(input, mask=None)
            output_coarse = output_coarse[-1]

            # Fine-grained alignment using dynamic programming
            text_all_sorts = list(nx.all_topological_sorts(text_graph))
            sorted_seq_ind, cross_adj_matrix, aligned_aggregated, text_feat = self.dp_align(text_all_sorts, vid_feat)
            aligned_aggregated = torch.sigmoid(aligned_aggregated)

            num_vid = len(vid_feat)
            num_text = len(text_feat)

            
            # sequential edges
            adj_tensor = torch.zeros(num_vid+num_text,num_vid+num_text).to('cuda')
            for i in range(num_vid-1):
                adj_tensor[i][i+1] = 1
            for i in range(num_text-1):
                adj_tensor[i+num_vid][i+num_vid+1] = 1

            # cross-modal edges + sequential edges
            adj_tensor = cross_adj_matrix + adj_tensor

            
            # Prepare input for fine-grained transformer
            input = vid_feat
            input = torch.cat((input, text_feat), dim=0)
            learnable_node_expanded = self.learnable_node
            input = torch.cat([input, learnable_node_expanded], dim=0)
            input = self.pos_encoder_fine(input.unsqueeze(0)).squeeze(0)

            # Extend adjacency matrix for transformer for entity
            adj_tensor = torch.cat([adj_tensor, torch.zeros((1, adj_tensor.size(1)), device=adj_tensor.device)], dim=0)
            adj_tensor = torch.cat([adj_tensor, torch.zeros((adj_tensor.size(0), 1), device=adj_tensor.device)], dim=1)
            adj_tensor[-2][-1] = 1  # Add connection between learnable node and the graph
            adj_tensor = torch.where(adj_tensor == 0, -1e9, 0.0).to(dtype=torch.float64) # Generate mask


            # Fine-grained transformer output with mask
            output_fine = self.transformer_fine(input, mask=adj_tensor)
            output_fine = output_fine[-1]

            # Combine results from coarse and fine transformers
            output = self.fc(torch.cat((output_coarse, output_fine), dim=0))
            output = self.sigmoid(output)

            dp_preds.append(aligned_aggregated)
            map_preds.append(output)
            labels_ls.append(label)

        return torch.stack(dp_preds).view(-1), torch.stack(map_preds).view(-1), torch.stack(labels_ls)
 