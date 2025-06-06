import argparse


def Arguments():
    parser = argparse.ArgumentParser(description="egotv")

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--num_workers', type=int, default=0, help='workers for dataloaders')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training, validation, test; '
                                                                  'batch size is divided across the number of workers')
    # '''<command> --preprocess''' to set preprocess
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--visual_feature_extractor', type=str, default='clip', choices=['resnet','clip', 'coca', 'mvit'],
                        help='clip/coca features for video segments')
    parser.add_argument('--text_feature_extractor', type=str, default='clip', choices=['clip', 'coca', 'bert'],
                        help='clip/coca features for query arguments')
    parser.add_argument('--fp_seg', type=int, default=20, help='frames per segment')
    parser.add_argument('--sample_rate', type=int, default=2, help='video sub-sample rate')

    # data
    parser.add_argument('--data_path', type=str, default="data", help='data file path for training and testing')

    #log
    parser.add_argument('--log_name', type=str, default="egotv.txt", help='log file name')

    #ckpt
    parser.add_argument('--ckpt_name', type=str, default="checkpoint.pth", help='log file name')


    return parser.parse_args()
