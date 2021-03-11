import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Argument Parser for NetVLAD')

    ##### Training Configurations #####
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Batch=-size for training')

    parser.add_argument('--device',
                        type=str,
                        default='cuda')

    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='num_workers')

    parser.add_argument('--dataset',
                        type=str,
                        default='tokyo247',
                        choices=['tokyo247'],
                        help='dataset to train')

    parser.add_argument('--nNeg',
                        type=int,
                        default=10,
                        help='The number of negative samples per query')

    parser.add_argument('--cache_batch_size',
                        type=int,
                        default=256,
                        help='Batch size for caching')

    parser.add_argument('--cache_refresh_rate',
                        type=int,
                        default=1000,
                        help='number of queries after which feature cache is refreshed')

    parser.add_argument('--dbfeat_cache',
                        type=str,
                        default='./cache/dbfeat.h5py',
                        help='number of queries after which feature cache is refreshed')

    parser.add_argument('--data_parallel',
                        type=bool,
                        default=True,
                        help='whether to use data parallel (multi-gpu) to load model')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='whether to use data parallel (multi-gpu) to load model')

    parser.add_argument('--m',
                        type=float,
                        default=0.1,
                        help='margin fro triplet loss')

    parser.add_argument('--epochs',
                        type=int,
                        default=30,
                        help='number of epochs for training')

    parser.add_argument('--batch_iter',
                        type=int,
                        default=50,
                        help='number of epochs for training')

    parser.add_argument('--call_db',
                        type=str,
                        default=None,
                        help='directory of database feature extracted and saved as h5py file')

    parser.add_argument('--call_q',
                        type=str,
                        default=None,
                        help='directory of query feature extracted and saved as h5py file')


    ##### Model Configurations #####

    parser.add_argument('--num_clusters',
                        type=int,
                        default=64,
                        help='Number of clusters for aggregating feature representations')

    parser.add_argument('--normalize_feature',
                        type=bool,
                        default=True,
                        help='Whether to L2 normalize feature representations AND clusteres')

    parser.add_argument('--feature_dim',
                        type=int,
                        default=512,
                        help='Channel size of representations of convolutional block ResNet50: 512')

    parser.add_argument('--pretrained',
                        type=bool,
                        default=True,
                        help='whether to use pretrained parameters for CNN backbones')

    parser.add_argument('--backbone',
                        type=str,
                        default='vgg16',
                        choices=['resnet50', 'vgg16'],
                        help='Backbone model')

    parser.add_argument('--init_cent_desc_path',
                        type=str,
                        default='./cache/test.hdf5',
                        help='path of centroid / descriptor h5 data for initializing centroids and alpha')

    parser.add_argument('--nDescriptors',
                        type=int,
                        default=50000,
                        help='Number of descriptors to sample for initialization')

    parser.add_argument('--nPerImage',
                        type=int,
                        default=100,
                        help='number of descriptors to extract randomly from an image')






    args = parser.parse_args()

    return args