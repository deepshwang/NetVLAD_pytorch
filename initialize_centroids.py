from models.models import *
from utils.argparse import argument_parser
from utils.tools import import_yaml
import pdb
from math import ceil
from dataloader.dataset import *
from dataloader.transforms import *
from torch.utils.data import DataLoader, RandomSampler
import numpy as np

import h5py
import faiss


if __name__ == '__main__':
    opt = argument_parser()
    config = import_yaml(opt.config_type)
    device = config['hardware']['device']
    # Pretrained(by imagenet) encoder
    encoder = NetVLAD(config).encoder.to(device)

    # Dataloader to extract descriptors from
    dataset = Tokyo247TrainQueryDataset(config, T_TOKYO)

    # Number of images to randomly sample from
    nDescriptors = config['data']['centroids']['nDescriptors']
    nPerImage = config['data']['centroids']['nPerImage']

    nIm=ceil(nDescriptors/nPerImage)

    # Make dataloader that with random subsets of nIm images
    sampler = RandomSampler(dataset, replacement=True, num_samples=nIm)
    batch_size = config['loader']['cache_loader']['batch_size']
    dataloader = DataLoader(dataset=dataset, 
                            num_workers=config['loader']['cache_loader']['num_workers'], 
                            batch_size=config['loader']['cache_loader']['batch_size'], 
                            shuffle=config['loader']['cache_loader']['shuffle'], 
                            sampler=sampler)

    # Extract descriptor, calculate correesponding centroids, and save them as h5py file
    centroid_cache_filename = config['data']['centroid_cache']
    encoder_dim = config['model']['encoder_dim']
    with h5py.File(centroid_cache_filename, mode='w') as h5:
        encoder.eval()
        print("=============> Extracting Descriptors")
        dbFeat = h5.create_dataset("descriptors", [nDescriptors, encoder_dim], dtype=np.float32)
        for i, (input, _) in enumerate(dataloader):
            input = input.to(device)
            descriptors = encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)
            # descriptors.shape : (B, w * h , args.feature_dim)

            batchidx = i * batch_size * nPerImage
            for idx in range(descriptors.size(0)):
                sample = np.random.choice(descriptors.shape[1], nPerImage, replace=False)
                startidx = batchidx + idx * nPerImage
                dbFeat[startidx:startidx + nPerImage, :] = descriptors[idx, sample, :].detach().cpu().numpy()

            del input, descriptors

            print ((i+1) * 100 * batch_size, " / ", nDescriptors, 'COMPLETED')

        print("==============> Clustering the extracted descriptors")
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, config['model']['num_clusters'], niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
