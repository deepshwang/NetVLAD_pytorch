from models.models import *
from params import *
import pdb
from math import ceil
from dataloader.dataset import *
from dataloader.transforms import *
from torch.utils.data import DataLoader, RandomSampler
import numpy as np

import h5py
import faiss


if __name__ == '__main__':
    args = argument_parser()

    # Pretrained(by imagenet) encoder
    encoder = NetVLAD(args).encoder.to(args.device)

    # Dataloader to extract descriptors from
    dataset = None
    if args.dataset == 'tokyo247':
        dataset = Tokyo247Database(args, T_TOKYO)

    # Number of images to randomly sample from
    nIm=ceil(args.nDescriptors/args.nPerImage)

    # Make dataloader that with random subsets of nIm images
    sampler = RandomSampler(dataset, replacement=True, num_samples=nIm)
    dataloader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=args.cache_batch_size, shuffle=False, sampler=sampler)

    # Extract descriptor, calculate correesponding centroids, and save them as h5py file
    cache_filename = './cache/centroids_descriptors_' + args.dataset + "_" + args.backbone + '.hdf5'
    with h5py.File(cache_filename, mode='w') as h5:
        encoder.eval()
        print("=============> Extracting Descriptors")
        dbFeat = h5.create_dataset("descriptors", [args.nDescriptors, args.feature_dim], dtype=np.float32)
        for i, (input, _, _, _) in enumerate(dataloader):
            input = input.to(args.device)
            descriptors = encoder(input).view(input.size(0), args.feature_dim, -1).permute(0, 2, 1)
            # descriptors.shape : (B, w * h , args.feature_dim)

            batchidx = i * args.cache_batch_size * args.nPerImage

            for idx in range(descriptors.size(0)):
                sample = np.random.choice(descriptors.shape[1], args.nPerImage, replace=False)
                startidx = batchidx + idx * args.nPerImage
                dbFeat[startidx:startidx + args.nPerImage, :] = descriptors[idx, sample, :].detach().cpu().numpy()

            del input, descriptors

            print ((i+1) * 100 * args.cache_batch_size, " / ", args.nDescriptors, 'COMPLETED')

        print("==============> Clustering the extracted descriptors")
        niter = 100
        kmeans = faiss.Kmeans(args.feature_dim, args.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
