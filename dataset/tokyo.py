from torch.utils.data import Dataset
import os
import random
from PIL import Image
import torch
import csv
import pdb
import numpy as np
import pandas as pd
import faiss
import scipy.io
import h5py

from utils.tools import import_yaml, parse_dbStruct
from sklearn.neighbors import NearestNeighbors

class TokyoDataset(Dataset):
    '''
    Base Dataset Class for Tokyo247 and TokyoTM
    '''
    def __init__(self, config=None, transforms=None, dbStruct_path=None, dataset=None, split=None, db_q=None):
        """
        Args
            config: Configuration
            transforms: torch transform
            dbStruct_path: .mat dbStruct file path to parse
            dataset: TokyoTM / Tokyo 247
            split: train / test 
            db_q: whether the dataset is database or query
        """
        self.config = config
        self.dataset = dataset
        self.split = split
        self.db_q = db_q
        self.datasetroot = config['dataroot'][dataset]
        dbStruct= parse_dbStruct(dbStruct_path)
        self.qImage = dbStruct.qImage
        self.dbImage = [img for img in dbStruct.dbImage if img not in self.qImage] 
        nonoverlap_idx = np.array([i for i in range(len(dbStruct.dbImage)) if dbStruct.dbImage[i] not in self.qImage])
        self.utmDb = dbStruct.utmDb[nonoverlap_idx, :]
        self.utmQ = dbStruct.utmQ
        self.numDb = len(self.dbImage)
        self.numQ = dbStruct.numQ
        self.posDistThr = dbStruct.posDistThr 
        self.posDistSqThr = dbStruct.posDistSqThr
        self.nonTrivPosDistSqThr = dbStruct.nonTrivPosDistSqThr
        self.transforms = transforms



class TokyoValTestDataset(TokyoDataset):
    def __init__(self, *args, **kwargs):
        super(TokyoValTestDataset, self).__init__(*args, **kwargs)
        """
`       Base Dataset Class for Validation (TokyoTM val) & Test (Tokyo247) 
        """
        self.images = self._get_images()
        self.utm = None


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        utm = self.utm[idx, :]

        return img, utm

    def _get_images(self):
        if self.db_q is not None:
            img_list = []

            # This must be a test set (Tokyo 247)
            if self.dataset == 'tokyo247':
                root = os.path.join(self.datasetroot, self.db_q)
                if self.db_q == 'database':
                    for f in self.dbImage:
                        f_dir = os.path.join(root, f[:-3] + 'png')
                        img_list.append(f_dir)
                    return img_list

                elif self.db_q == '247query_v2':
                    for f in self.qImage:
                        f_dir = os.path.join(root, f)
                        img_list.append(f_dir)
                    return img_list

            # This must be validation set (Tokyo TM)
            elif self.dataset == 'tokyoTM':
                root = self.datasetroot
                if self.db_q == 'database':
                    for f in self.dbImage:
                        f_dir = os.path.join(root, f)
                        img_list.append(f_dir)
                    return img_list
                elif self.db_q == 'query':
                    for f in self.qImage:
                        f_dir = os.path.join(root, f)
                        img_list.append(f_dir)
                    return img_list

        else:
            return None


class TokyoTrainWholeDataset(TokyoDataset):
    """
    TokyoTM Whole (Q + DB) (Train)
    """
    def __init__(self, *args, **kwargs):
        super(TokyoTrainWholeDataset, self).__init__(*args, **kwargs)

        self.nNegSample = self.config['data']['nNegSample']
        self.nNeg = self.config['data']['nNeg']
        self.margin = self.config['data']['margin']

        
        print("=====> Dataset initialization with non-trivial positives and potential negatives")
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.utmQ,
                radius=self.nonTrivPosDistSqThr**0.5, 
                return_distance=False))

        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)

        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        potential_positives = knn.radius_neighbors(self.utmQ, radius=self.posDistThr, return_distance=False)


        del knn
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.utmDb.shape[0]),
                                                         pos, assume_unique=True))
        
        # filepath of HDF5 containing feature vectors for images 
        self.Qcache = self.config['cacheroot']['trainQ']  
        self.Dbcache = self.config['cacheroot']['trainDb']

        self.negCache = [np.empty((0,)) for _ in range(self.numQ)]
        print("=====> Dataset initialization done")



    def __len__(self):
        return len(self.qImage)

    def __getitem__(self, index):

        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.Qcache, mode='r') as Qh5:
            with h5py.File(self.Dbcache, mode='r') as Dbh5: 
                Qh5feat = Qh5.get("features")
                Dbh5feat = Dbh5.get("features")
                qFeat = Qh5feat[index]
                posFeat = Dbh5feat[self.nontrivial_positives[index].tolist()]

                faiss_index = faiss.IndexFlatL2(posFeat.shape[1])
                faiss_index.add(posFeat)
                dPos, posNN = faiss_index.search(np.expand_dims(qFeat, axis=0), 2)

                dPos = dPos.reshape(-1)[1]
                dPos = dPos.item()
                posIndex = self.nontrivial_positives[index][posNN[0][1]].item()

                negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
                negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

                negFeat = Dbh5feat[sorted(negSample.tolist())]

                faiss_index = faiss.IndexFlatL2(negFeat.shape[1])
                faiss_index.add(negFeat.copy(order='C').astype(np.float32))

                dNeg, negNN = faiss_index.search(np.expand_dims(qFeat, axis=0), self.nNeg*10)

                dNeg = dNeg.reshape(-1)
                negNN = negNN.reshape(-1)

                # try to find negatives that are within margin, if there aren't any return none
                violatingNeg = dNeg < dPos + self.margin**0.5
         
                if np.sum(violatingNeg) < 1:
                    #if none are violating then skip this query
                    return None

                negNN = negNN[violatingNeg][:self.nNeg]
                negIndices = negSample[negNN].astype(np.int32)
                self.negCache[index] = negIndices

        if self.dataset == 'tokyoTM':
            query = Image.open(os.path.join(self.datasetroot, self.qImage[index]))
            positive = Image.open(os.path.join(self.datasetroot, self.dbImage[posIndex]))
        else:
            query = Image.open(os.path.join(self.datasetroot, '247query_v2', self.qImage[index]))
            positive = Image.open(os.path.join(self.datasetroot, 'database', self.dbImage[posIndex][:-3] + "png"))           

        if self.transforms is not None:
            query = self.transforms(query)
            positive = self.transforms(positive)

        negatives = []
        for negIndex in negIndices:
            if self.dataset == 'tokyoTM':
                negative = Image.open(os.path.join(self.datasetroot, self.dbImage[negIndex]))
            else:
                negative = Image.open(os.path.join(self.datasetroot, 'database', self.dbImage[negIndex][:-3] + "png"))                
            if self.transforms is not None:
                negative = self.transforms(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex] + negIndices.tolist()    

    def collate_fn(self, batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
        
        Args:
            data: list of tuple (query, positive, negatives). 
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """

        batch = list(filter (lambda x:x is not None, batch))
        if len(batch) == 0: return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = torch.utils.data.dataloader.default_collate(query)
        positive = torch.utils.data.dataloader.default_collate(positive)
        negCounts = torch.utils.data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        import itertools
        indices = list(itertools.chain(*indices))

        return query, positive, negatives, negCounts, indices
