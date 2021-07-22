from dataset.tokyo import TokyoDataset, TokyoValTestDataset, TokyoTrainWholeDataset
from torch.utils.data import Dataset
import os
import pdb
import numpy as np
import pandas as pd
import faiss
import scipy.io
import h5py

from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torch



class TokyoTMTrainWholeDataset(TokyoTrainWholeDataset):
    def __init__(self, *args, **kwargs):
        super(TokyoTMTrainWholeDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyoTM/tokyoTM_train.mat', 
                                                                        dataset='tokyoTM', 
                                                                        split='train', 
                                                                        db_q=None)




class TokyoTMValQueryDataset(TokyoValTestDataset):
    """
    TokyoTM Query Dataset (Validation)
    """
    def __init__(self, *args, **kwargs):
        super(TokyoTMValQueryDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyoTM/tokyoTM_val.mat', 
                                                                       dataset='tokyoTM', 
                                                                       split='val', 
                                                                       db_q='query')
        self.utm = self.utmQ



class TokyoTMValDBDataset(TokyoValTestDataset):
    """
    TokyoTM DB Dataset (Validation)
    """
    def __init__(self, *args, **kwargs):
        super(TokyoTMValDBDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyoTM/tokyoTM_val.mat', 
                                                                   dataset='tokyoTM', 
                                                                   split='val', 
                                                                   db_q='database')
        self.utm = self.utmDb



class TokyoTMTrainQueryDataset(TokyoValTestDataset):
    """
    TokyoTM Q Dataset (Train) & For caching features during training
    """
    def __init__(self, *args, **kwargs):
        super(TokyoTMTrainQueryDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyoTM/tokyoTM_train.mat', 
                                                                       dataset='tokyoTM', 
                                                                       split='train', 
                                                                       db_q='query')
        self.utm = self.utmQ



class TokyoTMTrainDBDataset(TokyoValTestDataset):
    """
    TokyoTM DB Dataset (Train) & For caching features during training
    """
    def __init__(self, *args, **kwargs):
        super(TokyoTMTrainDBDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyoTM/tokyoTM_train.mat', 
                                                                       dataset='tokyoTM', 
                                                                       split='train', 
                                                                       db_q='database')
        self.utm = self.utmDb


