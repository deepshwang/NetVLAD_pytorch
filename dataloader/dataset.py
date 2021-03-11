from torch.utils.data import Dataset
import os
import random
from PIL import Image
import torch
import csv
import pdb
import numpy as np

class Tokyo247Queryset(Dataset):
    '''
    Tokyo 247 query_v2 used for TEST only (Used for evaluating NetVLAD in the paper)
    '''
    def __init__(self, args, transforms=None):
        self.rootdir = "/media/TrainDataset/tokyo247/247query_v2"
        self.queries = self._get_queries()
        self.transforms = transforms


    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        img_query = Image.open(self.queries[idx])
        if self.transforms is not None:
            img_query = self.transforms(img_query)
        label = self._get_label(self.queries[idx])
        return img_query, 0, 0, label


    def _get_queries(self):
        image_list = []
        for path, subdirs, files in os.walk(self.rootdir):
            for name in files:
                if name[-3:] == 'jpg':
                    image_list.append(os.path.join(path, name))

        return image_list

    def _get_label(self, img_query):
        labelfile = img_query[:-3] + 'csv'
        with open(labelfile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar="|")
            for row in reader:
                label = np.concatenate((np.array([np.float(row[-2])]), np.array([np.float(row[-1])])))
        return label





class Tokyo247Database(Dataset):
    """
    Tokyo24/7 Database dataset (DOES NOT INCLUDE QUERIES FOR TEST)
    """
    def __init__(self, args, transforms=None):
        """
        ARGS: Transforms
        Return: (query image, positive image, negative image)
        """
        self.rootdir = "/media/TrainDataset/tokyo247/database"
        self.queries = self._get_images()
        self.transforms = transforms
        self.nNeg = args.nNeg
        self.positives, self.negatives = self._get_triplets()



    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        img_query = Image.open(self.queries[idx])
        img_positive = Image.open(self.positives[idx])
        img_negatives = [Image.open(f) for f in self.negatives[idx]]

        if self.transforms is not None:
            img_query = self.transforms(img_query)
            img_positive = self.transforms(img_positive)
            img_negatives = [self.transforms(f) for f in img_negatives]

        label = self._get_label(self.queries[idx])

        return img_query, img_positive, img_negatives, label


    def _get_images(self):
        image_list = []
        for path, subdirs, files in os.walk(self.rootdir):
            for name in files:
                if name[-3:] == 'png':
                    image_list.append(os.path.join(path, name))

        return image_list

    def _get_triplets(self):
        positives = []
        negatives = []
        for query in self.queries:
            same_place_dir = "/".join(query.split("/")[:-1])
            angle_query = int(query.split("/")[-1].split("_")[-1].split(".")[0])
            left_or_right = 1 if random.random() < 0.5 else -1
            angle_positive = (angle_query + 30 * left_or_right) % 360
            if len(str(angle_positive)) == 2 :
                angle_positive = '0' + str(angle_positive)
            elif len(str(angle_positive)) == 1:
                angle_positive = '00' + str(angle_positive)
            else:
                angle_positive = str(angle_positive)
            positive = os.path.join(same_place_dir, query.split("/")[-1].split(".")[0][:-3]) + angle_positive + ".png"
            positives.append(positive)

            negative_set = []
            for i in range(self.nNeg):
                negativeFLAG = True
                while negativeFLAG:
                    negative = random.choice(self.queries)
                    if same_place_dir not in negative:
                        negativeFLAG = False

                negative_set.append(negative)

            negatives.append(negative_set)

        return positives, negatives

    def _get_label(self, img_query):
        labelfile = img_query[:-3] + 'csv'
        with open(labelfile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar="|")
            for row in reader:
                label = np.concatenate((np.array([np.float(row[-2])]), np.array([np.float(row[-1])])))
        return label

