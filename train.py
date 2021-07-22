from utils.argparse import argument_parser
from utils.tools import import_yaml, save_checkpoint, calculateRecalls, wandb_visualize_retrievals
from math import log10, ceil
import torch
from dataset.tokyo247 import *
from dataset.tokyoTM import *
from dataset.transforms import T_TOKYO
from models.models import *
import pdb
import h5py
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import faiss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import datetime
import wandb
from PIL import Image, ImageOps


class Trainer():
    def __init__(self, config, train_dataset, cacheDB_dataset, cacheQ_dataset, valQ_dataset, 
                 valDb_dataset, testQ_dataset, testDb_dataset, model, criterion):
        self.config = config
        self.wandb = config['train']['wandb']
        self.device = config['hardware']['device']
        self.epochs = config['train']['epochs']
        self.valFrequency = config['train']['valFrequency']
        self.train_batch_size = config['loader']['train_loader']['batch_size']
        self.model = model.to(self.device)
        self.train_dataset=train_dataset
        self.cacheDb_dataset=cacheDb_dataset
        self.cacheQ_dataset=cacheQ_dataset
        self.valQ_dataset=valQ_dataset
        self.valDb_dataset=valDb_dataset
        self.testQ_dataset=testQ_dataset
        self.testDb_dataset=testDb_dataset
        self.model = model.to(self.device)
        self.optimizer = optim.Adam((filter(lambda p: p.requires_grad, self.model.parameters())), lr=config['train']['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=config['train']['stepsize'],
                                                   gamma=config['train']['gamma'])
        self.criterion = criterion.to(self.device)
        self.cache_trainDb = config['cacheroot']['trainDb']
        self.cache_trainQ = config['cacheroot']['trainQ']
        self.val_cache = None


    def train(self):
        best_recall = 0
        not_improved = 0

        if self.config['train']['wandb']:
            wandb.init(project='netvlad-tokyotm', config=self.config)
            wandb.watch(self.model)

        recall_n1_cache = []
        recall_n5_cache = []
        recall_n10_cache = []
        for e in range(self.epochs):
            # self.train_epoch(e)
            # self.scheduler.step(e)

            if (e % self.valFrequency) == 0:
                if self.config['train']['val_dataset'] == 'tokyo247':
                    recalls = self.test(e, self.testQ_dataset, self.testDb_dataset)
                else:
                    recalls = self.test(e, self.valQ_dataset, self.valDb_dataset)
                   

                if recalls[2][1] > best_recall:
                    not_improved = 0
                    best_recall = recalls[2][1]
                    save_checkpoint(e=e, 
                                    model=self.model,
                                    recalls=recalls,
                                    filepath=self.config['statedict_root']['best'])
                    if self.wandb:
                        table = wandb.Table(data=recalls, columns=["@N", "Recall"])
                        wandb.log({'Best Recall Curve': wandb.plot.line(table, "@N", "Recall", title="Best Model's Recall @N" + " (" + self.config['train']['val_dataset'] + ")")})
                        wandb.log({'Recall @1': recalls[0][1]})
                        wandb.log({'Recall @5': recalls[4][1]})
                        wandb.log({'Recall @10': recalls[5][1]})

                else:
                    not_improved += 1

                save_checkpoint(e=e,
                                model=self.model,
                                recalls=recalls,
                                filepath=self.config['statedict_root']['checkpoint'])

                # Recall @1 Plot
                if self.wandb:
                    recall_n1_cache.append([e, recalls[0][1]])
                    table = wandb.Table(data=recall_n1_cache, columns=["Epoch", "Recall @1"])
                    wandb.log({"Recall @1 Changes over Training" : wandb.plot.line(table, "Epoch", "Recall @1",
                               title="Recall @1 Changes over Training" + " (" + self.config['train']['val_dataset'] + ")")})


                    recall_n5_cache.append([e, recalls[4][1]])
                    table = wandb.Table(data=recall_n5_cache, columns=["Epoch", "Recall @5"])
                    wandb.log({"Recall @5 Changes over Training" : wandb.plot.line(table, "Epoch", "Recall @5",
                               title="Recall @5 Changes over Training" + " (" + self.config['train']['val_dataset'] + ")")})

                    recall_n10_cache.append([e, recalls[5][1]])
                    table = wandb.Table(data=recall_n10_cache, columns=["Epoch", "Recall @10"])
                    wandb.log({"Recall @10 Changes over Training" : wandb.plot.line(table, "Epoch", "Recall @10",
                               title="Recall @10 Changes over Training" + " (" + self.config['train']['val_dataset'] + ")")})




    def train_epoch(self, epoch):
        epoch_loss = 0
        
        cacheRefreshRate = self.config['train']['cacheRefreshRate']

        if cacheRefreshRate > 0:
            subsetN = ceil(len(self.train_dataset) / cacheRefreshRate)
            subsetIdx = np.array_split(np.arange(len(self.train_dataset)), subsetN)
        else:
            subsetN = 1
            subsetIdx = [np.arange(len(self.train_dataset))]
        nBatches = (len(self.train_dataset) + self.train_batch_size - 1) // self.train_batch_size
        
        self.model.train()
        for subIter in range(subsetN):
            self.update_featCache()

            self.sub_train_dataset = Subset(dataset=self.train_dataset, indices=subsetIdx[subIter])

            train_dataloader = DataLoader(dataset=self.sub_train_dataset,
                                          num_workers=config['loader']['train_loader']['num_workers'],
                                          batch_size=config['loader']['train_loader']['batch_size'],
                                          shuffle=config['loader']['train_loader']['shuffle'],
                                          pin_memory=config['loader']['train_loader']['pin_memory'],
                                          collate_fn=self.train_dataset.collate_fn)


            batch_loss = 0
            for iteration, (query, positives, negatives, negCounts, indices) in enumerate(train_dataloader):
                # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
                # where N = batchSize * (nQuery + nPos + nNeg)
                if query is None: continue # in case we get an empty batch

                B, C, H, W = query.shape
                nNeg = torch.sum(negCounts)
                input_data = torch.cat([query, positives, negatives])

                input_data = input_data.to(self.device)
                vlad_encoding = self.model(input_data)

                vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

                self.optimizer.zero_grad()
                
                # calculate loss for each Query, Positive, Negative triplet
                # due to potential difference in number of negatives have to 
                # do it per query, per negative
                loss = 0
                for i, negCount in enumerate(negCounts):
                    for n in range(negCount):
                        negIx = (torch.sum(negCounts[:i]) + n).item()
                        loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

                loss /= nNeg.float().to(self.device) # normalise by actual number of negatives
                loss.backward()
                self.optimizer.step()
                del input_data, vlad_encoding, vladQ, vladP, vladN
                del query, positives, negatives

                batch_loss += loss.item()
                epoch_loss += batch_loss

                if iteration % 20 == 0 or nBatches <= 10:
                    print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                        nBatches, batch_loss), flush=True)
                    global_batch = (epoch*len(self.train_dataset) + subIter*len(self.sub_train_dataset) + iteration * self.train_batch_size) / len(self.train_dataset)
                    if self.config['train']['wandb']:
                        wandb.log({'Train Batch Loss': batch_loss})
                    batch_loss = 0

                    print('Allocated:', torch.cuda.memory_allocated())
                    print('Cached:', torch.cuda.memory_reserved())

            del train_dataloader, batch_loss
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            # remove(train_set.cache) # delete HDF5 cache

        avg_loss = epoch_loss / nBatches

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
                flush=True)


    def update_featCache(self):
        print('=====> Building Cache')
        self.model.eval()
        self._writeCache(self.cacheDb_dataset, self.cache_trainDb)
        print('=====> DB Cache Built')
        self._writeCache(self.cacheQ_dataset, self.cache_trainQ)
        print('=====> Query Cache Bulit')


    def _writeCache(self, cache_dataset, cache_path):
        with h5py.File(cache_path, mode='w') as h5:
            if self.config['model']['backbone'] == 'vgg16':
                pool_size = 512
            elif self.config['model']['backbone'] == 'resnet50':
                pool_size = 2048
            else:
                print("Model not specified!")
                pdb.set_trace() 

            pool_size *= config['model']['num_clusters']
            h5feat = h5.create_dataset("features", 
                    [len(cache_dataset), pool_size], 
                    dtype=np.float32)

            cache_loader_config = config['loader']['cache_loader']
            cache_dataloader = DataLoader(dataset=cache_dataset,
                                        num_workers=cache_loader_config['num_workers'],
                                        batch_size=cache_loader_config['batch_size'],
                                        shuffle=cache_loader_config['shuffle'],
                                        pin_memory=cache_loader_config['pin_memory'])

            with torch.no_grad():
                for iteration, (image, utm) in enumerate(cache_dataloader):
                    start_iter = iteration * cache_loader_config['batch_size']
                    end_iter = min( (iteration + 1) * cache_loader_config['batch_size'], len(cache_dataset))
                    image = image.to(self.device)
                    vlad_encoding = self.model(image)
                    try:
                        h5feat[start_iter:end_iter, :] = vlad_encoding.detach().cpu().numpy()
                    except TypeError:
                        pdb.set_trace()
                    del image, vlad_encoding


    def test(self, e, Q_dataset, Db_dataset, n_values=[1, 3, 5, 10]):
        print("=====> Evaluating")
        Q_dataloader = DataLoader(dataset=Q_dataset,
                                num_workers=self.config['loader']['cache_loader']['num_workers'], 
                                batch_size=self.config['loader']['cache_loader']['batch_size'],
                                shuffle=self.config['loader']['cache_loader']['shuffle'],
                                pin_memory=self.config['loader']['cache_loader']['pin_memory'])


        Db_dataloader = DataLoader(dataset=Db_dataset,
                                num_workers=self.config['loader']['cache_loader']['num_workers'], 
                                batch_size=self.config['loader']['cache_loader']['batch_size'],
                                shuffle=self.config['loader']['cache_loader']['shuffle'],
                                pin_memory=self.config['loader']['cache_loader']['pin_memory'])
        dbUtm = Db_dataset.utm
        qUtm = Q_dataset.utm

        self.model.eval()
        print("=====> Extracting Features for Indexes")
        with torch.no_grad():
            pool_size = self.config['model']['encoder_dim'] * self.config['model']['num_clusters']
            dbFeat = np.empty((len(Db_dataset), pool_size))
            for i, (images, utm) in enumerate(Db_dataloader):
                images = images.to(self.device)
                vlad_encoding = self.model(images)
                start_iter = i * self.config['loader']['cache_loader']['batch_size']
                end_iter = min((i+1) * self.config['loader']['cache_loader']['batch_size'], len(Db_dataset))
                dbFeat[start_iter: end_iter, :] = vlad_encoding.cpu().numpy()

            del Db_dataloader

            qFeat = np.empty((len(Q_dataset), pool_size))
            for i, (images, utm) in enumerate(Q_dataloader):
                images = images.to(self.device)
                vlad_encoding = self.model(images)
                start_iter = i * self.config['loader']['cache_loader']['batch_size']
                end_iter = min((i+1) * self.config['loader']['cache_loader']['batch_size'], len(Q_dataset))
                qFeat[start_iter: end_iter, :] = vlad_encoding.cpu().numpy()

            del Q_dataloader

            qFeat = qFeat.astype(np.float32)
            dbFeat = dbFeat.astype(np.float32)



        start_time = datetime.datetime.now()
        print('=====> Building faiss index')
        faiss_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(pool_size))
        faiss_index.add(dbFeat)

        # print('=====> Calculating recall @ N')
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 25]

        _, predicted_idxes = faiss_index.search(qFeat, max(n_values))
        end_time = datetime.datetime.now()
        print("Elapsed time for FAISS: ", end_time - start_time)


        recalls, success_idx, fail_idx, score_mat = calculateRecalls(n_values, qUtm, dbUtm, predicted_idxes)
        print(recalls)


        if self.wandb and e % 2 == 0:
            wandb_visualize_retrievals(success_idx, Q_dataset, Db_dataset, predicted_idxes, score_mat,
                                       log_name='Successful_Retrievals_Ex#',
                                       caption='Green - Correct, Red - Incorrect (Left ones are more confident)')

            wandb_visualize_retrievals(fail_idx, Q_dataset, Db_dataset, predicted_idxes, score_mat,
                                       log_name='Failed_Retrievals_Ex#',
                                       caption='Green - Correct, Red - Incorrect (Left ones are more confident)',
                                       n_examples=1)
        return recalls




if __name__ == '__main__':
    opt = argument_parser()
    config = import_yaml(opt.config_type)

    # Initialize model (centroid and alpha appropriately initialized with cache hdf5)
    model = NetVLAD(config)
    model.netvlad.init_parameters()

    # Load datasets
    # train_dataset = TokyoTMTrainWholeDataset(config, T_TOKYO)
    # cacheDb_dataset = TokyoTMTrainDBDataset(config, T_TOKYO)
    # cacheQ_dataset = TokyoTMTrainQueryDataset(config, T_TOKYO)
    # valQ_dataset = TokyoTMValQueryDataset(config, T_TOKYO)
    # valDb_dataset = TokyoTMValDBDataset(config, T_TOKYO)
    # testQ_dataset = Tokyo247QueryDataset(config, T_TOKYO)
    # testDb_dataset = Tokyo247DBDataset(config, T_TOKYO)

    train_dataset = TokyoTMTrainWholeDataset(config, T_TOKYO)
    cacheDb_dataset = TokyoTMTrainDBDataset(config, T_TOKYO)
    cacheQ_dataset = TokyoTMTrainQueryDataset(config, T_TOKYO)
    valQ_dataset = TokyoTMValQueryDataset(config, T_TOKYO)
    valDb_dataset = TokyoTMValDBDataset(config, T_TOKYO)
    testQ_dataset = Tokyo247QueryDataset(config, T_TOKYO)
    testDb_dataset = Tokyo247DBDataset(config, T_TOKYO)

    # Loss function
    m = config['train']['m']
    criterion = nn.TripletMarginLoss(margin=m**0.5, p=2, reduction='sum')

    # Trainer
    trainer = Trainer(config=config,
                      train_dataset=train_dataset,
                      cacheDB_dataset=cacheDb_dataset,
                      cacheQ_dataset=cacheQ_dataset,
                      valQ_dataset=valQ_dataset,
                      valDb_dataset=valDb_dataset,
                      testQ_dataset=testQ_dataset,
                      testDb_dataset=testDb_dataset,
                      model=model,
                      criterion=criterion)

    # Train
    trainer.train()

