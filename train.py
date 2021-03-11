from params import argument_parser
import torch
from dataloader.dataloader import *
from dataloader.transforms import *
from models.models import *
import pdb
import h5py
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom
import faiss

class Trainer():
    def __init__(self, args, train_dataloader, cache_dataloader, test_dataloader, model, criterion):
        self.args=args
        self.device = args.device
        self.model = None
        self.train_dataloader = train_dataloader
        self.cache_dataloader = cache_dataloader
        self.test_dataloader = test_dataloader
        if args.data_parallel:
            self.model = nn.DataParallel(model)
        else:
            self.model = model.to(self.device)
        self.optimizer = optim.Adam((filter(lambda p: p.requires_grad, self.model.parameters())), lr=args.lr)
        self.criterion = criterion.to(args.device)
        self.dbcache = None
        self.qcache = None

    def build_feat(self, dataloader, db=True):
        featname = "dbfeat_" if db else "qfeat_"
        with torch.no_grad():
            self.model.eval()
            now = str(datetime.now().time())
            if db:
                self.dbcache = "./cache/" + featname + args.dataset + "_" + args.backbone + "_" + now.split(".")[1] +".hdf5"
                cachename = self.dbcache
            else:
                self.qcache = "./cache/" + featname + args.dataset + "_" + args.backbone + "_" + now.split(".")[1] +".hdf5"
                cachename = self.qcache
            print("===============> Start Writing....")
            with h5py.File(cachename, mode='w') as h5:
                pool_size = args.feature_dim * args.num_clusters
                DBfeat = h5.create_dataset("features", [len(dataloader.dataset), pool_size], dtype=np.float32)
                DBlabel = h5.create_dataset("labels", [len(dataloader.dataset), 2], dtype=np.float32)
                with torch.no_grad():
                    for i, (img_query, _, _, label) in enumerate(dataloader):
                        img_query = img_query.to(args.device)
                        vlad = self.model(img_query)
                        end = min((i+1)*args.cache_batch_size, len(dataloader.dataset))
                        DBfeat[i * args.cache_batch_size:end, :] = vlad.detach().cpu().numpy()
                        DBlabel[i * args.cache_batch_size:end, :] = label
                        del img_query, vlad
                        print(str(i+1) + "/" + str(len(dataloader)))



    def loss_plotter(self, viz, loss_plot, loss_value, x):
        viz.line(X=x, Y=loss_value, win=loss_plot, update='append')


    def train(self):
        self.model.train()
        viz = Visdom()
        viz.close(env="main")
        loss_plt = viz.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))

        print("Training Start!")
        for e in range(args.epochs):
            print("======= EPOCH ", e, " =======")
            loss_tracker = 0
            for i, (query, positive, negatives, _) in enumerate(self.train_dataloader):
                B, C, H, W = query.shape
                loss = 0
                self.optimizer.zero_grad()
                vlad_query = self.model(query.to(args.device))
                vlad_positive = self.model(positive.to(args.device))

                for negative in negatives:
                    vlad_negative = self.model(negative.to(args.device))
                    loss += criterion(vlad_query, vlad_positive, vlad_negative)

                loss /= args.nNeg
                loss.backward()
                self.optimizer.step()
                del vlad_query, vlad_positive, vlad_negative

                loss_tracker +=loss.item()

                if (i+1) % args.batch_iter == 0:
                    print("==> Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(e, args.epochs, i+1,
                                                                      len(self.train_dataloader),
                                                                         loss_tracker/args.batch_iter), flush=True)
                    self.loss_plotter(viz, loss_plt, torch.Tensor([loss_tracker/args.batch_iter]), torch.Tensor([e + (i+1)/len(self.train_dataloader)]))
                    loss_tracker = 0

    def test(self, recallN=(1, 5, 10, 20)):
        print("=========> Building Features of DB and test Queries for evaluation" )
        self.model.eval()
        if self.args.call_db is None:
            self.build_feat(self.cache_dataloader, db=True)
        if self.args.call_q is None:
            self.build_feat(self.test_dataloader, db=False)

        print("=========> Building faiss index for evaluation")
        faiss_index = faiss.IndexFlatL2(self.args.feature_dim * self.args.num_clusters)
        dbcachename = self.dbcache if self.args.call_db is None else self.args.call_db
        qcachename = self.qcache if self.args.call_q is None else self.args.call_q

        with h5py.File(dbcachename, mode='r') as h5:
            print("DB feature called: ", dbcachename)
            DBfeat = h5.get("features")[...]
            DBlabel = h5.get("labels")[...]
        with h5py.File(qcachename, mode='r') as h5:
            print("Test query feature called: ", qcachename)
            Qfeat = h5.get("features")[...]
            Qlabel = h5.get("labels")[...]


        faiss_index.add(DBfeat)
        print("=====> Calculating recall @N")
        n_values = list(recallN)

        for n in n_values:
            _, predictions = faiss_index.search(Qfeat, n)
            pdb.set_trace()




if __name__ == '__main__':
    args = argument_parser()

    # Initialize model (centroid and alpha appropriately initialized with cache hdf5)
    model = NetVLAD(args)
    model.netvlad.init_parameters()
    model.to(args.device)

    # Load dataloaders
    dataloader = Tokyo247DatabaseDataloader(args, T_TOKYO)
    cache_dataloader = Tokyo247DatabaseDataloader(args, T_TOKYO, batch_size=args.cache_batch_size)
    test_dataloader = Tokyo247QuerysetDataloader(args, T_TOKYO, batch_size=args.cache_batch_size)

    # Loss function
    criterion = nn.TripletMarginLoss(margin=args.m**0.5, p=2, reduction='sum')

    # Trainer
    trainer = Trainer(args=args,
                      train_dataloader=dataloader,
                      cache_dataloader=cache_dataloader,
                      test_dataloader=test_dataloader,
                      model=model,
                      criterion=criterion)

    # Train
    trainer.test()

