from dataloader.dataset import *
from torch.utils.data.dataloader import DataLoader

def Tokyo247DatabaseDataloader(args, T, batch_size=None):
    dataset = Tokyo247Database(args, T)
    if batch_size == None:
        batch_size = args.batch_size
    dataloader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dataloader

def Tokyo247QuerysetDataloader(args, T, batch_size=None):
    dataset = Tokyo247Queryset(args, T)
    if batch_size == None:
        batch_size = args.batch_size
    dataloader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dataloader