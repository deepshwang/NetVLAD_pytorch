from dataloader.dataloader import *
from dataloader.transforms import *
from params import argument_parser
import pdb

if __name__ == '__main__':
    args = argument_parser()
    dataloader = Tokyo247QuerysetDataloader(args, T_TOKYO)

    for i, (query, label) in enumerate(dataloader):
        pdb.set_trace()