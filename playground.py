from utils.argparse import argument_parser
from utils.tools import import_yaml, parse_dbStruct
import pdb
import h5py
import scipy.io
from dataset.tokyo import *

if __name__ == '__main__':
	opt = argument_parser()
	config = import_yaml(opt.config_type)


	# Qdataset = Tokyo247QueryDataset(config['data'])
	# DBdataset = Tokyo247DBDataset(config['data'])
	
	# for i in range(len(DBdataset)):
	# 	query, positive, negatives, label = DBdataset[i]
	# 	pdb.set_trace()
	# 	print("showing image")
	# 	query.show()
	# 	positive.show()
	# 	negatives[1].show()
	# 	pdb.set_trace()

	dbStruct = parse_dbStruct('/media/TrainDataset/tokyo247/tokyo247.mat')
	DBdataset = Tokyo247DatabaseDataset(config=config)
	qDataset = Tokyo247QueryDataset(config=config)
	pdb.set_trace()