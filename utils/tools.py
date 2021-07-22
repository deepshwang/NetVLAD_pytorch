import yaml
import faiss
from collections import defaultdict
import numpy as np
import pdb
from scipy.io import loadmat
from collections import namedtuple
import torch
import wandb
from PIL import Image, ImageOps


def import_yaml(fpath):
	with open(fpath) as stream:
		config = yaml.load(stream, Loader=yaml.FullLoader)
	return config


def save_checkpoint(e=None, model=None, recalls=None, filepath=None):
	torch.save({'e': e,
				'state_dict': model.state_dict(),
				'recalls': recalls,}, filepath)


def parse_dbStruct(path):
	dataset = path.split("/")[-1][:-4]

	if dataset == 'tokyo247':
		dbStruct = namedtuple('dbStruct', ['whichSet', 
		'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
		'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

		mat = loadmat(path)
		matStruct = mat['dbStruct'].item()

		whichSet = matStruct[0].item()

		dbImage = [f[0].item() for f in matStruct[1]]
		utmDb = matStruct[2].T

		qImage = [f[0].item() for f in matStruct[3]]
		utmQ = matStruct[4].T

		numDb = matStruct[5].item()
		numQ = matStruct[6].item()

		posDistThr = matStruct[7].item()
		posDistSqThr = matStruct[8].item()
		nonTrivPosDistSqThr = matStruct[9].item()

		return dbStruct(whichSet, dbImage, utmDb, qImage, 
				utmQ, numDb, numQ, posDistThr, 
				posDistSqThr, nonTrivPosDistSqThr)


	elif 'tokyoTM' in dataset:
		dbStruct = namedtuple('dbStruct', ['whichSet', 
		'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
		'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

		mat = loadmat(path)
		matStruct = mat['dbStruct'].item()

		whichSet = matStruct[0].item()

		dbImage = [f[0].item() for f in matStruct[1]]
		utmDb = matStruct[2].T

		qImage = [f[0].item() for f in matStruct[4]]
		utmQ = matStruct[5].T

		numDb = matStruct[7].item()
		numQ = matStruct[8].item()

		posDistThr = matStruct[9].item()
		posDistSqThr = matStruct[10].item()
		nonTrivPosDistSqThr = matStruct[11].item()

		return dbStruct(whichSet, dbImage, utmDb, qImage, 
				utmQ, numDb, numQ, posDistThr, 
				posDistSqThr, nonTrivPosDistSqThr)


def calculateRecalls(n_values, qUtm, dbUtm, predicted_idxes):
	knn_predicted_Utm = dbUtm[predicted_idxes]

	diff = knn_predicted_Utm - np.repeat(qUtm[:, np.newaxis, :], max(n_values), axis=1)

	L2_diff = np.linalg.norm(diff, axis=2)

	recall_list = []
	for n in n_values:
		diff_mat = L2_diff[:, :n]
		score_mat = diff_mat < 25
		total_in_count = np.sum(score_mat, axis=1) > 0
		if n == 1:
			success_idx = np.where(total_in_count == True)
		elif n == 25:
			fail_idx = np.where(total_in_count == False)
		recall_list.append([n, np.sum(total_in_count)/qUtm.shape[0]])

	return recall_list, success_idx[0], fail_idx[0], score_mat


def wandb_visualize_retrievals(indexes, Q_dataset, Db_dataset, predicted_idxes, score_mat, log_name=None, caption=None, n_examples=10, n_retrievals=5, border=10):
	qidx_sample = np.random.choice(indexes.shape[0], n_examples, replace=False)
	total_q_list = []
	for idx in qidx_sample:
		ret_list = []
		q_image = ImageOps.expand(Image.open(Q_dataset.images[idx]), border=border).resize((480, 640))
		ret_list.append(np.array(q_image))
		ret_idxes = predicted_idxes[idx, :n_retrievals]
		for i, ret_idx in enumerate(ret_idxes):
			# Reprsenting correct retrieval
			if score_mat[idx, i] == True:
				color = 'darkgreen'
			else:
				color =  'indianred'
			db_image = ImageOps.expand(Image.open(Db_dataset.images[ret_idx]), border=border, fill=color).resize((480, 640))
			ret_list.append(np.array(db_image))
		q_out = np.concatenate(ret_list, axis=1)
		total_q_list.append(q_out)
		
	total_q_dict = {log_name + str(i): wandb.Image(q,caption=caption + str(i)) for i, q in enumerate(total_q_list)}
	wandb.log(total_q_dict)
	
