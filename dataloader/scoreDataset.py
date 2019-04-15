import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class ScoreDataset(Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.data_pth = []
		for video in os.listdir(self.data_dir):
			video_dir = os.path.join(self.data_dir, video)
			if os.path.isdir(video_dir):
				for frame in os.listdir(video_dir):
					if frame.endswith('txt'):
						self.data_pth.append(os.path.join(video_dir, frame))
		np.random.seed(0)
		idx = np.random.permutation(len(self.data_pth))
		nTest = len(self.data_pth)//3
		self.train_pth = [self.data_pth[x] for x in idx[nTest:]]
		self.test_pth = [self.data_pth[x] for x in idx[nTest:]]
		self.test_pth = self.train_pth

	def __len__(self):
		return len(self.train_pth)

	def __getitem__(self, idx):
		pth = self.train_pth[idx]
		# print(pth)
		return self._load_pth(pth)
	
	def test_len(self):
		return len(self.test_pth)

	
	def get_test_item(self, idx):
		pth = self.test_pth[idx]
		return self._load_pth(pth)

	def _load_pth(self, pth):
		f = open(pth, 'r')
		feats, ious = [], []
		for l in f:
			feat = l.strip().split()
			# print(feat)
			feats.append([float(x) for x in feat[:6]])
			ious.append([float(feat[-1])])
			assert(float(feat[-1]) >= 0)
		f.close()
		sample = {
			# 'features': Variable(torch.from_numpy(np.array(feats)[:,:3]).float().cuda()),
			'penalties': np.array(feats)[:,3:],
			'scores': np.array(feats),
			# 'gt': Variable(torch.from_numpy(np.array(ious)).float().cuda()),
			'iou': np.array(ious).ravel(),
		}
		return sample

def stats():
	dataloader = ScoreDataset('/home/jianingq/vot/bbox_more_info/')
	print(len(dataloader))
	N = 20
	N = min(len(dataloader), N)
	inds = np.random.permutation(N)
	# score_table = {}
	det_scores = np.zeros(N*100)
	gt_ious = np.zeros(N*100)
	for i in inds[:N]:
		example = dataloader[i]
		scores = example['scores']
		ious = example['iou'] 
		# key = (int(score[0]*100), int(score[1]*100), int(score[2]*100))
		# if key in score_table.keys():
		# 	score_table[key].append(example['iou'][idx])
		# else:
		# 	score_table[key] = [example['iou'][idx]]
		det_scores[i*100:(i*100+100)] = scores[:,0] 
		gt_ious[i*100:(i*100+100)] = ious
	from matplotlib import pyplot as plt
	plt.switch_backend('agg')
	plt.figure()
	plt.plot(det_scores, gt_ious, '*')
	plt.xlabel('det_scores')
	plt.ylabel('gt_ious')
	plt.savefig('corr.png')
	# keys = score_table.keys()
	# lengths = np.array([len(score_table[key]) for key in keys])

	# idx = np.argmax(lengths)
	# print(lengths[idx], list(keys)[idx])
	# if l > 1:
	# 	print(l)

if __name__ == '__main__':
	stats()

	# dataloader = ScoreDataset('/home/jianingq/vot/bbox_more_info/')
	# print(dataloader[0].keys())
	# print(dataloader[0]['features'].shape, dataloader[0]['gt'].shape)
	# print(dataloader[0]['penalties'].shape)
	# print(dataloader.get_test_item(0).keys())