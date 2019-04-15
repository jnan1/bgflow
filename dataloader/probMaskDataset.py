import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import numpy as np
import sys
import cv2

sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')

from normal_loader import VOT
from data_format import bbox_format
from utils import *

class ProbMaskDataset(Dataset):
	def __init__(self, 
		bbox_dir='/home/jianingq/vot/bbox_more_info/', 
		vot_root='/scratch/jianingq/vot_data/', 
		confidence_dir='/scratch/jianingq/backward_flow_confidence_vot/',
		mask_dir=['/scratch/jianingq/vot_info', ],
		tg_size=[100,100]):
		self.vot = VOT(vot_root)
		self.video_names = vot.get_video_names()
		self.bbox_dir = bbox_dir
		self.vot_root = vot_root
		self.confidence_dir = confidence_dir

		self.test_data_pth, self.train_data_pth = [], []
		self.gts = {}
		self.tg_size = tg_size

		nTest = len(self.video_names) // 3
		idx = np.random.permutation(range(len(self.video_names)))

		for i, video_name in enumerate(self.video_names):
			video_length = self.vot.get_frame_length(video_name)
			if i in idx[:nTest]:
				self.test_data_pth.append([(video_name, i) for i in range(video_length)])
			else:
				self.train_data_pth.append([(video_name, i) for i in range(video_length)])

	def __len__(self):
		return len(self.train_data_pth)

	def __getitem__(self, idx):
		video_name, frame_num = self.train_data_pth[idx]
		# print(pth)
		return self._load_pth(video_name, frame_num)
	
	def test_len(self):
		return len(self.test_data_pth)

	def get_test_item(self, idx):
		pth = self.test_data_pth[idx]
		return self._load_pth(pth)

	def get_vot_image(self, idx, test=False):
		pth = self.train_data_pth[idx] if not test else self.test_data_pth[idx]
		video_name, frame_num = self._info_from_pth(pth)
		img = self.vot.get_frames(video_name)[frame_num]
		img = cv2.resize(img, (self.tg_size[1], self.tg_size[0]))
		# cv2.rectangle(img,  (int(gt[0]),int(gt[1])), (int(gt[2]),int(gt[3])), (255,255, 0), 3)
		# cv2.imwrite('results/gt_mask.png', gt_mask[:,:,np.newaxis] * img)
		# cv2.imwrite('results/img.png', img)
		return img

	def _load_pth(self, video_name, frame_num):
		fs = ['entropy', 'confidence', 'fgmask', 'lab']
		entropy, confidence, fgmask, lab = [ np.load(pth + '_' + x +'.npy') for x in fs]
		if os.path.exists(pth + '_' + 'prev_fgmask.npy'):
			prev_fgmask = np.load(pth + '_' + 'prev_fgmask.npy')
		else:
			prev_fgmask = fgmask.copy()
		feats = np.dstack([entropy[:,:,0], confidence[:,:,0], 
			fgmask[:,:,0], prev_fgmask[:,:,0], lab[:,:,0]])
		reordered_feats = np.zeros([feats.shape[2], *self.tg_size])
		h, w = feats.shape[:2]
		for i in range(feats.shape[2]):
			reordered_feats[i] = cv2.resize(feats[:,:,i], (self.tg_size[1], self.tg_size[0]))
		video_name, frame_num = self._info_from_pth(pth)
		if frame_num == 0:
			return None
		bboxes, scores = self._load_bbox(video_name, frame_num)
		masks = []
		tg_w, tg_h = self.tg_size
		for j, bbox in enumerate(bboxes):
			x, y, bw, bh = bbox
			mask = np.zeros(self.tg_size)
			x, y, x1, y1 = np.array([x, y, x+bw, y+bh]) * np.array([tg_h / h, tg_w / w, tg_h / h, tg_w / w])
			mask[int(y):int(y1), int(x):int(x1)] = 1.0
			masks.append(Variable(torch.from_numpy(mask.copy()).float().cuda().unsqueeze(0).unsqueeze(0)))
		sample = {
			'feats': Variable(torch.from_numpy(reordered_feats).unsqueeze(0).float().cuda()),
			'gt': scores, 
			'masks': masks, 
			'pth': pth,
		}
		return sample

	def _load_bbox(self, video_name, frame_num):
		suffix = '{}.txt'.format(int(1e8+frame_num))[1:]
		pth = os.path.join(self.bbox_dir, video_name, suffix)
		f = open(pth, 'r')
		bboxes, scores = [], []
		for line in f:
			nums = line.split(' ')
			bboxes.append([float(x) for x in nums[-5:-1]])
			scores.append(Variable(torch.tensor(float(nums[-1])).cuda().float()))
		return bboxes, scores

if __name__ == '__main__':
	mask_ds = ProbMaskDataset(['/scratch/jianingq/vot_info'], 
		'/home/jianingq/vot/bbox_more_info/', 
		'/scratch/jianingq/vot_data')
	example = mask_ds.get_test_item(2)
	print(example['feats'].shape)
	# for key in example.keys():
	# 	print(key, example[key])
	print(example['pth'])
	print(len(example['gt']), example['gt'][0].shape)
	print(len(example['masks']), example['masks'][0].shape)
	print(example['feats'].shape)
	print(torch.cat((example['feats'], example['masks'][0]), 1).shape)
	print(len(mask_ds), mask_ds.test_len())