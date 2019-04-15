import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import numpy as np
import sys
import cv2

from utils import flow_warp, bbox
from dataloader.train_youtubebb.demo_youtubebb import generate_masks
import pickle
from scipy import misc

class YoutubeDataset(Dataset):
	def __init__(self, 
		data_root='/home/jianingq/bgflow/DaSiamRPN/code/jenny/dataloader/train_youtubebb',
		train_img_pth='train_youtubebb_image_python3.txt',
		train_label_pth='train_youtubebb_labels_python3.txt',
		test_img_pth='test_youtubebb_image_python3.txt',
		test_label_pth='test_youtubebb_labels_python3.txt',):
		np.random.seed(0)
		train_img_paths = pickle.load(open(os.path.join(data_root, train_img_pth), 'rb'))
		train_label_paths = pickle.load(open(os.path.join(data_root,train_label_pth),'rb'))

		# test_img_paths = pickle.load(open(os.path.join(data_root,test_img_pth), 'rb'))
		# test_label_paths = pickle.load(open(os.path.join(data_root,test_label_pth),'rb'))

		N = 20000
		idx = np.random.permutation(N)

		nTest = N//4
		# print(idx[nTest:][11615])
		self.train_img_pths = [train_img_paths[x] for x in idx[nTest:]]
		self.train_label_pths = [train_label_paths[x] for x in idx[nTest:]]

		# self.test_img_pths = test_img_paths[idx[:nTest]]
		# self.test_label_pths = test_label_paths[idx[:nTest]]
		self.test_img_pths = [train_img_paths[x] for x in idx[:nTest]]
		self.test_label_pths = [train_label_paths[x] for x in idx[:nTest]]
		self.tg_size = [128, 128]
		# return c_net, eval_out, session
		# self.c_net, self.eval_out, self.session = self.initialize_cnet()

	def __len__(self):
		return len(self.train_img_pths)

	def __getitem__(self, idx):
		return self._load_index(idx, self.train_img_pths, self.train_label_pths)

	def test_len(self):
		return len(self.test_img_pths)

	def get_test_item(self, idx):
		return self._load_index(idx, self.test_img_pths, self.test_label_pths)

	def _load_index(self, index, img_pths, label_pths):
		
		flows, entropies, labs, ious, det_scores, boxes = generate_masks(index, img_pths, 
			label_pths)
		# process information from previous frames
		fg_mask, prev_fg_mask = self._process_masks(flows[0], flows[1], 
											boxes[0], boxes[1],
											det_scores[0], det_scores[1])

		# entropy = self.normalize_entropy(entropies[1])
		entropy = entropies[1]
		masks = [entropy[:,:,0], 1-entropy[:,:,0], labs[1], fg_mask, prev_fg_mask]
		# rgb_img = cv2.imread(img_pths[index][2])
		# cv2.imwrite('results/fg_mask.png', fg_mask[:,:,np.newaxis]*img)
		# cv2.imwrite('results/prev_fg_mask.png', prev_fg_mask[:,:,np.newaxis]*img)
		# cv2.imwrite('results/entropy.png', masks[0][:,:,np.newaxis] * img)
		# cv2.imwrite('results/confidence.png', masks[1][:,:,np.newaxis] * img)
		# cv2.imwrite('results/lab.png', masks[2][:,:,np.newaxis] * img)
		# zoom in to the masks

		tg_w, tg_h = self.tg_size
		all_feats = []
		h, w = masks[0].shape[:2]
		det_scores = det_scores[2]
		for j, bbox in enumerate(boxes[2]):
			x, y, bw, bh = bbox
			if bw == 0 or bh == 0:
				print('zero size bounding box')
				continue
			mask = np.zeros([h,w])
			x, y, xp, yp = np.array([x, y, x+bw, y+bh])
			x, y, xp, yp = [int(k) for k in [x, y, xp, yp]]
			mask[y:yp, x:xp] = det_scores[j]

			x, y, xp, yp = self.zoom_out_det(x, y, xp, yp, mask)
			# mask = mask[:,:,np.newaxis] * rgb_img
			new_mask = cv2.resize(mask[y:yp, x:xp], (self.tg_size[1], self.tg_size[0]))
			feats = [new_mask]
			# if j < 10:
			# 	cv2.imwrite('results/{}.jpg'.format(j), new_mask)
			for i, mask in enumerate(masks):
				# mask = mask[:,:,np.newaxis] * rgb_img
				crop_mask = mask[y:yp, x:xp].copy()
				rsz_mask = cv2.resize(crop_mask, (self.tg_size[1], self.tg_size[0]))
				# if j < 10: 
				# 	cv2.imwrite('results/{}_{}.jpg'.format(j,i), rsz_mask)
				feats.append(rsz_mask)
			all_feats.append(Variable(torch.from_numpy(np.array(feats)).unsqueeze(0).float().cuda()))
		sample = {
			'feats': all_feats,
			'gt': Variable(torch.from_numpy(np.array(ious[2])).unsqueeze(1).float().cuda()), 
			'pth': img_pths[index],
		}
		return sample

	def _process_masks(self, flow21, flow32, boxes1, boxes2, scores1, scores2):

		# print([x.shape for x in [flow21, flow32, boxes1, boxes2, scores1, scores2]])
		mask1 = np.zeros([flow21.shape[0], flow21.shape[1]])
		mask1 = self._mask_from_boxes(boxes1, scores1, mask1)
		# cv2.imwrite('results/mask1.png', mask1 * 255)
		# # print(np.max(flow21), np.min(flow21), np.max(flow32), np.min(flow32))
		mask1 = flow_warp(mask1, flow21)
		# cv2.imwrite('results/mask1_warped1.png', mask1 * 255)
		prev_fg_mask = flow_warp(mask1, flow32)
		# cv2.imwrite('results/mask1_warped.png', prev_fg_mask * 255)
		# prev_fg_mask = mask1
		mask2 = np.zeros([flow21.shape[0], flow21.shape[1]])
		mask2 = self._mask_from_boxes(boxes2, scores2, mask2)
		# cv2.imwrite('results/mask2.png', mask2 * 255)
		fg_mask = flow_warp(mask2, flow32)
		# cv2.imwrite('results/mask2_warped.png', fg_mask * 255)
		# fg_mask = mask2
		return fg_mask, prev_fg_mask

	def _mask_from_boxes(self, bboxes, det_scores, mask):
		for bbox, score in zip(bboxes, det_scores):
			x1, y1, w, h = bbox
			x1, y1, w, h = int(x1), int(y1), int(w), int(h)
			if h == 0 or w == 0:
				continue
			gaussian_mask = self._gaussian_box(h, w)
			mask[y1:y1+h, x1:x1+w] = np.maximum(mask[y1:y1+h, x1:x1+w],
				score*gaussian_mask)
		return mask

	def _gaussian_box(self, w, h):
		'''
		outputs w by h array
		'''
		std = max(w,h) // 2
		x, y = np.mgrid[(-w//2+1):(w//2+1), (-h//2+1):(h//2+1)]
		g = np.exp(-((x**2 + y**2)/(2*std**2)))
		return g / np.max(g.ravel())
		# return np.ones([w,h])

	# def normalize_entropy(self, entropy, mean=0.0724658410441471, 
	# 	std=2.491323154407462):
	# 	return (entropy - mean) / std
	def zoom_out_det(self, x, y, xp, yp, mask, scale=2):
		xmin, ymin, xmax, ymax = x, y, xp, yp
		xmid, ymid = (xmin + xmax) // 2,  (ymin + ymax) // 2
		w, h = (xmax-xmin) * scale, (ymax-ymin) * scale
		xmin, xmax = max(xmid - w // 2, 0), min(xmid + w//2, mask.shape[1])
		ymin, ymax = max(ymid - h // 2, 0), min(ymid + h//2, mask.shape[0])

		x, y, xp, yp = [int(a) for a in [xmin, ymin, xmax, ymax]]
		return x, y, xp, yp

def process_masks():
	mask_ds = MaskDataset(data_dir='/scratch/jianingq/vot_info')
	# mask_ds.process(4952)
	# [4951, 13853, 4793]:
	for i in range(len(mask_ds)):
		mask_ds.process(i)
	for i in range(len(mask_ds)):
		mask_ds.process(i, prev=True)

def test_load_ds():
	mask_ds = YoutubeDataset()
	# mask_ds.process(4850)
	from time import time
	t0 = time()
	example = mask_ds[13462]
	print(time() - t0)
	t0 = time()
	example = mask_ds.get_test_item(83)
	print(time() - t0)
	# print(example['feats'].shape)
	# t1 = time()
	# print('1st example taken {}'.format(t1-t0))
	# for i in range(1, 10):
	# 	example = mask_ds[i]
	# 	t2 = time()
	# 	print('{} example taken {}'.format(i, t2 - t1))
	# 	t1 = t2
	# print(example['feats'].shape)
	# print(example['pth'])
	# print(len(example['gt']), example['gt'].shape)
	# print(len(example['masks']), example['masks'][0].shape)
	# print(example['feats'].shape)
	# print(torch.cat((example['feats'], example['masks'][0]), 1).shape)
	# print(len(mask_ds), mask_ds.test_len())

if __name__ == '__main__':
	np.random.seed(0)
	# process_masks()
	test_load_ds()
