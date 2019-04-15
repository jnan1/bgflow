import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import os
import numpy as np
from dataloader.youtubeDataset import YoutubeDataset
from network.maskNet import BBoxMaskNet
# from coord_conv import coord_conv
# from utils import calculate_iou
from time import time
import sys
import cv2

sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')

from normal_loader import VOT

def train(epoch, dataloader, net, s=0):
	optimizer = optim.Adam(net.parameters(), lr=1e-4)
	# criterion = nn.MSELoss()
	vot = VOT
	count = 0
	log_step = 10
	
	for t in range(s, epoch):
		total_loss = 0
		inds = np.random.permutation(len(dataloader))
		for i, idx in enumerate(inds): #len(dataloader)
			try:
				example = dataloader[idx]
			except:
				print(idx)
				example = None
			if example == None:
				continue
			xs, y = example['feats'], example['gt']
			
			for inp_idx, inp in enumerate(xs):
				count += 1
				optimizer.zero_grad()
				out = net(inp)[0]
				loss = net.criterion(out, y[inp_idx])
				loss.backward()
				optimizer.step()
				total_loss += loss.data
				# if i % 50 == 0 and mask_idx % 50 == 0:
				# 	print(out, gt)
			if i % log_step == log_step - 1:
				print('training loss {}'.format(total_loss / count))
				count = 0
				total_loss = 0

		print('saving model ...', end='')
		torch.save(net.state_dict(), '../results/models/youtube_mask/ckpt_{}.pth'.format(t))
		print(' saved')
		eval(dataloader, net)

def eval(dataloader, net):
	reset, total_iou = 0, 0
	N = dataloader.test_len()
	# N = 10
	zero_count = 0
	# with torch.no_grad(): 

	print('evaluating on the test set...')
	for i in range(N):
		try:
			example = dataloader.get_test_item(i)
		except:
			print(i)
			example = None
		if example == None:
			continue


		inps, y = example['feats'], example['gt']
		# masks = example['masks']
		pred_ious = []
		# out_pth = example['pth'].replace('vot_info', 'out_result')
		# out_folder = out_pth[:out_pth.rfind('/')]
		# if not os.path.exists(out_folder):
			# os.makedirs(out_folder)
			# out = net(inp).data.cpu().numpy()[0]
			# pred_ious.append(out[0].copy())
		pred_ious = []
		for inp in inps:
			pred = net(inp)[0].data.cpu().numpy()
			pred_ious.append(pred)
		gt_iou = np.array(y.cpu().numpy())
		# pred_ious = np.array(pred_ious)
		# print('saving to ' + out_pth+'_pred.npy')
		# np.save(out_pth+'_pred.npy', pred_ious)
		# print(pred_ious)
		iou = gt_iou[np.argmax(np.array(pred_ious))]
		total_iou += iou
		reset += (iou == 0)
	# print('end')
	print('average iou = {} for {} resets in {} examples'.format(total_iou / N, reset, N))

def test_net(net, ds):
	# print(mask_ds[724]['feats'].shape)
	# print(mask_ds[0]['feats'].shape)
	x = ds[0]['feats']
	# y = ds[0]['gt_mask']
	# print(x.shape, y.shape)
	out = net(x)
	# print(out.shape)
	# criterion = nn.MSELoss()
	# loss = criterion(out[0], y)
	# print(loss)

def main():
	# ['/scratch/jianingq/vot_info'], 
	# 	'/home/jianingq/vot/bbox_more_info/',
	# 	'/scratch/jianingq/vot_data'
	mask_ds = YoutubeDataset()
	net = BBoxMaskNet()
	net.cuda()
	ckpt_num = 0
	#net.load_state_dict(torch.load('../results/models/youtube_mask/ckpt_{}.pth'.format(ckpt_num)))
	#eval(mask_ds, net)
	# test_net(net, mask_ds)
	print('start training')
	train(100, mask_ds, net, ckpt_num+1)

if __name__ == '__main__':
	main()