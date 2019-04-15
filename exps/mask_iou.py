import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import os
import numpy as np
from dataloader.maskDataset import MaskDataset
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
	optimizer = optim.Adam(net.parameters(), lr=1e-6)
	# criterion = nn.MSELoss()
	vot = VOT
	count = 0
	log_step = 100
	pred_ious, resets = 0, 0
	for t in range(s, epoch):
		total_loss = 0
		total_iou, total_resets = 0, 0
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
			
			outs = []
			for inp_idx, inp in enumerate(xs):
				count += 1
				optimizer.zero_grad()
				out = net(inp)[0]
				loss = net.criterion(out, y[inp_idx])
				loss.backward()
				optimizer.step()
				total_loss += loss.data
				outs.append(out.data)
				# if i % 50 == 0 and mask_idx % 50 == 0:
				# 	print(out, gt)
			outs = [x.cpu().numpy() for x in outs]
			# print(outs)
			pred_iou = y.cpu().numpy()[np.argmax(outs)]
			resets += pred_iou == 0
			pred_ious += pred_iou
			total_iou += pred_iou
			total_resets += (pred_iou == 0)
			if i % log_step == log_step - 1:
				print('training loss {}'.format(total_loss / count))
				print('avg iou {}, number of resets {} out of {}'.format(
					pred_ious / log_step, resets, log_step))
				count = 0
				total_loss = 0
				resets = 0
				pred_ious = 0


		print('saving model ...', end='')
		torch.save(net.state_dict(), '../results/models/train_mask_onethird/ckpt_{}.pth'.format(t))
		print('average iou = {} for {} resets in {} examples'.format(total_iou / len(inds), total_resets, len(inds)))
		# print(' saved')
		# eval(dataloader, net)

def eval(dataloader, net):
	reset, total_iou = 0, 0
	video_total_iou, video_N, video_reset = 0, 0, 0
	N = dataloader.test_len()
	# N = 10
	zero_count = 0
	# with torch.no_grad(): 

	prev_video_name = None
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
		gt_iou = np.array(y.cpu().numpy())
		for idx, inp in enumerate(inps):
			pred = net(inp)[0].data.cpu().numpy()
			pred_ious.append(pred)
			# print(pred[0], gt_iou[idx])
		
		# pred_ious = np.array(pred_ious)
		# print('saving to ' + out_pth+'_pred.npy')
		# np.save(out_pth+'_pred.npy', pred_ious)
		# print(np.array(pred_ious[:10]))
		# print(gt_iou[:10])
		iou = gt_iou[np.argmax(np.array(pred_ious))]
		total_iou += iou
		reset += (iou == 0)
		video_name, _ = dataloader._info_from_pth(example['pth'])
		if not video_name == prev_video_name:
			if video_N > 0:
				print('finish evaluating {}, avg iou: {}, reset: {}'.format(
					prev_video_name, video_total_iou / video_N, video_reset))
			print('evaluating video {}'.format(video_name))
			prev_video_name = video_name
			video_reset = 0
			video_N = 0
			video_total_iou = 0
		video_reset += (iou == 0)
		video_N += 1
		video_total_iou += iou
	# print('end')
	print('average iou = {} for {} resets in {} examples'.format(total_iou / N, reset, N))

def test_net(net, ds):
	# print(mask_ds[724]['feats'].shape)
	# print(mask_ds[0]['feats'].shape)
	inp = ds[0]['feats'][0]
	# inp = torch.cat(inp, 0)
	print(inp.shape)
	out = net(inp)
	print(out.shape)

def main():
	# ['/scratch/jianingq/vot_info'], 
	# 	'/home/jianingq/vot/bbox_more_info/',
	# 	'/scratch/jianingq/vot_data'
	mask_ds = MaskDataset()
	net = BBoxMaskNet(input_channels=6)
	net.cuda()
	ckpt_num = 30
	net.load_state_dict(torch.load('../results/models/train_mask_onethird/ckpt_{}.pth'.format(ckpt_num)))
	eval(mask_ds, net)
	# test_net(net, mask_ds)
	print('start training')
	np.random.seed(0)
	train(200, mask_ds, net, ckpt_num)

if __name__ == '__main__':
	main()