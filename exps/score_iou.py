import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import os
import numpy as np

from dataloader.scoreDataset import ScoreDataset
from network.scoreNet import ScoreNet

def train(epoch, dataloader, net):
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	for t in range(epoch):
		total_loss = 0
		for i in range(len(dataloader)): #len(dataloader)
			example = dataloader[i]
			x, y = example['features'], example['gt']
			# print(y.shape)
			optimizer.zero_grad()
			out = net(x)
			loss = net.criterion(out, y)
			loss.backward()
			optimizer.step()
			total_loss += loss.data
			if i % 500 == 499:
				print('training loss {}'.format(total_loss/500))
				total_loss = 0
		eval(dataloader, net)
		torch.save(net.state_dict(), '../results/score_models/ckpt_{}.pth'.format(t))
			# for param in net.parameters():
  	# 			print(param.data)

def eval(dataloader, net):
	reset, total_iou = 0, 0
	N = dataloader.test_len()
	zero_count = 0
	# with torch.no_grad(): 
	for i in range(N):
		example = dataloader.get_test_item(i)
		x, y = example['features'], example['iou']
		out = net(x).data.cpu().numpy()
		pens = example['penalties']
		size_penalty, scale_penalty, cosine_score = pens[:,0], pens[:,1], pens[:,2]
		penalty = np.exp(-(scale_penalty * size_penalty - 1.) *0.04)
		pscore = penalty * out
		pscore = pscore * (1 - 0.44) + cosine_score * 0.44
		iou = y[np.argmax(out)]
		total_iou += iou
		reset += (iou == 0)
		if 0 in y:
			zero_count += 1
	print('has {} zeros in total'.format(zero_count))
	print('average iou = {} for {} resets in {} examples'.format(total_iou / N, reset, N))


if __name__ == '__main__':
	dataloader = ScoreDataset('/home/jianingq/vot/bbox_more_info/')
	net = ScoreNet(3, 1)
	net.cuda()
	# net.load_state_dict(torch.load('../results/models/ckpt_5.pth'))
	# eval(dataloader, net)
	train(500, dataloader, net)
