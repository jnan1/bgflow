import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class ScoreNet(nn.Module):
	def __init__(self, inDim, outDim):
		super(ScoreNet, self).__init__()
		self.fc1 = nn.Linear(inDim, 50)
		self.fc3 = nn.Linear(50, outDim)
		# self.softmax = nn.Softmax(0)
		self.loss_fn = nn.MSELoss()
	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		# x = F.sigmoid(self.fc2(x))
		x = F.softmax(self.fc3(x), dim=0)
		# x = self.fc1(x)
		return x

	def criterion(self, pred, gt):
		gt_iou = torch.max(gt)
		pred_iou = (pred * gt).sum()
		return torch.abs(pred_iou - gt_iou)
		# return self.loss_fn(pred, gt)