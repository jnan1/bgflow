import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

'''
Input: 2d arrays [detection mask from prev frame, detection mask from prev prev, 
			  detection mask, confidence, entropy , lab]
Output: iou of current detection box with gt
'''
class BBoxMaskNet(nn.Module):
	def __init__(self, input_channels=6, out_channels=1):
		super(BBoxMaskNet, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(input_channels, 16, kernel_size=2, stride=2, padding=1),  # b, 16, 10, 10
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=4, stride=3),  # b, 16, 5, 5
			nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),  # b, 8, 3, 3
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1),  # b, 8, 3, 3
			# nn.ReLU(True),
			# nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.fc1 = nn.Linear(800, 64)
		self.fc2 = nn.Linear(64, 1)
		self.loss_fn = nn.MSELoss()

	def forward(self, x):
		x = self.encoder(x)
		x = x.view(-1, 800)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def criterion(self, pred, gt):
		loss = torch.abs(pred - gt)
		return loss