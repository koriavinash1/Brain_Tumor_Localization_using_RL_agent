import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict

from torchvision.models.densenet import model_urls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.models.resnet import model_urls as resnetmodel_urls
from tqdm import tqdm

class QNetwork(nn.Module):
	"""
		
	"""
	def __init__(self, num_inputs, num_actions):
		super(QNetwork, self).__init__()
		# num_inputs = number of feature outputs + 4 past actions 24
		# self.features     = nn.Sequential(OrderedDict([]))
		# self.features.add_module('layer1', nn.ReLU(nn.Linear(1048, 1024)))
		# self.features.add_module('layer2', nn.ReLU(nn.Linear(1024, 512)))
		# self.features.add_module('layer3', nn.ReLU(nn.Linear(512, 6)))


		self.layer1 = nn.Linear(1048,1024)
		self.layer2 = nn.Linear(1024, 512)
		self.layer3 =nn.Linear(512, 6)

	def forward(self, state, past_actions):
		x = torch.cat([state.float(), past_actions.float()], 1)
		actions= self.layer3(self.layer2(self.layer1(x)))
		# actions = self.features(x)
		return actions


class featureExtractor(nn.Module):
	"""
		
	"""
	def __init__(self):
		super(featureExtractor, self).__init__()
		self.net = torchvision.models.densenet121(pretrained=True)
		self.net = nn.Sequential(*list(self.net.features.children())[:-3])
		self.net.add_module('adaptive', nn.AdaptiveAvgPool2d((1,1)))

	def forward(self, x):
		x = self.net(x)
		shape = x.size()[1]
		x = x.view(-1, shape)
		return x


class combinedNetwork(nn.Module):
	"""
		combined net for 
	"""
	def __init__(self, ninputs, nactions):
		super(combinedNetwork, self).__init__()
		self.features = featureExtractor().cuda()
		self.Qnet     = QNetwork(num_inputs = ninputs, num_actions = 4).cuda()

	def forward(self, x, history_vec):
		# print (x.size(), history_vec.size())
		x = self.features(x)		
		x = self.Qnet(x, history_vec)
		return x
