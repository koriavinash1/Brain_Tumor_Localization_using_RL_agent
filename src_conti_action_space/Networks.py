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


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class DPGNetwork(nn.Module):
	"""
		
	"""
	def __init__(self, num_inputs = 1024, num_actions = 2):
		super(DPGNetwork, self).__init__()
		# num_inputs = number of feature outputs + 4 past actions 24
		
		self.features     = nn.Sequential(OrderedDict([]))
		# self.features.add_module('layer1', nn.Linear(1024, 1024))
		# self.features.add_module('tanh', nn.ELU())
		self.features.add_module('layer2', nn.Linear(1024, 512))
		self.features.add_module('tanh', nn.ELU())
		self.features.add_module('layer3', nn.Linear(512, num_actions))
		self.features.add_module('tanh', nn.Tanh())

	def forward(self, state, past_actions):
		x = torch.cat([state.float(), past_actions.float()], 1)
		# actions= self.layer3(self.layer2(self.layer1(x)))
		actions = self.features(x)
		#print (x)
		return actions

class DVNetwork(nn.Module):
	"""
	Deep Value Network
	"""
	def __init__(self, num_inputs = 1024, num_actions = 2, hidden_size = 512, init_w=3e-3):
		super(DVNetwork, self).__init__()

		self.linear1 = nn.Linear(num_inputs+num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state, action):
		# print (state.size(), history.size(), action.size())
		x = torch.cat([state, action], 1)
		x = F.elu(self.linear1(x))
		x = F.elu(self.linear2(x))
		x = self.linear3(x)
		return x

class featureExtractor(nn.Module):
	"""
	Deep feature extractor modle
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


class ValueNetwork(nn.Module):
	"""
	Deep featureextractor + Value network
	"""
	def __init__(self):
		self.features = featureExtractor()
		self.vnet     = DVNetwork()

	def forward(self, state, policy):
		# state >> image, torch variable
		features = self.features(state)
		value    = self.vnet(features, policy)
		return value

	def getValue(self, state):
		# state >> numpy array
		state = troch.FloatTensor(state).to(device)
		value = self.forward(state)
		print (value.detach().cpu().data.numpy().shape)
		return value.detach().cpu().data.numpy()[0,0]


class PolicyNetwork(nn.Module):
	"""
	Deep featureextractor + policy network
	"""
	def __init__(self):
		self.features = featureExtractor()
		self.dpg      = DPGNetwork()

	def forward(self, state):
		# state >> image, torch variable
		features = self.features(state)
		actions  = self.dpg(features)
		return actions

	def getAction(self, state):
		# state >> numpy nD array
		state = torch.FloatTensor(state).to(device)
		action= self.forward(action)
		print (action.detach().cpu().data.numpy().shape)
		return action.detach().cpu().data.numpy()[0,0]

#===================================================================================================
class GLNFeatureExtractor(nn.Module):
	"""
	
	"""
	def __init__(self, isTrained = True, num_channel=3):
		super(GLNFeatureExtractor, self).__init__()
		model_urls['densenet121'] = model_urls['densenet121'].replace('https://', 'http://')
		self.first_conv  =nn.Sequential(nn.BatchNorm2d(num_channel),nn.Conv2d(num_channel, 3, kernel_size=3, padding=1))
		self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
		self.features    = self.densenet121.features

	def forward(self, x):
		x = self.first_conv(x)
		x = self.features(x)
		x = nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
		# x = self.classifier(x)
		return x

class combinedNetwork(nn.Module):
	"""
		combined net for 
	"""
	def __init__(self, ninputs=1024, nactions = 2, history_count = 4):
		super(combinedNetwork, self).__init__()
		self.features = GLNFeatureExtractor().to(device) # ? x 1024
		self.dpg     = DPGNetwork(num_inputs = ninputs, num_actions = nactions, history_count = history_count).to(device)
		self.value   = DVNetwork().to(device)

	def forward(self, x, history_vec, action_ = None):
		x = x.float()
		history_vec = history_vec.float()
		x = self.features(x)		
		try :
			action_ == None
			policy = self.dpg(x, history_vec)
		except :
			policy = action_
		value  = self.value(x, history_vec, policy)
		return policy.view(-1, 2), value


if __name__ == "__main__":
	a   = torch.autograd.Variable(troch.rand(2, 9, 240, 240))
	h   = torch.autograd.Variable(torch.rand(2, 16))
	net = combinedNetwork()
	b   = net(a, h) 
	print (b.size())
