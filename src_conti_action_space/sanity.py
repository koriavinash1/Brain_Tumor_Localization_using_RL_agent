import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict

a = torch.rand(2,3,32, 32)
hist = torch.rand(2, 24)

class QNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions):
		super(QNetwork, self).__init__()
		# num_inputs = number of feature outputs + 4 past actions 24
		self.features     = nn.Sequential(OrderedDict([]))
		self.features.add_module('layer1', nn.ReLU(nn.Linear(num_inputs, 1024)))
		self.features.add_module('layer2', nn.ReLU(nn.Linear(1024, 512)))
		self.features.add_module('layer3', nn.ReLU(nn.Linear(512, num_actions)))

	def forward(self, state, past_actions):
		x = torch.cat([state, past_actions], 1)
		actions = self.features(x)
		return actions


class featureExtractor(nn.Module):
	def __init__(self):
		super(featureExtractor, self).__init__()
		self.net = torchvision.models.densenet121(pretrained=True)
		self.net = nn.Sequential(*list(self.net.features.children())[:-1])
		self.net.add_module('adaptive', nn.AdaptiveAvgPool2d((1,1)))

	def forward(self, x):
		x = self.net(x)
		shape = x.size()[1]
		x = x.view(-1, shape)
		print (x.size())
		return x


class combinedNetwork(nn.Module):
	def __init__(self, ninputs, nactions):
		super(combinedNetwork, self).__init__()
		self.features = featureExtractor()
		self.Qnet     = QNetwork(num_inputs = ninputs, num_actions = nactions)

	def forward(self, x, history_vec):
		x = self.features(x)
		x = self.Qnet(x, history_vec)
		return x


net = combinedNetwork(1024, 6)
b = net(a, hist)
print(b)
