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

class replayBuffer(object):
	"""
		
	"""
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer   = []
		self.position = 0

	def sample(self, batch_size=1):
		# random sampling for training Q network
		batch = random.sample(self.buffer, batch_size)
		state, action, hist_vec, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, hist_vec, reward, next_state, done

	def push(self, _vec):
		# _vec:[state, action, reward, next_state, done]
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = _vec
		self.position = (self.position + 1) % self.capacity
		pass

	def __len__(self):
		return len(self.buffer)


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

def calculate_iou(img_mask, gt_mask):
	"""
		for reward
	"""
	img_mask = np.uint8(img_mask)
	gt_mask  = np.uint8(gt_mask)
	# print (np.unique(img_mask), np.unique(gt_mask))
	img_and = np.sum((img_mask > 0)*(gt_mask > 0))
	img_or = np.sum((img_mask > 0)) + np.sum((gt_mask > 0))
	iou = 2.0 * float(img_and)/(float(img_or) + 1e-3)
	return iou


def calculate_overlapping(img_mask, gt_mask):
	"""
	"""
	gt_mask *= 1.0
	img_and = cv2.bitwise_and(img_mask, gt_mask)
	j = np.count_nonzero(img_and)
	i = np.count_nonzero(gt_mask)
	overlap = float(j)/(abs(float(i)) + 1e-3)
	return overlap


class Agent(object):
	def __init__(self, max_steps=6, max_frames=5000, epsilon=1):
		"init actions"
		super(Agent, self).__init__()
		self.state       = None
		self.states      = []
		self.action      = 0
		self.reward      = 0
		self.cum_reward  = 0
		self.max_steps   = max_steps
		self.max_frames  = max_frames
		self.epsilon     = epsilon
		self.curr_step   = 0
		self.curr_frame  = 0
		self.gamma       = 0.1 # discount factor
		self.iou_thresh  = 0.5
		self.prev_iou    = 0
		self.memory_capacity    = 1500
		self.terminal_reward    = 3
		self.momentum_reward    = 1
		self.number_of_actions  = 6
		self.history_of_actions = 4 # number of actions to be used in QNetwork
		self.history_vec = np.zeros((self.history_of_actions, self.number_of_actions))
		self.exp_memory  = replayBuffer(self.memory_capacity)
		self.scale_subregion = float(3)/4
		self.scale_mask = float(1)/(self.scale_subregion*4)
		self.cum_rewards = []
		self.done = False

		self.cnet       = combinedNetwork(1024 + self.history_of_actions*self.number_of_actions, 
							self.number_of_actions) # combined network fro training
		# data= torch.autograd.Variable(torch.rand(1,3,240,240).cuda())
		# his_vec =   torch.autograd.Variable(torch.rand(1,24).cuda())
		# pred = self.cnet(data,his_vec)
		# print ('i am here')
		# print (pred.size())

	def reset(self):
		"resets action and steps..."
		self.state       = None
		self.states      = []
		self.region_masks= []
		self.actions     = []
		self.rewards     = []
		self.ious        = []
		self.action      = 1
		self.reward      = 0
		self.cum_reward  = 0
		self.curr_step   = 0
		self.curr_frame  = 0
		self.prev_iou    = 0
		self.curr_iou    = 0
		self.offset      = (0, 0)
		self.size_mask   = (224, 224)
		# self.exp_memory  = replayBuffer(self.memory_capacity)
		self.history_vec = np.zeros((self.history_of_actions, self.number_of_actions))
		self.done        = False

	def buffer_reset(self):
		"""
		"""
		self.exp_memory  = replayBuffer(self.memory_capacity)

	def step(self, state, gt_mask):
		"""Each step to update reward, state and action:
		Action space:
			>> action 1: no motion
			>> action 2: right motion
			>> action 3: left motion
			>> action 4: diag motion 0.5 step
			>> action 5: diag motion 0.4 step
			>> action 6: terminal action...
		"""
		if self.curr_step == 0: self.prev_mask = np.ones_like(gt_mask)
		self.curr_state = state
		self.gt_mask = gt_mask
		self.curr_step += 1
		self.region_mask = np.zeros_like(self.gt_mask)
		self.action = self.get_action()
		self.state = torch.zeros_like(state)
		
		if self.action == self.number_of_actions:
			self.reward = self.get_reward(terminal=True)
			self.done = True
			self.cum_rewards.append(self.cum_reward)
			self.region_mask = self.prev_mask
			
		else:
			if self.action == 1:
				offset_aux = (0, 0)
			
			elif self.action == 2:
				offset_aux = (0, self.size_mask[0] * self.scale_subregion/2)
				
			elif self.action == 3:
				offset_aux = (self.size_mask[0] * self.scale_subregion/2, 0)

			elif self.action == 4:
				offset_aux = (self.size_mask[0] * self.scale_mask, self.size_mask[1] * self.scale_mask)

			elif self.action == 5:
				offset_aux = (self.size_mask[0] * self.scale_mask/2., self.size_mask[1] * self.scale_mask/2.)
			
			self.offset = (int(self.offset[0] + offset_aux[0]), int(self.offset[1] + offset_aux[1]))

			# self.state = self.curr_state[:, :, int(offset_aux[0]):int(offset_aux[0] + size_mask[0]), int(offset_aux[1]):int(offset_aux[1] + size_mask[1])]
			self.region_mask[int(self.offset[0]):int(self.offset[0] + self.size_mask[0]), int(self.offset[1]):int(self.offset[1] + self.size_mask[1])] = 1
			self.prev_mask = self.region_mask
			self.size_mask = (int(self.size_mask[0] * self.scale_subregion), int(self.size_mask[1] * self.scale_subregion))
			self.reward  = self.get_reward()
			
				
		self.state[:,0,:,:] = self.curr_state[:,0,:,:]*torch.from_numpy(self.region_mask).float()
		self.state[:,1,:,:] = self.curr_state[:,1,:,:]*torch.from_numpy(self.region_mask).float()
		self.state[:,2,:,:] = self.curr_state[:,2,:,:]*torch.from_numpy(self.region_mask).float()

		self.states.append(self.curr_state)
		self.region_masks.append(self.region_mask)
		self.actions.append(self.action)
		self.rewards.append(self.reward)
		self.ious.append(self.curr_iou)

		self.prev_iou = self.curr_iou
		self.cum_reward += self.reward
		return self.curr_state, self.action, self.history_vec, self.reward, self.state, self.done

	def get_reward(self, terminal=False):

		if not terminal:
			self.curr_iou = calculate_iou(self.region_mask, self.gt_mask)
			if self.curr_iou <= self.prev_iou:
				reward = -1*self.momentum_reward
			else:
				reward = 1*self.momentum_reward
		else:
			self.curr_iou = calculate_iou(self.prev_mask, self.gt_mask)
			if self.curr_iou > self.iou_thresh:
				reward = 1*self.terminal_reward
			else:
				reward = -1*self.terminal_reward
		
		# reward = 100*(self.curr_iou - self.prev_iou)
		return reward

	def update_replay_buffer(self, _vec):
		"push vec.  [old_state, action, reward, new_state, done]"
		self.exp_memory.push(_vec)
		pass

	def get_action(self):
		"returns action for a given state..."
		# terminating action
		# curr_state shape > (32, 32, 3)
		qval = np.zeros(self.number_of_actions)
		
		if (len(self.states) > 2)and ((self.curr_step > self.max_steps) or self.curr_iou >= 0.8 or self.curr_iou <= 0.05):
			action = self.number_of_actions
			qval[action -1] = 1.

		elif random.random() < self.epsilon:
			action = np.random.randint(1, self.number_of_actions+1)
			qval[action -1] = 1
		
		else:	
			_hist = torch.autograd.Variable(torch.from_numpy(self.history_vec.reshape(1, self.history_of_actions*self.number_of_actions)).cuda())
			_state = torch.autograd.Variable(self.curr_state.cuda())
			qval = self.cnet(_state, _hist).detach().cpu().numpy()
			action = (np.argmax(qval))+1
		
		print ("Mask size: ", self.size_mask, "; Current IOU: ", self.curr_iou,  "; Done: ", self.done, "; Action: ", action, "; Reward: ", self.reward, "; Offset: ", self.offset, "; Exploration fraction: ", self.epsilon)

		self.update_history_vec(qval)
		return action

	def visualization(self, path, save=False, display= False):
		"saves an image for visualization also does contour based segmentation"
		# TODO:
		if len(self.states) == 0:
			return
		else:
			for i in range(len(self.states)):
				plt.subplot(3, len(self.states), i+1)
				plt.imshow(np.array(self.states[i].numpy()[0,0,:,:]), cmap = 'gray')
				plt.title('a:' + str(self.actions[i]) + ' r:' + str(self.rewards[i]))
				plt.xlabel('iou: {:.2f}'.format(self.ious[i]))
				plt.gca().axes.get_xaxis().set_ticks([])
				plt.gca().axes.get_yaxis().set_ticks([])
				plt.subplot(3, len(self.states), len(self.states) + i+ 1)
				plt.imshow(self.gt_mask, cmap='gray')
				plt.gca().axes.get_xaxis().set_ticks([])
				plt.gca().axes.get_yaxis().set_ticks([])
				plt.subplot(3, len(self.states), 2*len(self.states) + i+ 1)
				plt.imshow(self.region_masks[i], cmap='gray')
				plt.gca().axes.get_xaxis().set_ticks([])
				plt.gca().axes.get_yaxis().set_ticks([])
			plt.savefig(path)
		pass

	def plot_cum_reward(self, save = False):
		save = False
		pass

	def generate_data(self, state, action, hist_vec, reward, next_state, done, batch_size = 32):
		# print (state.shape, state[0].shape)
		
		return_state = torch.zeros(self.memory_capacity, 3, 224, 224).cuda()
		return_y  = torch.zeros(self.memory_capacity, self.number_of_actions).cuda()
		return_hist_vec  = torch.zeros(self.memory_capacity, self.history_of_actions*self.number_of_actions).cuda()
		# print (hist_vec.shape, next_state.shape)
		for i in range(int(self.memory_capacity/ batch_size)):
			state_ = torch.autograd.Variable(torch.from_numpy(state[i*batch_size:(i+1)*batch_size,:,:,:,:].reshape(batch_size, 3, 224, 224)).cuda())
			hist_vec_=torch.autograd.Variable(torch.from_numpy(hist_vec[i*batch_size:(i+1)*batch_size,:,:].reshape(batch_size,self.history_of_actions*self.number_of_actions)).cuda())
			new_state_ = torch.autograd.Variable(torch.from_numpy(next_state[i*batch_size:(i+1)*batch_size,:,:,:,:].reshape(batch_size, 3, 224, 224)).cuda())

			old_qval = self.cnet(state_, hist_vec_).detach().cpu().numpy()
			new_qval = self.cnet(new_state_, hist_vec_).detach().cpu().numpy()

			max_qval = np.max(new_qval, 1)
			y = np.zeros([len(state_), self.number_of_actions])
			y = old_qval

			reward_ = reward[i*batch_size:(i+1)*batch_size]
			action_ = action[i*batch_size:(i+1)*batch_size]
			update = reward_

			update[action_ != self.number_of_actions] = reward_[action_ != self.number_of_actions] + self.gamma * max_qval[action_ != self.number_of_actions]
			y[:, action_-1] = update #target output

			y = torch.from_numpy(y).cuda()

			return_state[i*batch_size:(i+1)*batch_size,:,:,:] = state_
			return_hist_vec[i*batch_size:(i+1)*batch_size,:] = hist_vec_
			return_y[i*batch_size:(i+1)*batch_size,:] = y
		return return_state, return_hist_vec, return_y

	def fit(self, batch_size=32):
		"Weight update for combined network.."
		#-------------------- SETTINGS: LOSS
		loss = torch.nn.MSELoss()
		optimizer = optim.Adam (self.cnet.parameters(),  lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
		scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'max')

		step = 0
		prev_loss = float('inf')
		state, action, hist_vec, reward, next_state, done = self.exp_memory.sample(self.memory_capacity)
		X_state_train, X_hist_train, y_train = self.generate_data(state, action, hist_vec, reward, next_state, done)

		for i in tqdm(range(int(self.memory_capacity/ batch_size))):
			if step % 150 == 149:
				loss_val = self.batch_valid(X_state_train[i*batch_size:(i+1)*batch_size, :, :, :], 
							X_hist_train[i*batch_size:(i+1)*batch_size, :], 
							y_train[i*batch_size:(i+1)*batch_size, :], 
							loss)
				if loss_val < prev_loss:
					prev_loss = loss_val
					model_name = '../models/cum_reward = ' + str(loss_val)+ '.pth.tar'
					print ('Model saved -------- loss: {}'.format(loss_val))
					torch.save(self.cnet, model_name)
			else:
				self.batch_train(X_state_train[i*batch_size:(i+1)*batch_size, :, :, :], 
							X_hist_train[i*batch_size:(i+1)*batch_size, :], 
							y_train[i*batch_size:(i+1)*batch_size, :], 
							loss, 
							optimizer)
			step += 1
		pass

	def batch_train(self, X_state_train, X_hist_train, y_train, loss, optimizer):

		varOutput = self.cnet(X_state_train, X_hist_train)
		# print varInput.size(), varOutput.size(), target.size()
		# varOutput = torch.FloatTensor([0])
		# lossfn = loss(weights = weights)

		lossvalue = loss(varOutput, y_train)
		l2_reg = None
		for W in self.cnet.parameters():
		    if l2_reg is None:
		        l2_reg = W.norm(2)
		    else:
		        l2_reg = l2_reg + W.norm(2)

		lossvalue = lossvalue + l2_reg * 1e-3
		optimizer.zero_grad()
		lossvalue.backward()
		optimizer.step()
		pass

	def batch_valid(self, X_state_train, X_hist_train, y_train, loss):
		varOutput = self.cnet(X_state_train, X_hist_train)
		losstensor = loss(varOutput, y_train)
		lossVal = losstensor.data[0]
		return lossVal

	def update_history_vec(self, qval):
		"updates history vector"
		self.history_vec = self.history_vec[:-1]
		self.history_vec = np.insert(self.history_vec, 0, qval, 0)
		pass
