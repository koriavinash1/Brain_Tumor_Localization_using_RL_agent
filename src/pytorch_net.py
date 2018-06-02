import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.Functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torchvision.models.densenet import model_urls
from torchvision.models.resnet import model_urls as resnetmodel_urls

class replayBuffer(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer   = []
		self.position = 0

	def sampler(self, batch_size):
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

	def __len__(self)
		return len(self.buffer)


class QNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions):
		# num_inputs = number of feature outputs + 4 past actions 24
		self.features     = nn.Sequential(OrderedDict([]))
		self.features.add_module('layer1', F.relu(nn.Linear(num_inputs, 1024)))
		self.featuers.add_module('layer2', F.relu(nn.Linear(1024, 512)))
		self.features.add_module('layer3', F.relu(nn.Linear(512, num_actions)))

	def forward(self, state, past_actions):
		x = torch.cat([state, past_actions], 1)
		actions = self.features(x)
		return actions


class featureExtractor(nn.Module):
	def __init__(self, classCount=1024, isTrained=True):
		super(featureExtractor, self).__init__()
		model_urls['densenet121'] = model_urls['densenet121'].replace('https://', 'http://')
		self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
		kernelCount = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.densenet121(x)
		return x


class combinedNetwork(nn.Module):
	def __init__(self, ninputs, nactions):
		super(combinedNetwork, self).__init__()
		self.features = nn.Sequential(OrderedDict([]))
		self.features.add_module('featureExtractor', featureExtractor())
		self.Qnet     = QNetwork(num_inputs = ninputs, num_actions = nactions)

	def forward(self, x, history_vec):
		x = self.features(x)
		x = self.Qnet(x, history_vec)
		return x

def calculate_iou(img_mask, gt_mask):
	gt_mask *= 1.0
	img_and = cv2.bitwise_and(img_mask, gt_mask)
	img_or = cv2.bitwise_or(img_mask, gt_mask)
	j = np.count_nonzero(img_and)
	i = np.count_nonzero(img_or)
	iou = float(float(j)/float(i))
	return iou


def calculate_overlapping(img_mask, gt_mask):
	gt_mask *= 1.0
	img_and = cv2.bitwise_and(img_mask, gt_mask)
	j = np.count_nonzero(img_and)
	i = np.count_nonzero(gt_mask)
	overlap = float(float(j)/float(i))
	return overlap


class Agent(object):
	def __init__(self, max_steps=6, max_frames=5000, epsilon=1):
		"init actions"
		super(Agent, self).__init__()
		self.state       = None
		self.action      = 0
		self.reward      = -1
		self.max_steps   = max_steps
		self.max_frames  = max_frames
		self.epsilon     = epsilon
		self.curr_step   = 0
		self.curr_frame  = 0
		self.gamma       = 0.1 # discount factor
		self.iou_thresh  = 0.6
		self.prev_iou    = 0
		self.memory_capacity    = 5000
		self.terminal_reward    = 3
		self.momentum_reward    = 1
		self.number_of_actions  = 6
		self.history_of_actions = 4 # number of actions to be used in QNetwork
		self.history_vec = np.zeros(self.history_of_actions, self.number_of_actions)
		self.exp_memory  = replayBuffer(self.memory_capacity)
		self.scale_subregion = float(3)/4
		self.scale_mask = float(1)/(scale_subregion*4)

		# define deep networks
		# use qnet for filling experiance replay memory
		# fnet for feature extractor
		# cnet for network weights updates...

		self.qnet       = QNetwork(1048, self.number_of_actions)#
		self.fnet       = featureExtractor()
		self.cnet       = combinedNetwork(1048, self.number_of_actions) # combined network fro training


	def reset(self):
		"resets action and steps..."
		self.state       = None
		self.action      = 0
		self.reward      = -1
		self.curr_step   = 0
		self.curr_frame  = 0
		self.prev_iou    = 0
		# self.exp_memory  = replayBuffer(self.memory_capacity)
		self.history_vec = np.zeros(self.history_of_actions, self.number_of_actions)
		self.done        = False

	def buffer_reset(self):
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
		size_mask = original_shape = gt_mask.shape
		self.curr_state = state
		self.gt_mask = gt_mask
		self.curr_step += 1
		self.action = self.get_action()

		if self.action == 6:
			self.reward = get_reward(terminal=True)
			self.done = True
		else:
			self.region_mask = np.zeros(original_shape)
			size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
			offset = (0, 0)

			if action == 1:
				offset_aux = (0, 0)

			elif action == 2:
				offset_aux = (0, size_mask[1] * scale_mask)
				offset = (offset[0], offset[1] + size_mask[1] * scale_mask)

			elif action == 3:
				offset_aux = (size_mask[0] * scale_mask, 0)
				offset = (offset[0] + size_mask[0] * scale_mask, offset[1])

			elif action == 4:
				offset_aux = (size_mask[0] * scale_mask, size_mask[1] * scale_mask)
				offset = (offset[0] + size_mask[0] * scale_mask, offset[1] + size_mask[1] * scale_mask)

			elif action == 5:
				offset_aux = (size_mask[0] * scale_mask / 2, size_mask[0] * scale_mask / 2)
				offset = (offset[0] + size_mask[0] * scale_mask / 2, offset[1] + size_mask[0] * scale_mask / 2)

			self.state = self.curr_state[int(offset_aux[0]):int(offset_aux[0] + size_mask[0]),
											int(offset_aux[1]):int(offset_aux[1] + size_mask[1])]

			self.region_mask[int(offset[0]):int(offset[0] + size_mask[0]), int(offset[1]):int(offset[1] + size_mask[1])] = 1
			self.reward  = get_reward(iou, new_iou)
			self.prev_iou = self.curr_iou

		self.cum_reward += self.reward

		return self.curr_state, self.action, self.history_vector, self.reward, self.state, self.done

	def get_reward(self, terminal=True):
		self.curr_iou = calculate_iou(self.region_mask, self.gt_mask)
		if not terminal:
			if self.curr_iou > self.prev_iou:
				reward = -1*self.momentum_reward
			else:
				reward = 1*self.momentum_reward
		else
			if self.curr_iou > self.iou_thresh:
				reward = 1*self.terminal_reward
			else:
				reward = -3*self.terminal_reward
		return reward

	def update_replay_buffer(self, _vec)
		"push vec.  [old_state, action, reward, new_state, done]"
		self.exp_memory.push(_vec)
		pass

	def get_action(self):
		"returns action for a given state..."
		# terminating action
		# curr_state shape > (32, 32, 3)
		qval = np.zeros(6)
		if (self.curr_step > self.max_steps) and (self.curr_state.shape(0) <= 32):
			action = 6
			qval[action -1] = 1.
		# epsilon-greedy policy
		elif random.random() < self.epsilon:
			action = np.random.randint(1, 7)
			qval[action - 1] = 1.
		else:
			qval = self.cnet(self.curr_state, self.history_vec.flatten(order='F'))
			action = (np.argmax(qval))+1

		self.update_history_vec(qval)
		return action

	def visualization(self,):
		"saves an image for visualization also does contour based segmentation"
		# TODO:
		pass

	def generate_data(self, state, action, hist_vec, reward, next_state, done):
		state = torch.autograd.Variable(state.cuda())
		hist_vec = torch.autograd.Variable(hist_vec.cuda())
		new_state = torch.autograd.Variable(new_state.cuda())

		old_qval = self.cnet(state, hist_vec)
		new_qval = self.cnet(new_state, hist_vec)
		max_qval = np.max(new_qval, 1)

		y = np.zeros([len(state), 6])
		update = reward
		update[action != 6] = reward[action != 6] + self.gamma * max_qval[action != 6]
		y[:, action-1] = update #target output
		return state, hist_vec, y

	def fit(self, batch_size):
		"Weight update for combined network.."
		#-------------------- SETTINGS: LOSS
		loss = torch.nn.MSELoss()
		step = 0
		prev_loss = float('inf')
		for i in range(int(self.memory_capacity/ batch_size)):
			state, action, hist_vec, reward, next_state, done = self.exp_memory.sample(batch_size)
			X_state_train, X_hist_train, y_train = self.generate_data(state, action, hist_vec, reward, next_state, done)
			if step % 10 == 9:
				loss_val = self.batch_valid(X_state_train, X_hist_train, y_train, loss)
				if loss_val < prev_loss:
					prev_loss = loss_val
					model_name = '../models/cum_reward = ' + str(loss_val)+ '.pth.tar'
					torch.save(model, model_name)
			else:
				self.batch_train(X_state_train, X_hist_train, y_train, loss)
			step += 1

		pass

	def batch_train(self, X_state_train, X_hist_train, y_train, loss):

		varOutput = self.cnet(X_state_train, X_hist_train)
		# print varInput.size(), varOutput.size(), target.size()
		# varOutput = torch.FloatTensor([0])

		# lossfn = loss(weights = weights)
		lossvalue = loss(varOutput, varTarget)
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
		losstensor = loss(varOutput, varTarget)
		lossVal = losstensor.data[0]
		return lossVal

	def update_history_vec(self, qval):
		"updates history vector"
		self.history_vector = self.history_vector[:-1]
		self.history_vector = np.insert(self.history_vector, 0, qval)
		pass
