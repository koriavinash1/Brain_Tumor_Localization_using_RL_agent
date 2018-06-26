import os
import random
import numpy as np
import cv2
from PIL import Image

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
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done

	def push(self, _vec):
		# _vec:[state, action, reward, next_state, done]
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = _vec
		self.position = (self.position + 1) % self.capacity
		pass

	def __len__(self):
		return len(self.buffer)

