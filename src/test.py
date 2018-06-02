import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import h5py

import torch
import torch.nn as nn
import torch.Functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torchvision.models.densenet import model_urls
from pytorch_net import replayBuffer, Agent

env = Agent()

root_path = "../slices/train_set/lesion"
nepochs   = 50

def get_data(image_name, path = root_path):
    h5      = h5py.File(os.path.join(path, image_name), 'r')
    img     = h5['Sequence'][:]
    mask_gt = h5['Label'][:]
    return img, mask_gt

ids = next(os.walk(root_path))[2]


for ee in range(nepochs):
	np.random.shuffle(paths)

	for id_ in ids:
		img, gt = get_data(id_)
		env.reset()
		done = False
		if env.exp_memory.__len__() < env.memory_capacity:
			while not done:
				state = img
				curr_state, action, history_vector, reward, state, done = env.step(state, gt)
				env.update_replay_buffer([curr_state, action, history_vector, reward, state, done])
		else:
			env.fit(batch_size = 32)
			env.buffer_reset()


	if env.epsilon > 0.1:
		env.epsilon -= 0.1
