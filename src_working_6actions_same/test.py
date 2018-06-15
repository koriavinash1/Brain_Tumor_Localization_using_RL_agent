import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from skimage.transform import resize
import torchvision.transforms as transforms

from torchvision.models.densenet import model_urls
from pytorch_net_working import replayBuffer, Agent
import matplotlib
# matplotlib.use('Qt4Agg')
# matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

normalize = transforms.Normalize([0.511742964836, 0.243537961753, 0.0797484182405], [0.223165616204, 0.118469339976, 0.0464971614141])
trans = transforms.ToTensor()


root_path = "../slices/train_set/lesion"
nepochs   = 100

def get_data(image_name, path = root_path):
    h5      = h5py.File(os.path.join(path, image_name), 'r')
    img     = np.zeros((224, 224, 3))	
    t = resize(h5['Sequence'][:], (224, 224), order = 1)
    img[:,:,0] = t[:,:,0]
    img[:,:,1] = t[:,:,0]
    img[:,:,2] = t[:,:,0]

    img     = np.uint8(img) # 1 for bilinear interpolation
    mask_gt = resize(h5['label'][:], (224, 224), order = 0)# 0 for nearest neighbour interpolation 
    # print (np.unique(h5['label'][:]))
    mask_gt[mask_gt > 0] = 1
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    return img, mask_gt

ids = next(os.walk(root_path))[2]
np.random.shuffle(ids)

env = Agent()
for ee in tqdm(range(nepochs)):
	np.random.shuffle(ids)
	# ids = ids[:5000]
	step = 0
	for id_ in tqdm(ids):
		img, gt = get_data(id_)
		state = img
		done = False
		env.reset()
		print ("========================================")
		if env.exp_memory.__len__() < env.memory_capacity:
			while not done:
				curr_state, action, history_vector, reward, state, done = env.step(state, gt)
				# print (done)
				env.update_replay_buffer([curr_state, action, history_vector, reward, state, done])
			if (ee > -1) and (step % 10 == 0): 
				path = path = '../logs/epoch_'+str(ee)
				if not os.path.exists(path): os.mkdir(path)
				env.visualization(save = True, display = False, path = os.path.join(path, id_.split('.')[0]+'.jpg'))
		else:
			env.fit()
			env.buffer_reset()

		step +=1


	if env.epsilon > 0.2:
		env.epsilon -= 0.1
