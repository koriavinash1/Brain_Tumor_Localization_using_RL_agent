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
from torch.utils.data import DataLoader
from torchvision.models.densenet import model_urls
from pytorch_net import replayBuffer, Agent
import matplotlib
# matplotlib.use('Qt4Agg')
# matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from DataGenerator import DatasetGenerator 


normalize = transforms.Normalize([0.5117, 0.2435, 0.0797], [0.2231, 0.1184, 0.0464])
transformList = []
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence=transforms.Compose(transformList)


root_path = "../Refuge/training/"
nepochs   = 100

DG = DatasetGenerator(root = root_path, transform = transformSequence)
dl = DataLoader(dataset=DG, batch_size=1, shuffle=True,  num_workers=8, pin_memory=False)

def get_data(image_name, path = root_path):
    h5      = h5py.File(os.path.join(path, image_name), 'r')
    img     = np.zeros((128, 128, 3))	
    img     = resize(h5['Sequence'][:], (128, 128), order = 1)
    # img[:,:,0] = t[:,:,0]
    # img[:,:,1] = t[:,:,0]
    # img[:,:,2] = t[:,:,0]

    img     = img # 1 for bilinear interpolation
    mask_gt = resize(h5['label'][:], (128, 128), order = 0)# 0 for nearest neighbour interpolation 
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
	# ids = ids[:20]
	step = 0
	for img, gt, _ in dl:
		# img, gt = get_data(id_)
		# if np.sum(np.uint8(gt)) < 300. or np.std(gt) < 0.1:
		# 	print (np.max(gt), np.sum(gt), np.min(gt)) 
		# 	continue 
		state = img
		gt   = gt.data.numpy()
		done = False
		env.reset()
		print ("========================================")
		print (_)
		if env.exp_memory.__len__() < env.memory_capacity:
			while not done:
				curr_state, action, reward, state, done, gt = env.step(state.float(), gt)
				env.update_replay_buffer([curr_state, action, reward, state, done])
			if (ee > -1) and (step % 10 == 0): 
				path = path = '../exp4logs/epoch_'+str(ee)
				if not os.path.exists(path): os.mkdir(path)
				env.visualization(save = True, display = False, path = os.path.join(path, id_.split('.')[0]+'.jpg'))
		else:
			env.fit()
			env.buffer_reset()

		step +=1

	if env.epsilon > 0.15:
		env.epsilon -= 0.2
