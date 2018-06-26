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
from skimage.transform import resize


from ReplayBuffer import *
from Networks import *
from helpers import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class Agent(object):
    def __init__(self, max_steps=5, max_frames=5000, epsilon=1, verbose= True):
        "init actions"
        super(Agent, self).__init__()
        self.max_steps   = max_steps
        self.memory_capacity = max_frames
        self.epsilon     = epsilon
        self.gamma       = 0.1 # discount factor
        self.iou_thresh  = 0.5
        self.reshapesize = 64
        self.terminal_reward    = 3
        self.momentum_reward    = 1
        self.number_of_actions  = 2 # 1 coordinates to define bounding box
        self.exp_memory  = replayBuffer(self.memory_capacity)
        self.state_shape = (128, 128)
        # exp withour history information 
        # self.history_vec = np.zeros((self.history_of_actions, self.number_of_actions))

        self.done = False
        self.verbose = verbose 

        self.vnet  = ValueNetwork().to(device)
        self.DPG   = PolicyNetowrk().to(device)

        self.tvnet = ValueNetwork().to(device)
        self.tDPG  = PolicyNetowrk().to(device)

        for target_param, param in zip(self.tvnet.parameters(), self.vnet.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.tDPG.parameters(), self.DPG.parameters()):
            target_param.data.copy_(param.data)

        # self.cnet  = combinedNetwork().to(device)
        # self.tcnet = combinedNetwork().to(device)

        # for target_param, param in zip(self.tcnet.parameters(), self.cnet.parameters()):
        #     target_param.data.copy_(param.data)


    def reset(self):
        "resets action and steps..."
        self.state       = None
        self.states      = []
        self.region_masks= []
        self.actions     = []
        self.rewards     = []
        self.ious        = []
        self.gts         = []
        self.reward      = 0
        self.cum_reward  = 0
        self.curr_step   = 0
        self.curr_frame  = 0
        self.prev_iou    = 0
        self.curr_iou    = 0
        self.locations   = np.array([0.,0.])
        self.size_mask   = self.state_shape
        # self.exp_memory  = replayBuffer(self.memory_capacity)
        # self.history_vec = np.zeros((self.history_of_actions, self.number_of_actions))
        self.done        = False

    def buffer_reset(self):
        """

        """
        self.exp_memory  = replayBuffer(self.memory_capacity)

    def step(self, state, gt_mask):
        """

        """
        self.region_mask = np.ones(self.size_mask)
        self.curr_state = state
        self.gt_mask = gt_mask
        self.curr_step += 1
        
        # for visualization....

        self.curr_iou  = calculate_iou(self.region_mask, self.gt_mask)
        self.states.append(self.curr_state)
        self.region_masks.append(self.region_mask)
        self.gts.append(self.gt_mask)
        self.actions.append(self.locations)
        self.rewards.append(self.reward)
        self.ious.append(self.curr_iou)

        self.locations = self.get_action()
        locations = np.zeros((2, 2), dtype='int32')
        shape = gt_mask.shape

        # denormalize location values
        temp = np.array([max(0, 0.5*(c+1)*shape[0] - 32) for c in self.locations[0]], dtype='int32')

        locations[0, 0] = temp[0]
        locations[0, 1] = temp[1]
        locations[1, 0] = locations[0, 0] + self.reshapesize
        locations[1, 1] = locations[0, 1] + self.reshapesize

        # if locations[1, 0] == locations[0, 0]: locations[1, 0] = locations[0, 0] + 32
        # if locations[1, 1] == locations[0, 1]: locations[1, 1] = locations[0, 1] + 32
        
        self.state = self.curr_state[:,:, locations[0, 0]:locations[1, 0], locations[0, 1]:locations[1, 1]]
        self.state = F.adaptive_avg_pool2d(self.state, self.size_mask)

        self.gt_mask = self.gt_mask[locations[0, 0]:locations[1, 0], locations[0, 1]:locations[1, 1]]
        self.gt_mask = resize(self.gt_mask, self.size_mask, order = 0)
        self.curr_iou  = calculate_iou(self.region_mask, self.gt_mask)

        if (self.curr_step > self.max_steps) or (self.curr_iou > 0.9):
            self.reward = self.get_reward(terminal=True)
            self.done = True

            self.cum_rewards.append(np.sum(self.rewards))
            self.states.append(self.state)
            self.region_masks.append(self.region_mask)
            self.gts.append(self.gt_mask)
            self.actions.append(self.locations)
            self.rewards.append(self.reward)
            self.ious.append(self.curr_iou)
            
        else:
            self.reward = self.get_reward()
            
        self.prev_iou = self.curr_iou
        return self.curr_state, self.locations, self.reward, self.state, self.done, self.gt_mask

    def get_reward(self, terminal=False):

        if not terminal:
            if self.curr_iou <= self.prev_iou:
                reward = -1*self.momentum_reward
            else:
                reward = 1*self.momentum_reward
        else:
            if self.curr_iou > self.iou_thresh:
                reward = 1*self.terminal_reward
            else:
                reward = -1*self.terminal_reward

        return reward

    def update_replay_buffer(self, _vec):
        "push vec.  [old_state, action, reward, new_state, done]"
        self.exp_memory.push(_vec)
        pass

    def get_action(self):
        "returns action for a given state..."
        # terminating action
        # curr_state shape > (32, 32, 3)

        if random.random() < self.epsilon:
            locations = np.random.rand(1, 2)
        
        else:   
            _state = torch.FloatTensor(self.curr_state).to(device)
            locations, _ = self.DPG(_state)
            locations = locations.detach().cpu().numpy()
        
        if self.verbose:
            print ("Mask size: ", self.size_mask, "; Current IOU: ", self.curr_iou,  "; Done: ", self.done, "; Locations: ", locations, "; Reward: ", self.reward,  "; Exploration fraction: ", self.epsilon)

        # self.update_history_vec(locations)
        return locations

    def visualization(self, path, save=False, display= False):
        "saves an image for visualization also does contour based segmentation"
        
        if len(self.states) == 0:
            return
        else:
            for i in range(len(self.states)):
                plt.subplot(2, len(self.states), i+1)
                plt.imshow(np.array(self.states[i].numpy()[0,0,:,:]), cmap = 'gray')
                plt.title(' r:' + str(self.rewards[i]))
                plt.xlabel('iou: {:.2f}'.format(self.ious[i]))
                plt.gca().axes.get_xaxis().set_ticks([])
                plt.gca().axes.get_yaxis().set_ticks([])
                plt.subplot(2, len(self.states), len(self.states) + i+ 1)
                plt.imshow(self.gts[i], cmap='gray')
                plt.gca().axes.get_xaxis().set_ticks([])
                plt.gca().axes.get_yaxis().set_ticks([])
                #plt.subplot(3, len(self.states), 2*len(self.states) + i+ 1)
                #plt.imshow(self.region_masks[i], cmap='gray')
                #plt.gca().axes.get_xaxis().set_ticks([])
                #plt.gca().axes.get_yaxis().set_ticks([])
            plt.savefig(path)
        pass

    def plot_cum_reward(self, save = False):
        save = False
        pass


    def fit(self, batch_size=16, soft_tau=1e-2):
        "Weight update for combined network.."
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.MSELoss()
        policy_optimizer = optim.Adam (self.cnet.parameters(),  lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        value_optimizer = optim.Adam (self.cnet.parameters(),  lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'max')

        step = 0
        prev_loss = float('inf')

        for i in tqdm(range(int(self.memory_capacity/ batch_size))):
            state, action, reward, next_state, done = self.exp_memory.sample(batch_size)

            state      = torch.FloatTensor(state).squeeze(1).to(device)
            next_state = torch.FloatTensor(next_state).squeeze(1).to(device)
            action     = torch.FloatTensor(action).squeeze(1).to(device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

            if step % 150 == 149:
                value_loss, policy_loss = self.batch_valid(state, next_state, action,reward. done, loss)
                loss_val = value_loss
                if loss_val < prev_loss:
                    prev_loss = loss_val
                    if os.path.exists('../exp5models'):
                        os.mkdir('../exp5models')

                    model_name = '../exp5models/best_model.pth.tar'
                    print ('Model saved -------- loss: {}'.format(loss_val))
                    dict_ = {'value_net': self.vnet, 'policy_net': self.DPG}
                    torch.save(dict_, model_name)
                print ('----------- loss: {}'.format(loss_val))
            else:
                self.batch_train(state, next_state, action,reward, hist_vec, done, loss, policy_optimizer, value_optimizer)
            step += 1

        for target_param, param in zip(self.tvnet.parameters(), self.vnet.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.tDPG.parameters(), self.DPG.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        # for target_param, param in zip(self.tcnet.parameters(), self.cnet.parameters()):
        #     target_param.data.copy_(
        #         target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        #     )

        pass

    def batch_train(self, state, next_state, action,reward, hist_vec, done, loss, policy_optimizer, value_optimizer):

        policy_loss = self.vnet(state, self.DPG(state))
        policy_loss = -policy_loss.mean()

        next_action    = self.tDPG(next_state)
        target_value   = self.tDPG(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.vnet(state, action)
        value_loss = loss(value, expected_value.detach())
        
        self.tvnet  = self.vnet.to(device)
        self..tDPG  = self.DPG.to(device)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        pass

    def batch_valid(self, state, next_state, action,reward, hist_vec, done, loss):
        policy_loss = self.vnet(state, self.DPG(state))
        policy_loss = -policy_loss.mean()

        next_action    = self.tDPG(next_state)
        target_value   = self.tDPG(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.vnet(state, action)
        value_loss = loss(value, expected_value.detach())
        return value_loss, policy_loss

    def update_history_vec(self, location):
        "updates history vector"
        self.history_vec = self.history_vec[:-1]
        self.history_vec = np.insert(self.history_vec, 0, location.flatten(), 0)
        pass
