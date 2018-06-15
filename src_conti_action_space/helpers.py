import os
import random
import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.models.resnet import model_urls as resnetmodel_urls
from tqdm import tqdm


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

