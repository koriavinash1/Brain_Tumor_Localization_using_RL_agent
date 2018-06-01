import cv2, numpy as np
import time
import math as mth
from PIL import Image, ImageDraw, ImageFont
import scipy.io
import random
import argparse
import keras

from features import get_image_descriptor_for_image, obtain_compiled_vgg_16, vgg_16, \
    get_conv_image_descriptor_for_image, calculate_all_initial_feature_maps

from tqdm import tqdm
from image_helper import *
from metrics import *
from visualization import *
from reinforcement import *


# python image_zooms_training.py -n 0

# Read number of epoch to be trained, to make checkpointing
parser = argparse.ArgumentParser(description='Epoch:')
parser.add_argument("-n", metavar='N', type=int, default=0)
args = parser.parse_args()
epochs_id = int(args.n)


if __name__ == "__main__":

    ######## PATHS definition ########

    # path of PASCAL VOC 2012 or other database to use for training
    path_brats = "../slices"
    # path of where to store the models
    path_model = "../models_image_zooms"
    # path of where to store visualizations of search sequences
    path_testing_folder = '../testing_visualizations'

    ######## PARAMETERS ########

    # Class category of PASCAL that the RL agent will be searching
    class_object = 1
    # Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
    scale_subregion = float(3)/4
    scale_mask = float(1)/(scale_subregion*4)

    # 1 if you want to obtain visualizations of the search for objects
    bool_draw = 0
    # How many steps can run the agent until finding one object
    number_of_steps = 5

    epochs = 50
    gamma = 0.90
    epsilon = 1 # for exploration policy
    batch_size = 100

    h = np.zeros([1])
    # Each replay memory (one for each possible category) has a capacity of 100 experiences
    buffer_experience_replay = 1000
    # Init replay memories
    replay = [[] for i in range(20)]
    reward = 0


    featureModel = convNet()
    # If you want to train it from first epoch, first option is selected. Otherwise,
    # when making checkpointing, weights of last stored weights are loaded for a particular class object

    if epochs_id == 0:
        Qnet_model = get_q_network_for_lesion("0", class_object)
    else:
        Qnet_model = get_q_network_for_lesion(path_model, class_object)

    ######## LOAD IMAGE NAMES ########
    image_names = np.array([load_file_names('trainval', path_brats)])

    ######## LOAD IMAGES ########
    # images = get_all_images(image_names, path_brats)


    for i in tqdm(range(epochs_id, epochs_id + epochs)):
        for j in range(np.size(image_names)):
            masked = 0
            not_finished = 1

            image_name = image_names[j]
            image      = get_image(image_name, path_brats)
            gt_masks, annotation = get_bb_of_get(image_name, path_brats) # needs changes for multiclass problems
            array_classes_gt_objects = get_ids_objects_from_annotation(annotation)

            region_mask       = np.ones([image.shape[0], image.shape[1]])
            shape_gt_masks    = np.shape(gt_masks)
            available_object = 0

            if available_object:
                gt_mask = gt_masks[:, :, class_object]
                step = 0
                new_iou = 0

                # this matrix stores the IoU of each object of the ground-truth, just in case
                # the agent changes of observed object
                last_matrix = np.zeros([np.size(array_classes_gt_objects)])
                region_image = image


                size_mask = (image.shape[0], image.shape[1])
                original_shape = size_mask
                old_region_mask = region_mask
                region_mask = np.ones([image.shape[0], image.shape[1]])

                # If the ground truth object is already masked by other already found masks, do not
                # use it for training
                if masked == 1:
                    iou = overlap = calculate_overlapping(old_region_mask, gt_mask)
                    if overlap > 0.60:
                        available_object = 0

                # init of the history vector that indicates past actions (6 actions * 4 steps in the memory)
                # TODO: global action-history setting.....
                history_vector = np.zeros([24])

                # computation of the initial state
                state = get_state(region_image, history_vector, featureModel)

                # status indicates whether the agent is still alive and has not triggered the terminal action
                status = 1
                action = 0
                reward = 0

                if step > number_of_steps and bool_draw:
                    # Init visualization
                    background = Image.new('RGBA', (600, 270), (255, 255, 255, 255))
                    draw = ImageDraw.Draw(background)
                    background = draw_sequences(i, step, action, draw, region_image, background,
                                                path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                                                bool_draw)
                    step += 1

                while (status == 1) & (step < number_of_steps) & not_finished:
                    category = 0
                    qval = Qnet_model.predict(state.T, batch_size=1)
                    if bool_draw:
                        background = Image.new('RGBA', (600, 270), (255, 255, 255, 255))
                        draw = ImageDraw.Draw(background)
                        background = draw_sequences(i, step, action, draw, region_image, background,
                                                path_testing_folder, iou, reward, gt_mask, region_mask, image_name,
                                                bool_draw)
                    step += 1



                    # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
                    if (i < 100) & (new_iou > 0.6):
                        action = 6

                    # epsilon-greedy policy
                    elif random.random() < epsilon:
                        action = np.random.randint(1, 7)
                    else:
                        action = (np.argmax(qval))+1


                    # terminal action
                    if action == 6:
                        new_iou = calculate_overlapping(region_mask, gt_mask)
                        reward = get_reward_trigger(new_iou)

                        if bool_draw:
                            background = Image.new('RGBA', (600, 270), (255, 255, 255, 255))
                            draw = ImageDraw.Draw(background)
                            background = draw_sequences(i, step, action, draw, region_image, background,
                                                        path_testing_folder, iou, reward, gt_mask, region_mask,
                                                        image_name, bool_draw)
                        step += 1

                    # movement action, we perform the crop of the corresponding subregion
                    else:
                        region_mask = np.zeros(original_shape)
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
                            offset_aux = (size_mask[0] * scale_mask,
                                          size_mask[1] * scale_mask)
                            offset = (offset[0] + size_mask[0] * scale_mask,
                                      offset[1] + size_mask[1] * scale_mask)

                        elif action == 5:
                            offset_aux = (size_mask[0] * scale_mask / 2,
                                          size_mask[0] * scale_mask / 2)
                            offset = (offset[0] + size_mask[0] * scale_mask / 2,
                                      offset[1] + size_mask[0] * scale_mask / 2)



                        region_image = region_image[int(offset_aux[0]):int(offset_aux[0] + size_mask[0]),
                                       int(offset_aux[1]):int(offset_aux[1] + size_mask[1])]

                        region_mask[int(offset[0]):int(offset[0] + size_mask[0]), int(offset[1]):int(offset[1] + size_mask[1])] = 1

                        new_iou = calculate_overlapping(region_mask, gt_mask)
                        reward  = get_reward_movement(iou, new_iou)
                        iou     = new_iou



                    history_vector = update_history_vector(history_vector, action)
                    new_state = get_state(region_image, history_vector, featureModel)

                    # Experience replay storage
                    # replay is memory storage which serves for training.....
                    if len(replay[category]) < buffer_experience_replay:
                        replay[category].append((state, action, reward, new_state))

                    else:
                        if h[category] < (buffer_experience_replay-1):
                            h[category] += 1
                        else:
                            h[category] = 0

                        h_aux = h[category]
                        h_aux = int(h_aux)
                        replay[category][h_aux] = (state, action, reward, new_state)
                        minibatch = random.sample(replay[category], batch_size)
                        X_train   = []
                        y_train   = []

                        # we pick from the replay memory a sampled minibatch and generate the training samples
                        for memory in minibatch:
                            old_state, action, reward, new_state = memory
                            old_qval = Qnet_model.predict(old_state.T, batch_size=1)
                            newQ = Qnet_model.predict(new_state.T, batch_size=1)
                            maxQ = np.max(newQ)
                            y = np.zeros([1, 6])
                            y = old_qval
                            y = y.T
                            if action != 6: #non-terminal state
                                update = (reward + (gamma * maxQ))
                            else: #terminal state
                                update = reward
                            y[action-1] = update #target output
                            X_train.append(old_state)
                            y_train.append(y)

                        X_train = np.array(X_train, dtype = 'float32')
                        y_train = np.array(y_train, dtype = 'float32')
                        X_train = X_train[:, :, 0]
                        y_train = y_train[:, :, 0]

                        hist = Qnet_model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, verbose=0)
                        state = new_state

                    if action == 6:
                        status = 0
                        masked = 1
                        # we mask object found with ground-truth so that agent learns faster
                        image = mask_image_with_mean_background(gt_mask, image)
                    else:
                        masked = 0

                available_object = 0

        # dynamic sdjustment of epsilon for exploration control.....
        if epsilon > 0.1:
            epsilon -= 0.1


        string = path_model + '/Qnet_model1_epoch_' + str(i) + '.h5'
        string2 = path_model + '/Qnet_model1.h5'
        Qnet_model.save_weights(string, overwrite=True)
        Qnet_model.save_weights(string2, overwrite=True)
