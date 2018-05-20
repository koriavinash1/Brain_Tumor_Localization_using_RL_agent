from keras.preprocessing import image
import numpy as np
import h5py
import os


def get_all_ids(annotations):
    all_ids = []
    for i in range(len(annotations)):
        all_ids.append(get_ids_objects_from_annotation(annotations[i]))
    return all_ids

def get_image(image_name, path):
    h5  = h5py.load(os.path.join(path, image_name), 'r')
    img = h5['Sequence'][:]
    return img

def get_bb_of_gt(image_name, path, nclass = 2):
    h5  = h5py.load(os.path.join(path, image_name), 'r')
    mask = h5['Label'][:]
    mask_gt = np.zeros((mask.shape[0], mask.shape[1], nclass))
    for ii in nclass:
        mask_gt[ii][np.where(mask == ii)] = 1

    x, y = np.where(mask == 1)
    category_and_bb = np.zeros([1, 5])
    category_and_bb[0][0] = 1
    category_and_bb[0][1] = np.min(x)
    category_and_bb[0][2] = np.max(x)
    category_and_bb[0][3] = np.min(y)
    category_and_bb[0][4] = np.max(y)
    return mask_gt, category_and_bb

def get_ids_objects_from_annotation(annotation):
    return annotation[:, 0]

def get_all_images(image_names, path):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def get_all_images_pool(image_names, path):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[j]
        string = path + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def load_file_names(set_name, path):
    assert set_name not in ['train_set', 'valid_set']
    file_paths = next(os.walk(path + set_name))[2]
    return file_paths


def load_images_labels_in_data_set(data_set_name, path):
    file_path = path + set_name + '.txt'
    f = open(file_path)
    images_names = f.readlines()
    images_names = [x.split(None, 1)[1] for x in images_names]
    images_names = [x.strip('\n') for x in images_names]
    return images_names


def mask_image_with_mean_background(mask_object_found, image):
    new_image = image
    size_image = np.shape(mask_object_found)
    for j in range(size_image[0]):
        for i in range(size_image[1]):
            if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 103.939
                    new_image[j, i, 1] = 116.779
                    new_image[j, i, 2] = 123.68
    return new_image
