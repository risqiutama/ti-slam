from scipy import misc
import numpy as np
import cv2

def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    y = round(x) - .5
    return int(y) + (y > 0)

def get_positive_negative_samples(loop_path,  exp_folder):
    positive_file = loop_path + '/' + exp_folder + '/rgb_sampled/positive_loop.csv'
    negative_file = loop_path + '/' + exp_folder + '/rgb_sampled/negative_loop.csv'
    with open(positive_file, 'r') as pos_file:
        pos_data = [line[:-1] for line in pos_file]
    with open(negative_file, 'r') as neg_file:
        neg_data = [line[:-1] for line in neg_file]
    return pos_data, neg_data

def get_positive_negative_samples_subt(loop_path,  exp_folder):
    positive_file = loop_path + '/' + exp_folder + '/rgb_left_sampled/positive_loop.csv'
    negative_file = loop_path + '/' + exp_folder + '/rgb_left_sampled/negative_loop.csv'
    with open(positive_file, 'r') as pos_file:
        pos_data = [line[:-1] for line in pos_file]
    with open(negative_file, 'r') as neg_file:
        neg_data = [line[:-1] for line in neg_file]
    return pos_data, neg_data

def get_image(img_path):
    img = misc.imread(img_path)  # load raw radiometric data
    img = img.astype('float32')
    # normalize thermal value using min-max-mean from odometry
    img = (img - 21828.0) * 1.0 / (26043.0 - 21828.0)
    # img -= 0.17684562275397941
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.expand_dims(img, axis=-1)
    return img

def get_image_subt(img_path):
    img = misc.imread(img_path)  # load raw radiometric data
    img = img.astype('float32')
    # normalize thermal value using min-max-mean from odometry
    img = (img - 21386.0) * 1.0 / (26043.0 - 21386.0) # 21386.0,64729.0
    # img -= 0.17684562275397941
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.expand_dims(img, axis=-1)
    return img

def get_image_washington(img_path):
    img = misc.imread(img_path)  # load raw radiometric data
    img = img.astype('float32')
    dsize = (640, 512)
    img = cv2.resize(img, dsize)

    # normalize thermal value using min-max-mean from odometry
    img = (img - 21828.0) * 1.0 / (26043.0 - 21828.0)
    # img -= 0.17684562275397941
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.expand_dims(img, axis=-1)
    return img

def get_image_14bit(img_path):
    img = misc.imread(img_path)  # load raw radiometric data
    img = img.astype('float32')
    # normalize thermal value using min-max-mean from odometry
    img = (img - 21828.0) * 1.0 / (26043.0 - 21828.0)
    # img -= 0.17684562275397941
    np.clip(img, 0, 1, out=img)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.expand_dims(img, axis=-1)
    return img

def load_eval_data(dataroot, eval_exp, img_h, img_w, img_c):
    # Load data based on the image list
    sampled_path = dataroot + '/' + eval_exp + '/sampled_odom_thermal_ref_rgb_imu.csv'
    with open(sampled_path, 'r') as sampled_file:
        sampled_data = [line[:-1] for line in sampled_file]
    val_length = len(sampled_data)

    eval_data = np.zeros((val_length, img_h, img_w, img_c))
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(val_length):
        img_path = img_root_path + sampled_data[j].split(',')[0]
        img = get_image(img_path)
        # img = np.repeat(img, 3, axis=-1)
        eval_data[j, :, :, :] = img
    return eval_data

def load_eval_data_subt(dataroot, eval_exp, img_h, img_w, img_c):
    # Load data based on the image list
    sampled_path = dataroot + '/' + eval_exp + '/sampled_odom_thermal_ref_rgb_imu.csv'
    with open(sampled_path, 'r') as sampled_file:
        sampled_data = [line[:-1] for line in sampled_file]
    val_length = len(sampled_data)

    eval_data = np.zeros((val_length, img_h, img_w, img_c))
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(val_length):
        img_path = img_root_path + sampled_data[j].split(',')[0]
        img = get_image_subt(img_path)
        # img = np.repeat(img, 3, axis=-1)
        eval_data[j, :, :, :] = img
    return eval_data

def load_eval_data_washington(dataroot, eval_exp, img_h, img_w, img_c, gap):
    # Load data based on the image list
    sampled_path = dataroot + '/' + eval_exp + '/imu_1562949112967_0_clean_all_100_cut2.csv'
    with open(sampled_path, 'r') as sampled_file:
        sampled_data = [line[:-1] for line in sampled_file]
    val_length = len(sampled_data)
    sample_length = int(val_length/float(gap))-iround(float(gap))
    # print(val_length)
    eval_data = np.zeros((sample_length, img_h, img_w, img_c))
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(sample_length):
        img_path = img_root_path + sampled_data[iround(j*float(gap))].split(',')[0]
        img = get_image_washington(img_path)
        # img = np.repeat(img, 3, axis=-1)
        eval_data[j, :, :, :] = img

    print('Shape evaluation data: ' + str(np.shape(eval_data)))
    return eval_data

def load_eval_data_14bit(dataroot, eval_exp, img_h, img_w, img_c):
    # Load data based on the image list
    sampled_path = dataroot + '/' + eval_exp + '/sampled_odom_thermal_ref_rgb_imu.csv'
    with open(sampled_path, 'r') as sampled_file:
        sampled_data = [line[:-1] for line in sampled_file]
    val_length = len(sampled_data)

    eval_data = np.zeros((val_length, img_h, img_w, img_c))
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(val_length):
        img_path = img_root_path + sampled_data[j].split(',')[0]
        img = get_image_14bit(img_path)
        # img = np.repeat(img, 3, axis=-1)
        eval_data[j, :, :, :] = img

    return eval_data