import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
np.random.seed(0)
from pylab import *
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from networks import base_network, build_neural_embedding
from data_tools import get_positive_negative_samples, get_image
from os.path import join, dirname
import yaml
from random import randrange
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

def load_validation_stack(loop_path, dataroot, validation_exps, img_h, img_w, img_c, adjacent_frame):
    # Reserve the validation stack data
    total_val_length = 0
    for i, validation_exp in enumerate(validation_exps):
        pos_data, neg_data = get_positive_negative_samples(loop_path, validation_exp)
        total_val_length += len(pos_data)
    val_triplets = [np.zeros((total_val_length, img_h, img_w, 1)) for a in range(3)]

    # Loop for all experimental folders
    triplet_idx = 0
    for i, validation_exp in enumerate(validation_exps):
        pos_data, neg_data = get_positive_negative_samples(loop_path, validation_exp)
        np.random.shuffle(neg_data)
        img_root_path = dataroot + '/' + validation_exp + '/thermal/'
        for j in range(len(pos_data)):
            anchor_path = img_root_path + pos_data[j].split(',')[1]
            pos_path = img_root_path + pos_data[j].split(',')[3]

            anchor_idx = int(pos_data[j].split(',')[0])
            range_not_allowed = range(max(0, anchor_idx - adjacent_frame),
                                      anchor_idx + adjacent_frame)  # list of not allowed negative index

            # Get random negative example, but not within the adjacent frames of anchor images
            for trial in range(100):
                rand_ned_idx = randrange(len(neg_data))  # gen random idx whithin the range of negative samples

                neg_idx = int(neg_data[rand_ned_idx].split(',')[0])  # actual index of negative image
                if not neg_idx is range_not_allowed:  # check if the image idx is not within the adjacent anchor frames
                    neg_path = img_root_path + neg_data[rand_ned_idx].split(',')[1]
                    break

            anchor_img = get_image(anchor_path)
            # anchor_img = np.repeat(anchor_img, 3, axis=-1)
            pos_img = get_image(pos_path)
            # pos_img = np.repeat(pos_img, 3, axis=-1)
            neg_img = get_image(neg_path)
            # neg_img = np.repeat(neg_img, 3, axis=-1)

            val_triplets[0][triplet_idx, :, :, :] = anchor_img
            val_triplets[1][triplet_idx, :, :, :] = pos_img
            val_triplets[2][triplet_idx, :, :, :] = neg_img
            triplet_idx += 1

    return val_triplets

def main():
    print('Training NeuLoop for Thermal data!')

    # === Load configuration and list of training data ===
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    datatype = cfg['training_opt']['dataset']
    if datatype == 'handheld':
        dataroot = cfg['handheld_data']['dataroot']
        loop_path = cfg['handheld_data']['loop_path']
        all_experiments = cfg['handheld_data']['all_exp_files']
        training_experiments = all_experiments[0:cfg['handheld_data']['total_training']]
        n_training = len(training_experiments)
    else:
        dataroot = cfg['robot_data']['dataroot']
        loop_path = cfg['robot_data']['loop_path']
        all_experiments = cfg['robot_data']['all_exp_files']
        training_experiments = all_experiments[0:cfg['robot_data']['total_training']]
        n_training = len(training_experiments)

    MODEL_NAME = cfg['training_opt']['thermal_params']['nn_name']
    lr_rate = cfg['training_opt']['thermal_params']['lr_rate']
    decay = cfg['training_opt']['thermal_params']['decay']
    margin_loss = cfg['training_opt']['thermal_params']['margin_loss']
    img_h = cfg['training_opt']['thermal_params']['img_h']
    img_w = cfg['training_opt']['thermal_params']['img_w']
    img_c = cfg['training_opt']['thermal_params']['img_c']
    adjacent_frame = cfg['training_opt']['thermal_params']['adjacent_frame']
    descriptor_size = cfg['training_opt']['thermal_params']['descriptor_size']
    n_epoch = cfg['training_opt']['thermal_params']['epoch']
    batch_size = cfg['training_opt']['thermal_params']['batch_size']
    input_size = (img_h, img_w, img_c)
    print("Building network model: ", MODEL_NAME)
    model_dir = join('./models', MODEL_NAME)

    # === Model Definition ===
    network = base_network(input_size, descriptor_size, trainable=True)
    network_train = build_neural_embedding((img_h, img_w, 1), network, margin_loss)
    optimizer = Adam(lr=lr_rate)
    network_train.compile(loss=None, optimizer=optimizer)
    network_train.summary()
    checkpoint_path = join('./models', MODEL_NAME, 'best').format('h5')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                   verbose=1)
    tensor_board = TensorBoard(log_dir=join(model_dir, 'logs'))

    # === Load validation triplets ===
    # Validation files are the same with test file as we dont use it to learn any hyperparameters
    validation_experiments = all_experiments[cfg['robot_data']['total_training']:]
    validation_triplets = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c, adjacent_frame)
    print('Validatio size: ', np.shape(validation_triplets))

    # === Training loops ===
    for e in range(0, n_epoch): # epoch
        print("|-----> epoch %d" % e)
        # Shuffle training sequences
        np.random.shuffle(training_experiments)
        for i in range(0, n_training): # training experiments/folders
            # Load all positive-negative examples in particular sequence
            pos_data, neg_data = get_positive_negative_samples(loop_path, training_experiments[i])
            pos_length = len(pos_data)

            # Important! Shuffle both positive and negative pairs!
            np.random.shuffle(pos_data)
            np.random.shuffle(neg_data)
            print('Epoch: ', str(e), ', Sequence: ', str(i), ' - ', training_experiments[i])

            batch_iteration = int(pos_length / batch_size) # For all positive pairs
            for j in range(0, batch_iteration): # how many batch per sequences/exp
                # Initialize triplets
                triplets = [np.zeros((batch_size, img_h, img_w, 1)) for a in range(3)]
                # Get the image batch
                for k in range(0, batch_size):
                    img_root_path = dataroot + '/' + training_experiments[i] + '/thermal/'
                    anchor_path = img_root_path + pos_data[(j*batch_size)+k].split(',')[1]
                    pos_path = img_root_path + pos_data[(j*batch_size)+k].split(',')[3]

                    anchor_idx = int(pos_data[(j*batch_size)+k].split(',')[0])
                    range_not_allowed = range(max(0, anchor_idx - adjacent_frame),
                                              anchor_idx + adjacent_frame)  # list of not allowed negative index

                    # Get random negative example, but not within the adjacent frames of anchor images
                    for trial in range(100):
                        rand_ned_idx = randrange(len(neg_data))  # gen random idx whithin the range of negative samples

                        neg_idx = int(neg_data[rand_ned_idx].split(',')[0])  # actual index of negative image
                        if not neg_idx is range_not_allowed:  # check if the image idx is not within the adjacent anchor frames
                            neg_path = img_root_path + neg_data[rand_ned_idx].split(',')[1]
                            break

                    anchor_img = get_image(anchor_path)
                    # anchor_img = np.repeat(anchor_img, 3, axis=-1)
                    pos_img = get_image(pos_path)
                    # pos_img = np.repeat(pos_img, 3, axis=-1)
                    neg_img = get_image(neg_path)
                    # neg_img = np.repeat(neg_img, 3, axis=-1)

                    triplets[0][k, :, :, :] = anchor_img
                    triplets[1][k, :, :, :] = pos_img
                    triplets[2][k, :, :, :] = neg_img

                print(np.shape(triplets))
                # Implement Get batch hard for hard triplet loss here!!!

                if i == (n_training - 1) and j == (batch_iteration - 1):
                    # Train on batch and validate
                    network_train.fit(x=[triplets[0], triplets[1], triplets[2]], y=None, verbose=1,
                                      validation_data=([validation_triplets[0], validation_triplets[1], validation_triplets[2]], None),
                                      callbacks=[checkpointer, tensor_board])
                else:
                    network_train.fit(x=[triplets[0], triplets[1], triplets[2]], y=None, verbose=1) # Train on batch

        if ((e % 50) == 0):
            network_train.save(join(model_dir, str(e).format('h5')))

if __name__ == "__main__":
    os.system("hostname")
    main()