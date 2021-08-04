import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
np.random.seed(0)
from pylab import *
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from utility.networks import build_neural_loop_closure
from utility.data_tools import get_pose_pairs, get_image
import os
from os.path import join
import yaml
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def load_validation_stack(loop_path, dataroot, validation_exps, img_h, img_w, img_c):
    # Reserve the validation stack data
    total_val_length = 0
    for i, validation_exp in enumerate(validation_exps):
        pose_data = get_pose_pairs(loop_path, validation_exp)
        total_val_length += len(pose_data)
    x_image_1 = np.zeros((total_val_length, 1, img_h, img_w, 3))
    x_image_2 = np.zeros((total_val_length, 1, img_h, img_w, 3))
    y_pose = np.zeros((total_val_length, 1, 6))
    print('Allocated validation data: ' + str(np.shape(x_image_1)) + ' - ' + str(np.shape(x_image_2)) + ' - ' + str(np.shape(y_pose)))

    # Loop for all experimental folders
    val_idx = 0
    for i, validation_exp in enumerate(validation_exps):
        pose_data = get_pose_pairs(loop_path, validation_exp)

        img_root_path = dataroot + '/' + validation_exp + '/thermal/'
        for j in range(len(pose_data)):
            image_1_path = img_root_path + pose_data[j].split(',')[1]
            image_2_path = img_root_path + pose_data[j].split(',')[3]

            img_1 = get_image(image_1_path)
            img_1 = np.repeat(img_1, 3, axis=-1)
            img_2 = get_image(image_2_path)
            img_2 = np.repeat(img_2, 3, axis=-1)

            x_image_1[val_idx, 0, :, :, :] = img_1
            x_image_2[val_idx, 0, :, :, :] = img_2
            for idx_p in range(6):
                y_pose[val_idx, 0, idx_p] = float(pose_data[j].split(',')[4+idx_p])
            val_idx += 1

    return x_image_1, x_image_2, y_pose


def main():
    print('Training NeuLoop for Thermal data!')

    # === Load configuration and list of training data ===
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    datatype = cfg['train_loop_pose_opt']['dataset']
    if datatype == 'handheld':
        dataroot = cfg['loop_handheld_data']['dataroot']
        loop_path = cfg['loop_handheld_data']['loop_path']
        all_experiments = cfg['loop_handheld_data']['all_exp_files']
        training_experiments = all_experiments[0:cfg['loop_handheld_data']['total_training']]
        n_training = len(training_experiments)
    else:
        dataroot = cfg['loop_robot_data']['dataroot']
        loop_path = cfg['loop_robot_data']['loop_path']
        all_experiments = cfg['loop_robot_data']['all_exp_files']
        training_experiments = all_experiments[0:cfg['loop_robot_data']['total_training']]
        n_training = len(training_experiments)

    MODEL_NAME = cfg['nn_opt']['loop_params']['nn_name']
    img_h = cfg['nn_opt']['loop_params']['img_h']
    img_w = cfg['nn_opt']['loop_params']['img_w']
    img_c = cfg['nn_opt']['loop_params']['img_c']
    n_epoch = cfg['nn_opt']['loop_params']['epoch']
    batch_size = cfg['nn_opt']['loop_params']['batch_size']
    input_size = (img_h, img_w, img_c)
    print("Building network model: ", MODEL_NAME)
    model_dir = join('./models', MODEL_NAME)

    # === Model Definition ===
    network_train = build_neural_loop_closure(cfg['nn_opt']['loop_params'])

    checkpoint_path = join('./models', MODEL_NAME, 'best').format('h5')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                   verbose=1)
    tensor_board = TensorBoard(log_dir=join(model_dir, 'logs'))

    # regulate learning rate
    def step_decay(epoch):
        initial_lrate = cfg['nn_opt']['loop_params']['lr_rate']  # 0.001, 0.0001
        drop = 0.75
        epochs_drop = 25.0
        lrate = initial_lrate * math.pow(drop,
                                         math.floor((1 + epoch) / epochs_drop))
        print('Learning rate: ' + str(lrate))
        return lrate
    lrate = LearningRateScheduler(step_decay)

    # === Load validation poses ===
    # Validation files are the same with test file as we dont use it to learn any hyperparameters
    validation_experiments = all_experiments[cfg['loop_robot_data']['total_training']:]

    x_val_img_1, x_val_img_2, y_val = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c)
    print('Validation size: ' + str(np.shape(x_val_img_1)) + ' - ' + str(np.shape(x_val_img_2)))

    # === Training loops ===
    for e in range(0, n_epoch): # epoch
        print("|-----> epoch %d" % e)
        # Shuffle training sequences
        np.random.shuffle(training_experiments)
        for i in range(0, n_training): # training experiments/folders
            # Load all positive-negative examples in particular sequence
            pose_data = get_pose_pairs(loop_path, training_experiments[i])
            data_length = len(pose_data)

            # Important! Shuffle the data!
            np.random.shuffle(pose_data)
            print('Epoch: ', str(e), ', Sequence: ', str(i), ' - ', training_experiments[i])

            batch_iteration = int(data_length / batch_size) # For all positive pairs
            for j in range(0, batch_iteration): # how many batch per sequences/exp
                # Initialize training batches
                x_img_1 = np.zeros((batch_size, 1, img_h, img_w, 3))
                x_img_2 = np.zeros((batch_size, 1, img_h, img_w, 3))
                y_pose = np.zeros((batch_size, 1, 6))
                # Get the image batch
                for k in range(0, batch_size):
                    img_root_path = dataroot + '/' + training_experiments[i] + '/thermal/'
                    img_1_path = img_root_path + pose_data[(j*batch_size)+k].split(',')[1]
                    img_2_path = img_root_path + pose_data[(j*batch_size)+k].split(',')[3]

                    img_1 = get_image(img_1_path)
                    img_1 = np.repeat(img_1, 3, axis=-1)

                    img_2 = get_image(img_2_path)
                    img_2 = np.repeat(img_2, 3, axis=-1)

                    x_img_1[k, 0, :, :, :] = img_1
                    x_img_2[k, 0, :, :, :] = img_2

                    for idx_p in range(6):
                        y_pose[k, 0, idx_p] = float(pose_data[(j*batch_size)+k].split(',')[4 + idx_p])

                # Implement Get batch hard for hard triplet loss here!!!
                if i == (n_training - 1) and j == (batch_iteration - 1):
                    # Train on batch and validate
                    network_train.fit(x=[x_img_1, x_img_2], y=[y_pose[:, :, 0:3], y_pose[:, :, 3:6]], verbose=1,
                                      validation_data=([x_val_img_1, x_val_img_2],
                                                       [y_val[:, :, 0:3], y_val[:, :, 3:6]]),
                                      # callbacks=[checkpointer, lrate, tensor_board])
                                      callbacks = [checkpointer, tensor_board])
                else:
                    network_train.fit(x=[x_img_1, x_img_2], y=[y_pose[:, :, 0:3], y_pose[:, :, 3:6]], verbose=1) # Train on batch

        if ((e % 50) == 0):
            network_train.save(join(model_dir, str(e).format('h5')))

if __name__ == "__main__":
    os.system("hostname")
    main()