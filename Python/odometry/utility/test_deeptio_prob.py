"""
Test the model using h5 files as the input
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import argparse
import numpy as np
import time
from eulerangles import mat2euler, euler2quat, euler2mat
import math
import plot_util
from os.path import join
import mdn
from keras import backend as K
from networks import build_neural_odometry
from data_tools import load_odom_data, load_data_multi_timestamp

K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))  #
K.set_learning_phase(0)  # Run testing mode

SCALER = 1.0 # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True, help='h5 file sequence, e.g. 01.h5')
    parser.add_argument('--model', type=str, required=True, help='model architecture')
    parser.add_argument('--epoch', type=str, required=True, help='which trained epoch to load in')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='specify the data dir of test data)')
    parser.add_argument('--imu_length', type=str, required=True,
                        help='specify imu length')
    parser.add_argument('--n_mixture', type=str, required=True,
                        help='number of mixture models')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test directory, e.g. val, val_dark')
    parser.add_argument('--data_type', type=str, required=True, help='specify dataset type: handheld/turtle')
    parser.add_argument('--out_pred', type=str, required=False,
                        help='specify the output of csv file for the prediction)')
    parser.add_argument('--out_gt', type=str, required=False,
                        help='specify the output of csv file for the ground truth)')
    args = parser.parse_args()

    # Define and construct model
    IMU_LENGTH = int(args.imu_length)
    n_mixtures = int(args.n_mixture)
    testest_dir = args.test_dir

    print("Building network model ......")
    deeptio_model = build_neural_odometry(join('./models', args.model, args.epoch),
                                                     imu_length=IMU_LENGTH, istraining=False,
                                                     n_mixture=n_mixtures)
    deeptio_model.summary(line_length=120)

    data_dir = args.data_dir
    test_file = join(data_dir, testest_dir, args.data_type + '_seq_' + args.seq + '.h5')

    # n_chunk, x_t, x_imu_t, y_t = load_odom_data(test_file, 'thermal')
    n_chunk, x_time, x_t, x_imu_t, y_t = load_data_multi_timestamp(test_file, 'thermal')

    y_t = y_t[0]
    print('Data shape: ', np.shape(x_t), np.shape(y_t))
    len_x_i = x_t[0].shape[0]
    print(len_x_i)

    # Set initial pose for GT and prediction
    gt_transform_t_1 = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]) # initial pose for gt
    pred_transform_t_1 = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]]) # initial pose for prediction

    # Initialize value and counter
    count_img = 0
    ls_time = [0,0,0,0]

    out_gt_array = [] # format (x,y) gt and (x,y) prediction
    out_pred_array = []  # format (x,y) gt and (x,y) prediction

    relative_poses_gt = []
    relative_poses_pred = []
    sigma_poses_pred = []

    with K.get_session() as sess:
        print('Reading images and imu ....')
        # for i in range(0, iround ((len_thermal_x_i-2)/2)):

        for i in range(0, (len_x_i-1)):

            # Make prediction
            img_1 = x_t[0][i]
            img_2 = x_t[0][i+1]

            img_1 = np.expand_dims(img_1, axis=0)
            img_2 = np.expand_dims(img_2, axis=0)
            img_1 = np.repeat(img_1, 3, axis=-1)
            img_2 = np.repeat(img_2, 3, axis=-1)
            #
            imu_t = x_imu_t[0][i + 1, 0:IMU_LENGTH, :]

            imu_t = np.expand_dims(imu_t, axis=0)

            st_cnn_time = time.time()
            predicted = sess.run([deeptio_model.outputs],
                                 feed_dict={deeptio_model.inputs[0]: img_1,
                                            deeptio_model.inputs[1]: img_2,
                                            deeptio_model.inputs[2]: imu_t})

            # print('output shape', np.shape(predicted))
            # (1, 2, 1, 1, 70)

            # Extract MDN data
            OUTPUT_DIMS = 3
            N_MIXES = n_mixtures

            pred_pos = predicted[0][0]
            pred_orient = predicted[0][1]

            temp = 1
            sigma_temp = 3 # 3 sigma
            pred_pos, sigma_pos = get_mu_sigma_MDN(pred_pos[0][0], OUTPUT_DIMS, temp, sigma_temp, N_MIXES)
            pred_orient, sigma_orient = get_mu_sigma_MDN(pred_orient[0][0], OUTPUT_DIMS, temp, sigma_temp, N_MIXES)

            relative_poses_gt.append(y_t[i])
            relative_poses_pred.append(
                [pred_pos[0], pred_pos[1], pred_pos[2], pred_orient[0], pred_orient[1], pred_orient[2]])
            sigma_poses_pred.append(
                [sigma_pos[0], sigma_pos[1], sigma_pos[2], sigma_orient[0], sigma_orient[1], sigma_orient[2]])

            final_pose = [pred_pos[0],pred_pos[1],pred_pos[2],pred_orient[0],pred_orient[1],pred_orient[2]]

            prediction_time = time.time() - st_cnn_time
            ls_time[0] += prediction_time

            print(final_pose)
            print('Running (Hz)', 1.0 / ( prediction_time))

            # Display the figure
            st_plot_time = time.time()

            # Composing the relative transformation for the prediction
            pred_transform_t = convert_rel_to_44matrix(0,0,0,final_pose)
            abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)
            # print(abs_pred_transform)

            # Composing the relative transformation for the ground truth
            gt_transform_t = convert_rel_to_44matrix(0, 0, 0, y_t[i])
            abs_gt_transform = np.dot(gt_transform_t_1, gt_transform_t)
            # print(abs_gt_transform)

            # Save the composed prediction and gt in a list
            out_gt_array.append([abs_gt_transform[0, 0], abs_gt_transform[0, 1], abs_gt_transform[0, 2], abs_gt_transform[0, 3],
                                 abs_gt_transform[1, 0], abs_gt_transform[1, 1], abs_gt_transform[1, 2], abs_gt_transform[1, 3],
                                 abs_gt_transform[2, 0], abs_gt_transform[2, 1], abs_gt_transform[2, 2], abs_gt_transform[2, 3]])

            out_pred_array.append([abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
                                     abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
                                     abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2], abs_pred_transform[2, 3]])

            plot_time = time.time() - st_plot_time
            ls_time[1] += plot_time

            gt_transform_t_1 = abs_gt_transform
            pred_transform_t_1 = abs_pred_transform
            count_img += 1

    if not os.path.exists('./results'):
        os.makedirs('./results')
    np.savetxt(join('./results', args.model + '_ep' + args.epoch + '_seq' + args.seq),
               out_pred_array, delimiter=",")
    np.savetxt(join('./results', 'gt_seq' + args.seq),
               out_gt_array, delimiter=",")
    np.savetxt(join('./results', 'gt_relpose_seq' + args.seq),
               relative_poses_gt, delimiter=",")
    np.savetxt(join('./results', 'relpose_' + args.model + '_ep' + args.epoch + '_seq' + args.seq),
               relative_poses_pred, delimiter=",")
    np.savetxt(join('./results', 'sigmapose_' + args.model + '_ep' + args.epoch + '_seq' + args.seq),
               sigma_poses_pred, delimiter=",")
    np.savetxt(join('./results', 'time_seq' + args.seq),
               x_time, delimiter="\n")

    fig_dir = join('./figs', args.model, args.seq)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    out_pred_array, out_gt_array = np.array(out_pred_array), np.array(out_gt_array)
    plot_util.plot2d(out_pred_array, out_gt_array,
                     join(fig_dir, args.model + '_ep' + args.epoch + '_seq' + args.seq + '.png'))

    for a in range(2):
        if a == 0:
            ls_time[a] = ls_time[a] / count_img
            ls_time[a] = int(round(ls_time[a] * 1000, 0))
        else:
            ls_time[a] = ls_time[a] / count_img
            ls_time[a] = int(round(ls_time[a] * 1000, 0))

    print('Model Prediction: {0} ms. Plot: {1} ms.'.format(str(ls_time[0]), str(ls_time[1])))
    print("Finished!")


def get_mu_sigma_MDN(params, output_dim, temp, sigma_temp, mixes):
    mus, sigs, pi_logits = mdn.split_mixture_params(params, output_dim, mixes)
    pis = mdn.softmax(pi_logits, t=temp)
    m = mdn.sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim] * sigma_temp  # adjust for temperature
    # cov_matrix = np.identity(output_dim) * sig_vector
    # sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return mus_vector, sig_vector


def transform44(l):
    _EPS = np.finfo(float).eps * 4.0
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)


def convert_rel_to_44matrix(rot_x, rot_y, rot_z, pose):
    R_pred = euler2mat(rot_x, rot_y, rot_z)
    rotated_pose = np.dot(R_pred, pose[0:3])
    DEGREE_2_RADIUS = np.pi / 180.0
    pred_quat = euler2quat(z=pose[5] * DEGREE_2_RADIUS, y=pose[4] * DEGREE_2_RADIUS,
                           x=pose[3] * DEGREE_2_RADIUS)
    pred_transform_t = transform44([0, rotated_pose[0], rotated_pose[1], rotated_pose[2],
                                    pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
    return pred_transform_t


def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    y = round(x) - .5
    return int(y) + (y > 0)

if __name__ == "__main__":
    os.system("hostname")
    main()
