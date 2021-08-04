import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import inspect
import numpy as np
from os.path import join

from data_tools import load_eval_data_from_neuloop
from networks import build_neural_loop_closure
from eulerangles import euler2quat
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))  #
K.set_learning_phase(0)  # Run testing mode

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import argparse
import mdn

def main():
    DESCRIPTION = """This script generate the pose from loop pose network."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--eval_exp', help='Experiment folder')
    parser.add_argument('--model', help='Model name for evaluation')
    parser.add_argument('--epoch', help='Model epoch')
    parser.add_argument('--dataroot', help='Dataroot')
    parser.add_argument('--loop_path', help='Path to the loop closure data')
    parser.add_argument('--net_name', help='Loop closure detection network name')
    parser.add_argument('--thres', help='Threshold for identifying loop pairs')
    parser.add_argument('--img_h', help='Image height')
    parser.add_argument('--img_w', help='Image width')
    parser.add_argument('--img_c', help='Image channels')
    args = parser.parse_args()

    # === Load configuration and list of training data ===
    eval_exp = args.eval_exp
    model = args.model
    epoch = args.epoch
    dataroot = args.dataroot
    loop_path = args.loop_path
    net_name = args.net_name
    thres = float(args.thres)
    img_h = int(args.img_h)
    img_w = int(args.img_w)
    img_c = int(args.img_c)

    # === Data Loader ===
    # x_image_1, x_image_2, loop_pairs_data = load_eval_data(dataroot, loop_path, eval_exp, img_h, img_w, img_c)
    print('Processing data: ' + eval_exp)
    x_image_1, x_image_2, loop_pairs_data = load_eval_data_from_neuloop(dataroot, loop_path, net_name, eval_exp, thres,
                                                                        img_h, img_w, img_c)
    len_x_i = np.shape(x_image_1)[0]
    print('Load evaluation data: ', eval_exp, ' ', np.shape(x_image_1), ' , length: ', str(len_x_i))

    # === Model Definition ===
    network = build_neural_loop_closure(join('./models', model, epoch), istraining=False)
    # model_path = join('./models', model, epoch)
    # network.load_weights(model_path, by_name=True)
    network.summary()
    # predicted_poses = network.predict([x_image_1, x_image_2])
    #
    # print('Predicted pose' + str(np.shape(predicted_poses)))

    relative_poses_pred = []
    sigma_poses_pred = []

    if not os.path.exists('./results'):
        os.makedirs('./results')
    result_path = join('./results', 'pose_' + model + '_' + net_name + '_ep' + epoch + '_' + eval_exp + '_' + str(thres) + '.csv')

    with K.get_session() as sess:
        with open(result_path, 'w') as result_file:
            print('Reading images ....')
            # for i in range(0, iround ((len_thermal_x_i-2)/2)):
            for i in range(0, len_x_i):

                img_1 = np.expand_dims(x_image_1[i], axis=0)
                img_2 = np.expand_dims(x_image_2[i], axis=0)
                predicted = sess.run([network.outputs],
                                     feed_dict={network.inputs[0]: img_1,
                                                network.inputs[1]: img_2})
                # print('output shape', np.shape(predicted))
                # (1, 2, 1, 1, 70)

                # Extract MDN data
                OUTPUT_DIMS = 3
                N_MIXES = 10

                pred_pos = predicted[0][0]
                pred_orient = predicted[0][1]

                temp = 1
                sigma_temp = 3  # 3 sigma
                pred_pos, sigma_pos = get_mu_sigma_MDN(pred_pos[0][0], OUTPUT_DIMS, temp, sigma_temp, N_MIXES)
                pred_orient, sigma_orient = get_mu_sigma_MDN(pred_orient[0][0], OUTPUT_DIMS, temp, sigma_temp, N_MIXES)

                relative_poses_pred.append(
                    [pred_pos[0], pred_pos[1], pred_pos[2], pred_orient[0], pred_orient[1], pred_orient[2]])
                sigma_poses_pred.append(
                    [sigma_pos[0], sigma_pos[1], sigma_pos[2], sigma_orient[0], sigma_orient[1], sigma_orient[2]])

                write_line = str(loop_pairs_data[i].split(',')[0]) + ',' + str(loop_pairs_data[i].split(',')[1]) + ',' + \
                                    str(loop_pairs_data[i].split(',')[2]) + ',' + str(loop_pairs_data[i].split(',')[3].rstrip('\n')) + ',' + \
                                    str(pred_pos[0]) + ',' + str(pred_pos[1]) + ',' + str(pred_pos[2]) + ',' + str(pred_orient[0]) + ',' + \
                                    str(pred_orient[1]) + ',' + str(pred_orient[2]) + ',' + str(sigma_pos[0]) + ',' + str(sigma_pos[1]) + ',' + \
                                    str(sigma_pos[2]) + ',' + str(sigma_orient[0]) + ',' + str(sigma_orient[1]) + ',' + str(sigma_orient[2]) + '\n'
                print(write_line)
                result_file.write(write_line)

def get_mu_sigma_MDN(params, output_dim, temp, sigma_temp, mixes):
    mus, sigs, pi_logits = mdn.split_mixture_params(params, output_dim, mixes)
    pis = mdn.softmax(pi_logits, t=temp)
    m = mdn.sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim] * sigma_temp  # adjust for temperature

    return mus_vector, sig_vector

if __name__ == "__main__":
    os.system("hostname")
    main()