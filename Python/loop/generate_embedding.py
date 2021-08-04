import numpy as np
import os
import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# import tools.data_tools as dt_tools
# from tools import data_tools
from data_tools import get_positive_negative_samples, get_image, load_eval_data
from networks import base_network, build_neural_embedding
import os
from os.path import join, dirname
from scipy import misc
import yaml
import time
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import argparse

def main():
    DESCRIPTION = """This script generate the output embedding from a sequence and save it in a file."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--eval_exp', help='Experiment folder')
    parser.add_argument('--model', help='Model name for evaluation')
    parser.add_argument('--epoch', help='Model epoch')
    args = parser.parse_args()

    eval_exp = args.eval_exp
    model = args.model
    epoch = args.epoch

    # === Load configuration and list of training data ===
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    dataroot = cfg['os_test_embed']['dataroot']
    img_h = cfg['training_opt']['thermal_params']['img_h']
    img_w = cfg['training_opt']['thermal_params']['img_w']
    img_c = cfg['training_opt']['thermal_params']['img_c']
    descriptor_size = cfg['training_opt']['thermal_params']['descriptor_size']
    input_size = (img_h, img_w, img_c)

    # === Data Loader ===
    eval_data = load_eval_data(dataroot, eval_exp, img_h, img_w, img_c)
    print('Load evaluation data: ', eval_exp, ' ', np.shape(eval_data))

    # === Model Definition ===
    network = base_network(input_size, descriptor_size, trainable=False)
    model_path = join('./models', model, epoch)
    network.load_weights(model_path, by_name=True)
    st_network_time = time.time()
    embed_descriptor = network.predict(eval_data)
    prediction_time = time.time() - st_network_time
    print('Average exec time ' + str(float(prediction_time/len(eval_data))))
    print(len(eval_data))
    print(np.shape(eval_data))

    list_embedding = []
    for j, embedding in enumerate(embed_descriptor):
        print(np.shape(embedding))
        list_embedding.append(embedding)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    np.savetxt(join('./results', 'embedding_' + model + '_ep' + epoch + '_' + eval_exp + '.csv'), list_embedding, delimiter=",")


if __name__ == "__main__":
    os.system("hostname")
    main()