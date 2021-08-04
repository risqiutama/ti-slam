'''
Test pose between loop pairs
'''
import os
from os.path import join
import yaml
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

def main():
    # === Load configuration and list of training data ===
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    # Evaluation Parameters
    model = cfg['loop_evaluation_opt']['model_name']
    evaluation_experiments = cfg['loop_evaluation_opt']['exp_files']

    dataroot = cfg['loop_evaluation_opt']['dataroot']
    loop_pairs_path = cfg['loop_evaluation_opt']['loop_pairs_path']
    net_name = cfg['loop_evaluation_opt']['loop_detection_network']
    thres = cfg['loop_evaluation_opt']['loop_threshold']
    img_h = cfg['nn_opt']['loop_params']['img_h']
    img_w = cfg['nn_opt']['loop_params']['img_w']
    img_c = cfg['nn_opt']['loop_params']['img_c']

    # Evaluate the model we defined
    print('Test Model {}'.format(model))
    epochs = []
    epochs.append('best')
    print(epochs)

    # Generate the output
    for i, eval_exp in enumerate(evaluation_experiments):
        for epoch in epochs:
            cmd = 'python -W ignore ' + 'utility/test_loop_pose_prob.py ' + '--eval_exp ' + eval_exp + ' --model ' + \
                  model + ' --epoch ' + epoch + ' --dataroot ' + dataroot + ' --loop_path ' + loop_pairs_path + ' --net_name ' +\
                  net_name + ' --thres ' + str(thres) + ' --img_h ' + str(img_h) + ' --img_w ' + str(img_w) + ' --img_c ' + str(img_c)
            os.system(cmd)

if __name__ == "__main__":
    os.system("hostname")
    main()