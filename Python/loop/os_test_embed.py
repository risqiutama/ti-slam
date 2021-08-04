import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
    model = cfg['os_test_embed']['model_name']
    evaluation_experiments = cfg['os_test_embed']['exp_files']

    print('Test Model {}'.format(model))
    epochs = []
    epochs.append('best')
    print(epochs)

    # Generate descriptor for each sequences and each model epoch (saved)
    for i, eval_exp in enumerate(evaluation_experiments):
        for epoch in epochs:
            cmd = 'python -W ignore ' + 'generate_embedding.py ' + '--eval_exp ' + eval_exp + ' --model ' + \
                  model + ' --epoch ' + epoch
            os.system(cmd)



if __name__ == "__main__":
    os.system("hostname")
    main()