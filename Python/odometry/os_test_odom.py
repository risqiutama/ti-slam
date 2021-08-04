import os
import yaml
from os.path import join
import inspect
import glob
import re

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
with open(join(currentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

model = cfg['os_odom_test']['model']
dataset_type = cfg['os_odom_test']['dataset']
data_dir = cfg['os_odom_test']['saved_dir_h5']
IMU_LENGTH = str(cfg['os_odom_test']['imu_length'])
n_mixture = str(cfg['os_odom_test']['n_mixture'])
test_dir = str(cfg['os_odom_test']['test_dir'])

test_files = glob.glob(join(data_dir, test_dir, '*.h5'))
print(test_files)

seqs = [re.search(dataset_type + '_seq_(.+?).h5', file).group(1) for file in test_files]


print('Test Model {}'.format(model))
# model_dir = join('./models', model)
# max_epochs = max([int(x) for x in os.listdir(model_dir) if str.isdigit(x[0])])
# epochs = [str(r) for r in range(0, max_epochs + 25, 25)]
epochs = [] # if you just want the best
epochs.append('best')
print(epochs)

for seq in seqs:
    for epoch in epochs:
        print(epoch)
        cmd = 'python -W ignore ' + 'utility/test_deeptio_prob.py' + ' ' + '--seq ' + str(seq) + ' ' + '--model ' + \
              model + ' --epoch ' + epoch + ' ' + '--data_dir ' + data_dir + ' ' + '--imu_length ' + IMU_LENGTH + \
            ' ' + '--data_type ' + dataset_type + ' ' + '--n_mixture ' + n_mixture + ' ' + '--test_dir ' + test_dir
        print(cmd)
        os.system(cmd)