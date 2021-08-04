from scipy import misc
import numpy as np
import cv2
import h5py

def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    y = round(x) - .5
    return int(y) + (y > 0)

def get_pose_pairs(loop_path,  exp_folder):
    loop_pairs_file = loop_path + '/' + exp_folder + '/rgb_sampled/rel_pose_loop_pairs.csv'
    with open(loop_pairs_file, 'r') as pose_file:
        pose_data = [line[:-1] for line in pose_file]
    return pose_data

def get_pose_pairs_subt(loop_path,  exp_folder):
    loop_pairs_file = loop_path + '/' + exp_folder + '/rgb_left_sampled/rel_pose_loop_pairs.csv'
    with open(loop_pairs_file, 'r') as pose_file:
        pose_data = [line[:-1] for line in pose_file]
    return pose_data

def get_image(img_path):
    # img = misc.imread(img_path)  # load raw radiometric data
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32')
    # normalize thermal value using min-max-mean from odometry
    img = (img - 21828.0) * 1.0 / (26043.0 - 21828.0)
    img -= 0.17684562275397941
    img = np.expand_dims(img, axis=-1)
    return img

def get_image_subt(img_path):
    # img = misc.imread(img_path)  # load raw radiometric data
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32')
    # normalize thermal value using min-max-mean from odometry
    img = (img - 21386.0) * 1.0 / (64729.0 - 21386.0) # # 21386.0,64729.0
    img -= 0.02886447479939211
    img = np.expand_dims(img, axis=-1)
    return img

def get_image_washington(img_path):
    # img = misc.imread(img_path)  # load raw radiometric data
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32')
    dsize = (640, 512)
    img = cv2.resize(img, dsize)
    # normalize thermal value using min-max-mean from odometry
    img = (img - 21828.0) * 1.0 / (26043.0 - 21828.0)
    img -= 0.17684562275397941
    img = np.expand_dims(img, axis=-1)
    return img

def load_eval_data(dataroot, loop_path, eval_exp, img_h, img_w, img_c):
    # Load data based on the image list
    loop_pairs_file = loop_path + '/' + eval_exp + '/rgb_sampled/raw_loop.csv' # rel_pose_loop_pairs.csv
    with open(loop_pairs_file, 'r') as pose_file:
        pose_data = [line[:-1] for line in pose_file]
    eval_length = len(pose_data)

    x_image_1 = np.zeros((eval_length, 1, img_h, img_w, 3))
    x_image_2 = np.zeros((eval_length, 1, img_h, img_w, 3))

    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(eval_length):
        image_1_path = img_root_path + pose_data[j].split(',')[1]
        image_2_path = img_root_path + pose_data[j].split(',')[3]

        img_1 = get_image(image_1_path)
        img_1 = np.repeat(img_1, 3, axis=-1)
        img_2 = get_image(image_2_path)
        img_2 = np.repeat(img_2, 3, axis=-1)

        x_image_1[j, 0, :, :, :] = img_1
        x_image_2[j, 0, :, :, :] = img_2

    return x_image_1, x_image_2, pose_data

def load_eval_data_from_neuloop(dataroot, loop_path, net_name, eval_exp, thres, img_h, img_w, img_c):
    # Load data based on the image list
    loop_pairs_file = loop_path + '/' + net_name + '_' + eval_exp + '_' + str(thres) + '.csv' # rel_pose_loop_pairs.csv
    with open(loop_pairs_file, 'r') as pose_file:
        loop_data = [line[:-1] for line in pose_file]
    eval_length = len(loop_data)

    img_name_path = dataroot + '/' + eval_exp + '/' + 'sampled_odom_thermal_ref_rgb_imu.csv'
    with open(img_name_path, 'r') as sensor_file:
        sensor_lines = [line[:-1] for line in sensor_file]

    x_image_1 = np.zeros((eval_length, 1, img_h, img_w, 3))
    x_image_2 = np.zeros((eval_length, 1, img_h, img_w, 3))

    loop_data_complete = []
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(eval_length):
        # image_1_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[0])].split(',')[0]
        # image_2_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[1])].split(',')[0]

        image_1_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[0])-1].split(',')[0] # minus one because indexing in matlab
        image_2_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[1])-1].split(',')[0]

        img_1 = get_image(image_1_path)
        img_1 = np.repeat(img_1, 3, axis=-1)
        img_2 = get_image(image_2_path)
        img_2 = np.repeat(img_2, 3, axis=-1)

        x_image_1[j, 0, :, :, :] = img_1
        x_image_2[j, 0, :, :, :] = img_2

        str_loop_data = loop_data[j].split(',')[0] + ',' + sensor_lines[int(loop_data[j].split(',')[0])-1].split(',')[0] \
                             + ',' + loop_data[j].split(',')[1] + ',' + sensor_lines[int(loop_data[j].split(',')[1])-1].split(',')[0]
        loop_data_complete.append(str_loop_data)

    return x_image_1, x_image_2, loop_data_complete

def load_eval_data_from_neuloop_subt(dataroot, loop_path, net_name, eval_exp, thres, img_h, img_w, img_c):
    # Load data based on the image list
    loop_pairs_file = loop_path + '/' + net_name + '_' + eval_exp + '_' + str(thres) + '.csv' # rel_pose_loop_pairs.csv
    with open(loop_pairs_file, 'r') as pose_file:
        loop_data = [line[:-1] for line in pose_file]
    eval_length = len(loop_data)

    img_name_path = dataroot + '/' + eval_exp + '/' + 'sampled_odom_thermal_ref_rgb_imu.csv'
    with open(img_name_path, 'r') as sensor_file:
        sensor_lines = [line[:-1] for line in sensor_file]

    x_image_1 = np.zeros((eval_length, 1, img_h, img_w, 3))
    x_image_2 = np.zeros((eval_length, 1, img_h, img_w, 3))

    loop_data_complete = []
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(eval_length):
        # image_1_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[0])].split(',')[0]
        # image_2_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[1])].split(',')[0]

        image_1_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[0])-1].split(',')[0] # minus one because indexing in matlab
        image_2_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[1])-1].split(',')[0]

        img_1 = get_image_subt(image_1_path)
        img_1 = np.repeat(img_1, 3, axis=-1)
        img_2 = get_image_subt(image_2_path)
        img_2 = np.repeat(img_2, 3, axis=-1)

        x_image_1[j, 0, :, :, :] = img_1
        x_image_2[j, 0, :, :, :] = img_2

        str_loop_data = loop_data[j].split(',')[0] + ',' + sensor_lines[int(loop_data[j].split(',')[0])-1].split(',')[0] \
                             + ',' + loop_data[j].split(',')[1] + ',' + sensor_lines[int(loop_data[j].split(',')[1])-1].split(',')[0]
        loop_data_complete.append(str_loop_data)

    return x_image_1, x_image_2, loop_data_complete

def load_eval_data_from_neuloop_washington(dataroot, loop_path, net_name, eval_exp, thres, img_h, img_w, img_c):
    # Load data based on the image list
    loop_pairs_file = loop_path + '/' + net_name + '_' + eval_exp + '_' + str(thres) + '.csv' # rel_pose_loop_pairs.csv
    with open(loop_pairs_file, 'r') as pose_file:
        loop_data = [line[:-1] for line in pose_file]
    eval_length = len(loop_data)

    img_name_path = dataroot + '/' + eval_exp + '/imu_1562949112967_0_clean_all_100_cut3.csv'
    with open(img_name_path, 'r') as sensor_file:
        sensor_lines = [line[:-1] for line in sensor_file]

    x_image_1 = np.zeros((eval_length, 1, img_h, img_w, 3))
    x_image_2 = np.zeros((eval_length, 1, img_h, img_w, 3))

    gap = 3.5
    loop_data_complete = []
    img_root_path = dataroot + '/' + eval_exp + '/thermal/'
    for j in range(eval_length):
        # image_1_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[0])].split(',')[0]
        # image_2_path = img_root_path + sensor_lines[int(loop_data[j].split(',')[1])].split(',')[0]

        image_1_path = img_root_path + sensor_lines[iround((int(loop_data[j].split(',')[0])-1) * float(gap))].split(',')[0] # minus one because indexing in matlab
        image_2_path = img_root_path + sensor_lines[iround((int(loop_data[j].split(',')[1])-1) * float(gap))].split(',')[0]

        img_1 = get_image_washington(image_1_path)
        img_1 = np.repeat(img_1, 3, axis=-1)
        img_2 = get_image_washington(image_2_path)
        img_2 = np.repeat(img_2, 3, axis=-1)

        x_image_1[j, 0, :, :, :] = img_1
        x_image_2[j, 0, :, :, :] = img_2

        str_loop_data = loop_data[j].split(',')[0] + ',' + sensor_lines[int(loop_data[j].split(',')[0])-1].split(',')[0] \
                             + ',' + loop_data[j].split(',')[1] + ',' + sensor_lines[int(loop_data[j].split(',')[1])-1].split(',')[0]
        loop_data_complete.append(str_loop_data)

    return x_image_1, x_image_2, loop_data_complete

def odom_validation_stack(validation_files, sensor='thermal', imu_length=0):
    x_sensor_val_1, x_sensor_val_2, x_imu_val_t, y_val_t = [], [], [], []
    for validation_file in validation_files:
        print('---> Loading validation file: {}'.format(validation_file.split('/')[-1]))
        if imu_length:
            n_chunk_val, tmp_x_sensor_val_t, tmp_x_imu_val_t, tmp_y_val_t = load_odom_data(validation_file, sensor) # y (1, 2142, 6)
        else:
            n_chunk_val, tmp_x_sensor_val_t, tmp_y_val_t = load_data_single_sensor(validation_file, sensor)

        tmp_y_val_t = tmp_y_val_t[0]
        tmp_y_val_t = np.expand_dims(tmp_y_val_t, axis=1)

        len_val_i = tmp_y_val_t.shape[0] # the length of gt is always less than the length of data
        # Prepare rgb validation data for t-0 and t-1
        tmp_x_sensor_val_1 = []
        for img_idx in range(0, (len_val_i)):
            temp_x = tmp_x_sensor_val_t[0][img_idx]
            tmp_x_sensor_val_1.append(temp_x)

        tmp_x_sensor_val_1 = np.array(tmp_x_sensor_val_1)
        # x_rgb_val_1 = np.expand_dims(x_rgb_val_1, axis=1)

        tmp_x_sensor_val_2 = []
        for img_idx in range(1, (len_val_i+1)):
            temp_x = tmp_x_sensor_val_t[0][img_idx]
            tmp_x_sensor_val_2.append(temp_x)

        tmp_x_sensor_val_2 = np.array(tmp_x_sensor_val_2)

        # for flownet
        if any(x in sensor for x in ['mmwave', 'depth', 'thermal']):
            tmp_x_sensor_val_1 = np.repeat(tmp_x_sensor_val_1, 3, axis=-1)
            tmp_x_sensor_val_2 = np.repeat(tmp_x_sensor_val_2, 3, axis=-1)

        # progressive stack file by file
        y_val_t = np.vstack((y_val_t, tmp_y_val_t)) if np.array(y_val_t).size else tmp_y_val_t
        x_sensor_val_1 = np.vstack((x_sensor_val_1, tmp_x_sensor_val_1)) if np.array(x_sensor_val_1).size else tmp_x_sensor_val_1
        x_sensor_val_2 = np.vstack((x_sensor_val_2, tmp_x_sensor_val_2)) if np.array(x_sensor_val_2).size else tmp_x_sensor_val_2

        if imu_length:
            # for imu
            tmp_x_imu_val_t = tmp_x_imu_val_t[0]
            tmp_x_imu_val_t = tmp_x_imu_val_t[:, 0:imu_length, :]
            tmp_x_imu_val_t = np.array(tmp_x_imu_val_t)

            # add data
            x_imu_val_t = np.vstack((x_imu_val_t, tmp_x_imu_val_t)) \
                if np.array(x_imu_val_t).size else tmp_x_imu_val_t

    if imu_length:
        return x_sensor_val_1, x_sensor_val_2, x_imu_val_t, y_val_t
    else:
        return x_sensor_val_1, x_sensor_val_2, y_val_t

def odom_validation_stack_imu(validation_files, sensor='rgb', imu_length=0):
    x_imu_val_t, y_val_t = [], []
    for validation_file in validation_files:
        print('---> Loading validation file: {}'.format(validation_file.split('/')[-1]))
        if imu_length:
            n_chunk_val, tmp_x_sensor_val_t, tmp_x_imu_val_t, tmp_y_val_t = load_odom_data(validation_file, sensor) # y (1, 2142, 6)
        else:
            n_chunk_val, tmp_x_sensor_val_t, tmp_y_val_t = load_data_single_sensor(validation_file, sensor)

        tmp_y_val_t = tmp_y_val_t[0]
        tmp_y_val_t = np.expand_dims(tmp_y_val_t, axis=1)

        # progressive stack file by file
        y_val_t = np.vstack((y_val_t, tmp_y_val_t)) if np.array(y_val_t).size else tmp_y_val_t

        if imu_length:
            # for imu
            tmp_x_imu_val_t = tmp_x_imu_val_t[0]
            tmp_x_imu_val_t = tmp_x_imu_val_t[:, 0:imu_length, :]
            tmp_x_imu_val_t = np.array(tmp_x_imu_val_t)

            # add data
            x_imu_val_t = np.vstack((x_imu_val_t, tmp_x_imu_val_t)) \
                if np.array(x_imu_val_t).size else tmp_x_imu_val_t

    return x_imu_val_t, y_val_t

def odom_validation_stack_hallucination(validation_files, hallucination_files, sensor='thermal', imu_length=0):
    x_sensor_val_1, x_sensor_val_2, x_imu_val_t, y_val_t, y_rgb_feat_val_t = [], [], [], [], []
    print(validation_files)
    print(hallucination_files)
    for i, validation_file in enumerate(validation_files):
        print('---> Loading validation file: {}'.format(validation_file.split('/')[-1]))
        if imu_length:
            n_chunk_val, tmp_x_sensor_val_t, tmp_x_imu_val_t, tmp_y_val_t = load_odom_data(validation_file,
                                                                                           sensor)  # y (1, 2142, 6)
        else:
            n_chunk_val, tmp_x_sensor_val_t, tmp_y_val_t = load_data_single_sensor(validation_file, sensor)

        n_chunk_feat, tmp_y_rgb_feat_val_t = load_hallucination_data(hallucination_files[i])
        print('Loading validation and hallucination: ', validation_file, hallucination_files[i])
        tmp_y_rgb_feat_val_t = tmp_y_rgb_feat_val_t[0]

        tmp_y_val_t = tmp_y_val_t[0]
        tmp_y_val_t = np.expand_dims(tmp_y_val_t, axis=1)

        len_val_i = tmp_y_val_t.shape[0]  # the length of gt is always less than the length of data
        # Prepare rgb validation data for t-0 and t-1
        tmp_x_sensor_val_1 = []
        for img_idx in range(0, (len_val_i)):
            temp_x = tmp_x_sensor_val_t[0][img_idx]
            tmp_x_sensor_val_1.append(temp_x)

        tmp_x_sensor_val_1 = np.array(tmp_x_sensor_val_1)
        # x_rgb_val_1 = np.expand_dims(x_rgb_val_1, axis=1)

        tmp_x_sensor_val_2 = []
        for img_idx in range(1, (len_val_i + 1)):
            temp_x = tmp_x_sensor_val_t[0][img_idx]
            tmp_x_sensor_val_2.append(temp_x)

        tmp_x_sensor_val_2 = np.array(tmp_x_sensor_val_2)

        # for flownet
        if any(x in sensor for x in ['mmwave', 'depth', 'thermal']):
            tmp_x_sensor_val_1 = np.repeat(tmp_x_sensor_val_1, 3, axis=-1)
            tmp_x_sensor_val_2 = np.repeat(tmp_x_sensor_val_2, 3, axis=-1)

        # progressive stack file by file
        y_val_t = np.vstack((y_val_t, tmp_y_val_t)) if np.array(y_val_t).size else tmp_y_val_t
        x_sensor_val_1 = np.vstack((x_sensor_val_1, tmp_x_sensor_val_1)) if np.array(
            x_sensor_val_1).size else tmp_x_sensor_val_1
        x_sensor_val_2 = np.vstack((x_sensor_val_2, tmp_x_sensor_val_2)) if np.array(
            x_sensor_val_2).size else tmp_x_sensor_val_2

        y_rgb_feat_val_t = np.vstack((y_rgb_feat_val_t, tmp_y_rgb_feat_val_t)) if np.array(
            y_rgb_feat_val_t).size else tmp_y_rgb_feat_val_t

        if imu_length:
            # for imu
            tmp_x_imu_val_t = tmp_x_imu_val_t[0]
            tmp_x_imu_val_t = tmp_x_imu_val_t[:, 0:imu_length, :]
            tmp_x_imu_val_t = np.array(tmp_x_imu_val_t)

            # add data
            x_imu_val_t = np.vstack((x_imu_val_t, tmp_x_imu_val_t)) \
                if np.array(x_imu_val_t).size else tmp_x_imu_val_t

    if imu_length:
        return x_sensor_val_1, x_sensor_val_2, x_imu_val_t, y_val_t, y_rgb_feat_val_t
    else:
        return x_sensor_val_1, x_sensor_val_2, y_val_t

# load multi-modal data with IMU always in
def load_odom_data(training_file, sensor):

    # Load data
    x_sensor, x_imu, y = [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_sensor_temp = hdf5_file.get(sensor+'_data')
    x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')

    print('Data shape: ' + str(np.shape(x_sensor_temp)))

    # this is for rgb
    # x_rgb_temp = np.squeeze(x_rgb_temp, axis=0)
    # x_imu_temp = np.squeeze(x_imu_temp, axis=0)
    # y_temp = np.squeeze(y_temp, axis=0)

    # this is for raw data

    if x_sensor_temp.shape[0] == 1:
        x_sensor_temp = x_sensor_temp[0]

    if x_imu_temp.shape[0] == 1:
        x_imu_temp = x_imu_temp[0]

    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_sensor_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_sensor_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 20000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x_sensor.append(x_sensor_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])

        x_sensor.append(x_sensor_temp[(data_size - data_per_chunk):data_size, :, :, :])
        x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x_sensor.append(x_sensor_temp[0:data_size, :, :, :])
        x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x_sensor, x_imu, y

def load_hallucination_data(training_file):

    # Load data
    rgb_features = []
    hdf5_file = h5py.File(training_file, 'r')
    rgb_feat_temp = hdf5_file.get('rgb_feat')

    print('Data shape: ' + str(np.shape(rgb_feat_temp)))

    data_size = np.size(rgb_feat_temp, axis=0)

    # Determine whether the data should be divided into several chunks to fit in memory
    data_per_chunk = 5000 # maximum sequence length
    is_special_case = False
    if data_size > 10000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            rgb_features.append(rgb_feat_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :])

        # 2300
        rgb_features.append(rgb_feat_temp[(data_size - data_per_chunk):data_size, :, :])

    else:
        rgb_features.append(rgb_feat_temp[0:data_size, :, :])

    return n_chunk, rgb_features

def load_data_single_sensor(training_file, sensor):
    # Load data
    x, x_imu, y = [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_temp = hdf5_file.get(sensor+'_data')
    # x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')

    print('Data shape: ' + str(np.shape(x_temp)))

    # this is for rgb
    # x_rgb_temp = np.squeeze(x_rgb_temp, axis=0)
    # x_imu_temp = np.squeeze(x_imu_temp, axis=0)
    # y_temp = np.squeeze(y_temp, axis=0)

    # this is for raw data
    if x_temp.shape[0] == 1:
        x_temp = x_temp[0]
    # x_imu_temp = x_imu_temp[0]
    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 10000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x.append(x_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            # x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])
        # 2300
        x.append(x_temp[(data_size - data_per_chunk):data_size, :, :, :])
        # x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x.append(x_temp[0:data_size, :, :, :])
        # x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x, y

def load_data_multi_timestamp(training_file, sensor):

    # Load data
    x_sensor, x_imu, y = [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_time = hdf5_file.get('timestamp')
    x_sensor_temp = hdf5_file.get(sensor+'_data')
    x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')


    print('Data shape: ' + str(np.shape(x_sensor_temp)))

    # this is for raw data
    if x_sensor_temp.shape[0] == 1:
        x_sensor_temp = x_sensor_temp[0]

    if x_imu_temp.shape[0] == 1:
        x_imu_temp = x_imu_temp[0]
    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_sensor_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_sensor_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 20000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x_sensor.append(x_sensor_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])

        x_sensor.append(x_sensor_temp[(data_size - data_per_chunk):data_size, :, :, :])
        x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x_sensor.append(x_sensor_temp[0:data_size, :, :, :])
        x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x_time, x_sensor, x_imu, y