# TI-SLAM

### [Paper](https://arxiv.org/abs/2104.07196) | [Youtube](https://www.youtube.com/watch?v=EZ1gpetEN8c) <br>

This is a simplified implementation of the following paper:

[Graph-based Thermal-Inertial SLAM with Probabilistic Neural Networks](https://arxiv.org/abs/2104.07196)  
Muhamad Risqi U. Saputra, Chris Xiaoxuan Lu, Pedro P. B. de Gusmao, Bing Wang, Andrew Markham, and Niki Trigoni.  

PS: At the moment, the implementation requires both python and matlab code. We will provide all python versions in the near future.

<p align="center"> <img src="https://github.com/risqiutama/ti-slam/blob/main/ti-slam_arch.png" width="100%"> </p>

## Dependencies (Python)
- Docker
- Python 3.6
- Tensorflow-gpu 1.14, Tensorflow-probability 0.7
- Keras 2.2.4, Keras-mdn-layer 0.2.2
- Others: scipy, matplotlib, opencv-python

## Dependencies (Matlab)
- Matlab R2019b or above
- Navigation Toolbox

## Getting Started

### Dataset
- Please download our dataset from [here](https://docs.google.com/forms/d/e/1FAIpQLSfdJBh5VUiIdvwQGhgInAlHzcmmOCnbvNqS_iHyo0KUDPBsaQ/viewform?usp=sf_link).

- After downloading, please put the dataset into your `<host dataset dir path>` in your host machine.
- Some files are zipped into a single zip file (e.g., loop.zip and all raw data in raw folder). For training, please unzip them all in the same directory. 

### Docker Installation

- Please install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
- Then, simply pull the correct version of tensorflow docker container as follows:
```
docker pull tensorflow/tensorflow:1.14.0-gpu-py3
```
- Create a docker image such as by using the following command. Please create your custom mapping between `<host dataset dir path>` to the virtual path inside the docker image.

```
nvidia-docker run --rm -it -v <host dataset dir path>:<virtual path> --name <docker image name> tensorflow/tensorflow:1.14.0-gpu-py3 bash
```
- Inside the docker image, install the remaining dependencies.
```
apt-get update
apt install python3-pil
apt install python3-pil.imagetk
apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install --upgrade scipy==1.1.0
pip3 install keras==2.2.4
pip3 install tensorflow-probability==0.7
python3 -m pip install keras-mdn-layer==0.2.2
python -mpip install matplotlib
pip3 install opencv-python
```

### Pre-trained Models
- Git clone this repository.
- Download the pre-trained models [here](https://docs.google.com/forms/d/e/1FAIpQLSfdJBh5VUiIdvwQGhgInAlHzcmmOCnbvNqS_iHyo0KUDPBsaQ/viewform?usp=sf_link).
- Create the following directories to load the model from your repository.
```
mkdir -p Python/odometry/models/neural_odometry
mkdir -p Python/odometry/models/neural_loop_closure
mkdir -p Python/loop/models/neural_embedding
```
- Move the models to the respective directories.

## Testing
Before running the test code, take a look at the config.yaml inside `Python/odometry` and `Python/loop`, and replace the path pointing to dataset with your `<host dataset dir path>`.

### Test Neural Odometry (Probabilistic DeepTIO)
Inside the docker image, run neural odometry with the pre-trained model by using the following commands:
```
cd Python/odometry/
python os_test_odom.py
```
The code will generate the output poses for all models and all trajectories in the folder `Python/odometry/results` (.txt files) and `Python/odometry/figs` (.png files showing the predicted trajectories).

### Test Neural Embedding
Inside the docker image, run neural embedding with the pre-trained model by using the following commands:
```
cd Python/loop/
python os_test_embed.py
```
The code outputs the embedding features in the folder `Python/loop/results` (.csv files).

To convert the embedding features into a list of loop pairs, execute the following matlab code:
```
generate_loop_from_embeddings.m # run it from Matlab
```
The matlab code will generate the loop pairs in `Python/odometry/results` (.csv files).

### Test Neural Loop Closure
Inside the docker image, run neural loop closure with the pre-trained model by using the following commands:
```
cd Python/odometry/
python os_test_loop.py
```
The code will generate the output poses for all models and all trajectories in the folder `Python/odometry/results` (.txt files).

### Test the SLAM Back End
To optimize the odometry by using our robust SLAM back end, please run the following Matlab code:
```
robust_pose_graph_optimization.m # run it from Matlab
```
The code generates the figure showing the optimized trajectory in `figures/optimized_odometry` (.pdf files).

## Training
Before training, take a look again at the config.yaml file and make sure that you have replaced the path with your `<host dataset dir path>`.
  
For training, you need to individually train the neural models (neural odometry, neural embedding, and neural loop closure).
Inside the docker image:
- run the following code to train the neural odometry and neural loop closure:
```
cd Python/odometry/
python train_deeptio_prob.py
python train_loop_pose.py
```
- run the following code to train the neural embedding:
```
cd Python/loop/
python train_neural_embedding.py
```
