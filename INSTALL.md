# Install Instructions

## Prerequisites
- Install ROS [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) (if you install Melodic, you'll need to set up a python3 specific workspace, but it does work.)
- Install CUDA (https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)
- Install pytorch and torchvision (if you're on a jetson, use this: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048 if not, just `pip install torch`, `pip install torchvision`, you could either do this in your virtualenv later, or now so that it's install overall.)

## Create VirtualEnv

- `pip3 install virtualenv virtualenvwrapper`
- Set up your virtual environment wrapper. You'll need to add stuff to your bashrc (what's below is what I have, your filepaths might be different so make sure you check!)

```
export PYTHONPATH=${PYTHONPATH}:/usr/bin
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/.virtualenvs
source $HOME/.local/bin/virtualenvwrapper.sh
```
- `mkvirtualenv --system-site-packages trt_pose`

Now you have a virtual environment, if you're not already on it you can use `workon trt_pose`.

## Add Packages

- `pip install pyaml rospkg`
- `pip install --upgrade cython`
- `pip install tqdm cython pycocotools`
- `pip install matplotlib`
- `pip install empy`
- `pip install Pillow==6.2.2`
- `pip install traitlets`
- `pip install gpustat`
- `pip install nvidia-pyindex`
- `pip install nvidia-tensorrt==7.2.*`

You might need to install tensorrt (last two pip installs) differentaly on the Jetson.

## Install Modules

- Go to the trt_modules directory in the repository
```
cd torch2trt
python setup.py install
cd ../trt_pose
python setup.py install
```

## Build Package

Build the package in the typical ROS way: go to the base of your workspace, then catkin_make.

You might have issues with using particular versions of Pytorch, torchvision, OpenCV, etc, especially when on the Jetson.  On the Jetson we have found a certain set of versions to work, I'll put them here at some point (Jetpack: , OpenCV: , Pytorch: , torchvision: )

