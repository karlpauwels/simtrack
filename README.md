SimTrack
========

[SimTrack](http://www.karlpauwels.com/simtrack), a simulation-based framework for tracking, is a [ROS](http://www.ros.org/)-package for detecting and tracking the pose of multiple (textured) rigid objects in real-time. SimTrack is released under the [BSD-license](http://opensource.org/licenses/BSD-3-Clause). Please cite the following paper if you use SimTrack in your research:

*Pauwels, Karl and Kragic, Danica (2015) [SimTrack: A Simulation-based Framework for Scalable Real-time Object Pose Detection and Tracking](http://www.karlpauwels.com/downloads/iros_2015/Pauwels_IROS_2015.pdf). IEEE/RSJ International Conference on Intelligent Robots and Systems, Hamburg, Germany, 2015.*

For more details see the following paper: 

*Pauwels, Karl; Rubio, Leonardo; Ros, Eduardo (2015) [Real-time Pose Detection and Tracking of Hundreds of Objects](http://www.karlpauwels.com/downloads/tcsvt_2015/Pauwels_IEEE_TCSVT_2015.pdf). IEEE Transactions on Circuits and Systems for Video Technology, in press.*


System Requirements
-------------------

* Ubuntu 12.04 or 14.04
* ROS Hydro or Indigo
* Installed and working [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) driver and toolkit
* CUDA-capable Graphics Processing Unit, Fermi class or above
* Monocular camera or RGB-D sensor (Asus Xtion, Primesense, Kinect v1 or Kinect v2)

Installation
------------

Install wstool:
```
sudo apt-get install python-wstool
```

Create your workspace:
```
mkdir -p ~/my-ws/src
```

Copy the contents of [simtrack.rosinstall](simtrack.rosinstall) into a file ~/my-ws/src/.rosinstall

Fetch the code:
```
cd ~/my-ws/src
wstool update
```

Install the dependencies:
```
cd ~/my-ws
sudo rosdep init # only if never run before
rosdep install --from-paths src --ignore-src
```

Build:
```
cd ~/my-ws
catkin_make
```

Test
----

Check `nvidia-settings` and verify that *Sync to VBlank* is disabled under *OpenGL Settings*.

Initialize the environment:

```
cd ~/my-ws
source devel/setup.bash
```

Build the SIFT-model of the THREE provided demo objects. Note that an absolute path is required:
```
rosrun interface cmd_line_generate_sift_model `pwd`/src/simtrack/data/object_models/ros_fuerte/ros_fuerte.obj
rosrun interface cmd_line_generate_sift_model `pwd`/src/simtrack/data/object_models/ros_groovy/ros_groovy.obj
rosrun interface cmd_line_generate_sift_model `pwd`/src/simtrack/data/object_models/ros_hydro/ros_hydro.obj
```

This process should display rotated versions of the objects with SIFT keypoints highlighted. Three new files will be created in the respective model folders: *ros_fuerte_SIFT.h5* (~4.2MB), *ros_groovy_SIFT.h5* (~7.0MB) and *ros_hydro_SIFT.h5* (~2.9MB).

Print [models_to_print.pdf](data/object_models/models_to_print.pdf) on three A4-pages. Check that the dimensions are correct using the printed ruler.

Adjust the GPU configuration in [parameters.yaml](simtrack_nodes/config/parameters.yaml) to your system:
```
simtrack/tracker/device_id: 0
simtrack/detector/device_id: 1
```
For example, set both *device_id's* to 0 for a single-GPU system.

The default configuration uses *openni_2-launch* which supports Asus Xtion and Primesense devices. To use a Kinect, point main.launch to the camera_kinect.launch file.

Run SimTrack:
```
roslaunch simtrack_nodes main.launch
```

The tracker output is available on the */simtrack/image* topic. It can be adjusted through *dynamic_reconfigure* to display either the tracker state or the optical flow.

If all goes well, the printed pages should be detected and tracked! See the tutorial for help on modeling and tracking
your own objects.

Tutorial
--------

[www.karlpauwels.com/simtrack](http://www.karlpauwels.com/simtrack)
