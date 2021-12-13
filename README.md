# ros2_trt_pose

This is a ROS package for trt_pose by the NVIDIA IOT team. This package is based off of [the NVIDIA package](https://github.com/NVIDIA-AI-IOT/ros2_trt_pose).

This package exists because we don't have a good ROS node for our robot, which is running old ROS versions still. Additionally, there are some refinements, including running TRT_pose only on a detected human in the image, and running a Gaussian filter on detections to keep the output consistent for underwater data.