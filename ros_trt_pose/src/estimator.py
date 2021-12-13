#! /home/michael/.virtualenvs/trt_pose/bin/python3


import rospy
from rtrt_pkg_lib.pose_estimator import TRTPoseNode

def main():
    rospy.init_node('trt_estimator')
    trt_node = TRTPoseNode()
    trt_node.start()

    rospy.spin()

if __name__ == '__main__':
    main()