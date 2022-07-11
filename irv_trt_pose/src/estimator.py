#! /home/michael/.virtualenvs/osg/bin/python3


import rospy
from rtrt_pkg_lib.pose_estimator import TRTPoseNode

def main():
    rospy.init_node('trt_estimator')
    trt_node = TRTPoseNode()
    trt_node.start()

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()

    rospy.spin()

    # rate = rospy.Rate(15)
    # while not rospy.is_shutdown():
    #     if trt_node.image is not None:
    #         trt_node.annotated_image = trt_node.execute()

    #     rate.sleep()
    

if __name__ == '__main__':
    main()
