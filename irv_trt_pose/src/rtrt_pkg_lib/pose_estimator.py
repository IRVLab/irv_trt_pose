#! /home/michael/.virtualenvs/trt_pose/bin/python3

# ---------------------------------------------------------------------------------------
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------------------

# ROS related
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image as ImageMsg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from pose_msgs.msg import BodypartDetection, PersonDetection  # For pose_msgs
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox

from cv_bridge import CvBridge, CvBridgeError

# TRT_pose related
import cv2
import numpy as np
import math
import os
from rtrt_pkg_lib.estimation_utils import preprocess, load_params, load_model, draw_objects


class TRTPoseNode(object):
    def __init__(self):
        self.hp_json_file = None
        self.model_weights = None
        self.width = 224
        self.height = 224
        self.input_width = None
        self.input_height = None
        self.i = 0
        self.image = None
        self.model_trt = None
        self.annotated_image = None
        self.counts = None
        self.peaks = None
        self.objects = None
        self.topology = None
        self.xy_circles = []
        self.p = None

        # ROS parameters
        self.base_dir = rospy.get_param('base_dir', '/home/michael/gesture_ws/src/irv_trt_pose/irv_trt_pose/trt_config_files/')
        # Based Dir should contain: model_file resnet/densenet, human_pose json file

        self.model_name = rospy.get_param('model', 'resnet18') # default to Resnet18
        self.point_range = rospy.get_param('point_range', 10) # default range is 0 to 10
        self.show_image_param = rospy.get_param('show_image', False)# Show image in cv2.imshow


        # Topic parameters
        self.in_image_topic = rospy.get_param('in_image_topic', 'image_raw')
        self.out_image_topic = rospy.get_param('out_image_topic', 'pose_image')
        self.joints_topic = rospy.get_param('joints_topic', 'body_joints')
        self.skeleton_topic = rospy.get_param('skeleton_topic', 'body_skeleton')
        self.pose_topic = rospy.get_param('pose_topic', 'pose_msgs')

        
        # Pre-crop init
        self.pre_crop = rospy.get_param('pre_crop', False)
        self.pre_crop_topic = rospy.get_param('pre_crop_topic', 'bbox')

        if self.pre_crop:
            self.crop_box_sub = rospy.Subscriber(self.pre_crop_topic, BoundingBoxes, self.bbox_callback, buff_size=10)
            self.last_bbox = None
            self.last_bbox_time = None
            self.pre_crop_time = rospy.get_param('pre_crop_time', 0.1) # How long a bbox can be considered valid
            self.pre_crop_prob = rospy.get_param('pre_crop_prob', 0.8)
            self.pre_crop_tolerance = rospy.get_param('pre_crop_tolerance', 0.15) # How much to expand bbox size.

        # Filter parameters

        # ROS related init
        # Image subscriber from cam2image
        self.subscriber_ = rospy.Subscriber(self.in_image_topic, ImageMsg, self.read_cam_callback, buff_size=10)

        # CVBridge initilization
        self.bridge_object = CvBridge()

        # Standard publishers
        self.image_pub = rospy.Publisher(self.out_image_topic, ImageMsg, queue_size=10)
        self.body_joints_pub = rospy.Publisher(self.joints_topic, Marker, queue_size=1000)
        self.body_skeleton_pub = rospy.Publisher(self.skeleton_topic, Marker, queue_size=10)
        self.publish_pose = rospy.Publisher(self.pose_topic, PersonDetection, queue_size=100)

    def start(self):
        # Convert to TRT and Load Params
        json_file = os.path.join(self.base_dir, 'human_pose.json')
        rospy.loginfo("Loading model weights\n")
        self.num_parts, self.num_links, self.model_weights, self.parse_objects, self.topology = load_params(base_dir=self.base_dir,
                                                                                             human_pose_json=json_file,
                                                                                             model_name=self.model_name)
        self.model_trt, self.height, self.width = load_model(base_dir=self.base_dir, model_name=self.model_name, num_parts=self.num_parts, num_links=self.num_links,
                                    model_weights=self.model_weights)
        rospy.loginfo("Model weights loaded...\n Waiting for images...\n")

    def execute(self):
        data = preprocess(image=self.image, width=self.width, height=self.height)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        self.counts, self.objects, self.peaks = self.parse_objects(cmap,
                                                                   paf)  # , cmap_threshold=0.15, link_threshold=0.15)
        annotated_image = draw_objects(image=self.image, object_counts=self.counts, objects=self.objects, normalized_peaks=self.peaks, topology=self.topology)
        self.parse_k()

        return annotated_image

    # Subscribe and Publish to image topic
    def read_cam_callback(self, msg):
        if self.model_trt == None:
            rospy.logwarn("TRT model not yet initialized, please wait...")
            return

        if self.input_width == None:
            self.input_width = msg.width
            self.input_height = msg.height

        self.image = self.bridge_object.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.pre_crop:
            self.image = self.crop_to_bbox(self.image, self.last_bbox, self.last_bbox_time)

        self.annotated_image = self.execute()

        image_msg = self.image_np_to_image_msg(self.annotated_image)
        self.image_pub.publish(image_msg)

        if self.show_image_param:
            cv2.imshow('frame', self.annotated_image)
            cv2.waitKey(1)

    def bbox_callback(self, msg):
        self.last_bbox = msg.bounding_boxes[0]
        self.last_bbox_time = rospy.Time.now().to_sec()

    def crop_to_bbox(self, image, bbox, time):
        now = rospy.Time.now().to_sec()

        if ((now - time) < self.pre_crop_time) and (bbox.probability > self.pre_crop_prob):
            # If the bbox is recent enough and of high enough probability, crop to it.
            x_grow = int(self.input_width * self.pre_crop_tolerance)
            y_grow = int(self.input_height * self.pre_crop_tolerance)

            # Expand min bbox coordinates
            min_x_g = bbox.xmin - x_grow
            xmin = 0 if min_x_g < 0 else min_x_g
            min_y_g = bbox.ymin - y_grow
            ymin = 0 if min_y_g < 0 else min_y_g

            # Expand max bbox coordinates
            max_x_g = bbox.xmax + x_grow
            xmax = self.input_width if max_x_g > self.input_width else max_x_g
            max_y_g = bbox.ymax + y_grow
            ymax = self.input_height if max_x_g > self.input_height else max_x_g

            cropped = image[xmin:xmax, ymin:ymax]
            return cropped

        else:
            return image

    # Borrowed from OpenPose-ROS repo
    def image_np_to_image_msg(self, image_np):
        image_msg = ImageMsg()
        image_msg.height = image_np.shape[0]
        image_msg.width = image_np.shape[1]
        image_msg.encoding = 'bgr8'
        image_msg.data = image_np.tostring()
        image_msg.step = len(image_msg.data) // image_msg.height
        image_msg.header.frame_id = 'map'
        return image_msg

    def init_body_part_msg(self):
        bodypart = BodypartDetection()
        bodypart.x = float('NaN')
        bodypart.y = float('NaN')
        bodypart.confidence = float('NaN')
        return bodypart

    def write_body_part_msg(self, pixel_location):
        body_part_pixel_loc = BodypartDetection()
        body_part_pixel_loc.y = float(pixel_location[0] * self.height)
        body_part_pixel_loc.x = float(pixel_location[1] * self.width)
        return body_part_pixel_loc

    def init_markers_spheres(self):
        marker_joints = Marker()
        marker_joints.header.frame_id = '/map'
        marker_joints.id = 1
        marker_joints.ns = "joints"
        marker_joints.type = marker_joints.SPHERE_LIST
        marker_joints.action = marker_joints.ADD
        marker_joints.scale.x = 0.7
        marker_joints.scale.y = 0.7
        marker_joints.scale.z = 0.7
        marker_joints.color.a = 1.0
        marker_joints.color.r = 1.0
        marker_joints.color.g = 0.0
        marker_joints.color.b = 0.0
        marker_joints.lifetime = rospy.Duration(secs=3, nsecs=5e2)
        return marker_joints


    def init_markers_lines(self):
        marker_line = Marker()
        marker_line.header.frame_id = '/map'
        marker_line.id = 1
        marker_line.ns = "joint_line"
        marker_line.header.stamp = rospy.Time.now()
        marker_line.type = marker_line.LINE_LIST
        marker_line.action = marker_line.ADD
        marker_line.scale.x = 0.1
        marker_line.scale.y = 0.1
        marker_line.scale.z = 0.1
        marker_line.color.a = 1.0
        marker_line.color.r = 0.0
        marker_line.color.g = 1.0
        marker_line.color.b = 0.0
        marker_line.lifetime = rospy.Duration(secs=3, nsecs=5e2)
        return marker_line

    def init_all_body_msgs(self, _msg, count):
        _msg.header = Header()
        _msg.person_id = count
        _msg.nose = self.init_body_part_msg()
        _msg.neck = self.init_body_part_msg()
        _msg.right_shoulder = self.init_body_part_msg()
        _msg.right_elbow = self.init_body_part_msg()
        _msg.right_wrist = self.init_body_part_msg()
        _msg.left_shoulder = self.init_body_part_msg()
        _msg.left_elbow = self.init_body_part_msg()
        _msg.left_wrist = self.init_body_part_msg()
        _msg.right_hip = self.init_body_part_msg()
        _msg.right_knee = self.init_body_part_msg()
        _msg.right_ankle = self.init_body_part_msg()
        _msg.left_hip = self.init_body_part_msg()
        _msg.left_knee = self.init_body_part_msg()
        _msg.left_ankle = self.init_body_part_msg()
        _msg.right_eye = self.init_body_part_msg()
        _msg.left_eye = self.init_body_part_msg()
        _msg.right_ear = self.init_body_part_msg()
        _msg.left_ear = self.init_body_part_msg()
        return _msg

    def add_point_to_marker(self, body_part_msg):
        p = Point()
        p.x = float((body_part_msg.x / self.width) * self.point_range)
        p.y = float((body_part_msg.y / self.height) * self.point_range)
        p.z = 0.0
        return p

    def valid_marker_point(self, body_part_msg):
        if math.isnan(body_part_msg.x) or math.isnan(body_part_msg.y):
            return False
        return True

    def parse_k(self):
        image_idx = 0
        try:
            count = int(self.counts[image_idx])
            primary_msg = PersonDetection()  # BodypartDetection()
            primary_msg.num_people_detected = count
            for i in range(count):
                primary_msg.person_id = i
                primary_msg = self.init_all_body_msgs(_msg=primary_msg, count=i)
                marker_joints = self.init_markers_spheres()
                marker_skeleton = self.init_markers_lines()
                for k in range(18):
                    _idx = self.objects[image_idx, i, k]
                    if _idx >= 0:
                        _location = self.peaks[image_idx, k, _idx, :]
                        if k == 0:
                            primary_msg.nose = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.nose))
                            rospy.loginfo(
                                "Body Part Detected: nose at X:{}, Y:{}".format(primary_msg.nose.x, primary_msg.nose.y))
                        if k == 1:
                            primary_msg.left_eye = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_eye))
                            rospy.loginfo(
                                "Body Part Detected: left_eye at X:{}, Y:{}".format(primary_msg.left_eye.x,
                                                                                    primary_msg.left_eye.y))
                            if self.valid_marker_point(primary_msg.nose):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.nose))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_eye))

                        if k == 2:
                            primary_msg.right_eye = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_eye))
                            rospy.loginfo(
                                "Body Part Detected: right_eye at X:{}, Y:{}".format(primary_msg.right_eye.x,
                                                                                     primary_msg.right_eye.y))
                            if self.valid_marker_point(primary_msg.nose):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.nose))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_eye))
                            if self.valid_marker_point(primary_msg.left_eye):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_eye))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_eye))

                        if k == 3:
                            primary_msg.left_ear = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_ear))
                            rospy.loginfo(
                                "Body Part Detected: left_ear at X:{}, Y:{}".format(primary_msg.left_ear.x,
                                                                                    primary_msg.left_ear.y))
                            if self.valid_marker_point(primary_msg.left_eye):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_eye))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_ear))

                        if k == 4:
                            primary_msg.right_ear = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_ear))
                            rospy.loginfo(
                                "Body Part Detected: right_ear at X:{}, Y:{}".format(primary_msg.right_ear.x,
                                                                                     primary_msg.right_ear.y))

                            if self.valid_marker_point(primary_msg.right_eye):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_eye))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_ear))

                        if k == 5:
                            primary_msg.left_shoulder = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_shoulder))
                            rospy.loginfo(
                                "Body Part Detected: left_shoulder at X:{}, Y:{}".format(primary_msg.left_shoulder.x,
                                                                                         primary_msg.left_shoulder.y))
                            if self.valid_marker_point(primary_msg.left_ear):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_ear))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_shoulder))

                        if k == 6:
                            primary_msg.right_shoulder = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_shoulder))
                            rospy.loginfo(
                                "Body Part Detected: right_shoulder at X:{}, Y:{}".format(primary_msg.right_shoulder.x,
                                                                                          primary_msg.right_shoulder.y))

                            if self.valid_marker_point(primary_msg.right_ear):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_ear))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_shoulder))

                        if k == 7:
                            primary_msg.left_elbow = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_elbow))
                            rospy.loginfo(
                                "Body Part Detected: left_elbow at X:{}, Y:{}".format(primary_msg.left_elbow.x,
                                                                                      primary_msg.left_elbow.y))

                            if self.valid_marker_point(primary_msg.left_shoulder):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_elbow))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_shoulder))

                        if k == 8:
                            primary_msg.right_elbow = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_elbow))
                            rospy.loginfo(
                                "Body Part Detected: right_elbow at X:{}, Y:{}".format(primary_msg.right_elbow.x,
                                                                                       primary_msg.right_elbow.y))

                            if self.valid_marker_point(primary_msg.right_shoulder):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_elbow))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_shoulder))

                        if k == 9:
                            primary_msg.left_wrist = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_wrist))
                            rospy.loginfo(
                                "Body Part Detected: left_wrist at X:{}, Y:{}".format(primary_msg.left_wrist.x,
                                                                                      primary_msg.left_wrist.y))

                            if self.valid_marker_point(primary_msg.left_elbow):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_elbow))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_wrist))

                        if k == 10:
                            primary_msg.right_wrist = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_wrist))
                            rospy.loginfo(
                                "Body Part Detected: right_wrist at X:{}, Y:{}".format(primary_msg.right_wrist.x,
                                                                                       primary_msg.right_wrist.y))

                            if self.valid_marker_point(primary_msg.right_elbow):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_elbow))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_wrist))

                        if k == 11:
                            primary_msg.left_hip = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_hip))
                            rospy.loginfo(
                                "Body Part Detected: left_hip at X:{}, Y:{}".format(primary_msg.left_hip.x,
                                                                                    primary_msg.left_hip.y))

                        if k == 12:
                            primary_msg.right_hip = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_hip))
                            rospy.loginfo(
                                "Body Part Detected: right_hip at X:{}, Y:{}".format(primary_msg.right_hip.x,
                                                                                     primary_msg.right_hip.y))

                            if self.valid_marker_point(primary_msg.left_hip):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_hip))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_hip))

                        if k == 13:
                            primary_msg.left_knee = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_knee))
                            rospy.loginfo(
                                "Body Part Detected: left_knee at X:{}, Y:{}".format(primary_msg.left_knee.x,
                                                                                     primary_msg.left_knee.y))

                            if self.valid_marker_point(primary_msg.left_hip):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_hip))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_knee))

                        if k == 14:
                            primary_msg.right_knee = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_knee))
                            rospy.loginfo(
                                "Body Part Detected: right_knee at X:{}, Y:{}".format(primary_msg.right_knee.x,
                                                                                      primary_msg.right_knee.y))

                            if self.valid_marker_point(primary_msg.right_hip):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_hip))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_knee))

                        if k == 15:
                            primary_msg.left_ankle = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.left_ankle))
                            rospy.loginfo(
                                "Body Part Detected: left_ankle at X:{}, Y:{}".format(primary_msg.left_ankle.x,
                                                                                      primary_msg.left_ankle.y))

                            if self.valid_marker_point(primary_msg.left_knee):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_ankle))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_knee))
                        if k == 16:
                            primary_msg.right_ankle = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.right_ankle))
                            rospy.loginfo(
                                "Body Part Detected: right_ankle at X:{}, Y:{}".format(primary_msg.right_ankle.x,
                                                                                       primary_msg.right_ankle.y))

                            if self.valid_marker_point(primary_msg.right_knee):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_ankle))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_knee))

                        if k == 17:
                            primary_msg.neck = self.write_body_part_msg(_location)
                            marker_joints.points.append(self.add_point_to_marker(primary_msg.neck))
                            rospy.loginfo(
                                "Body Part Detected: neck at X:{}, Y:{}".format(primary_msg.neck.x, primary_msg.neck.y))

                            if self.valid_marker_point(primary_msg.nose):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.nose))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.neck))
                            if self.valid_marker_point(primary_msg.right_shoulder):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_shoulder))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.neck))
                            if self.valid_marker_point(primary_msg.right_hip):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.right_hip))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.neck))
                            if self.valid_marker_point(primary_msg.left_shoulder):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_shoulder))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.neck))
                            if self.valid_marker_point(primary_msg.left_hip):
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.left_hip))
                                marker_skeleton.points.append(self.add_point_to_marker(primary_msg.neck))

                        self.publish_pose.publish(primary_msg)
                        self.body_skeleton_pub.publish(marker_skeleton)
                        self.body_joints_pub.publish(marker_joints)

                rospy.loginfo("Published Message for Person ID:{}".format(primary_msg.person_id))
        except Exception as err:
            rospy.logwarn("Some error in parse_k(): {},  {}".format(type(err), str(err)))
            pass
            