#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_EBV_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "change_dataset_balance")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/usb_cam/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/usb_cam/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_EBV_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_EBV_'+str(self.mode)+'/'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        self.baseline = np.zeros((480, 640), np.uint8)
        self.baseline_left = np.zeros((480, 640), np.uint8)
        self.baseline_right = np.zeros((480, 640), np.uint8)
        self.img_flag = True
        self.img_left_flag = True
        self.img_right_flag = True

        # with open(self.path + self.start_time + '/' +  'training_all.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)


    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def callback_model_save(self, data):
        model_res = SetBoolResponse()
        self.dl.save(self.save_path)
        model_res.message ="model_save"
        model_res.success = True
        return model_res

    def generate_events(self, img, baseline):
        diff = img.astype(np.int32) - baseline.astype(np.int32)
        pos_event = (diff > 1)
        neg_event = (diff < -1)

        event_img = np.full((img.shape[0], img.shape[1]), 128, dtype=np.uint8)  
        event_img[pos_event] = 255  
        event_img[neg_event] = 0 

        baseline[pos_event] = img[pos_event]
        baseline[neg_event] = img[neg_event]

        return event_img, baseline

    def generate_events_left(self, img_left, baseline_left):
        diff_left = img_left.astype(np.int32) - baseline_left.astype(np.int32)
        pos_event_left = (diff_left > 1)
        neg_event_left = (diff_left < -1)

        event_img_left = np.full((img_left.shape[0], img_left.shape[1]), 128, dtype=np.uint8)  
        event_img_left[pos_event_left] = 255  
        event_img_left[neg_event_left] = 0 

        baseline_left[pos_event_left] = img_left[pos_event_left]
        baseline_left[neg_event_left] = img_left[neg_event_left]

        return event_img_left, baseline_left

    def generate_events_right(self, img_right, baseline_right):
        diff_right = img_right.astype(np.int32) - baseline_right.astype(np.int32)
        pos_event_right = (diff_right > 1)
        neg_event_right = (diff_right < -1)

        event_img_right = np.full((img_right.shape[0], img_right.shape[1]), 128, dtype=np.uint8)  
        event_img_right[pos_event_right] = 255  
        event_img_right[neg_event_right] = 0 

        baseline_right[pos_event_right] = img_right[pos_event_right]
        baseline_right[neg_event_right] = img_right[neg_event_right]

        return event_img_right, baseline_right

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return

        img = cv2.cvtColor((self.cv_image * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        if self.img_flag == True:
            img, baseline = self.generate_events(img, self.baseline)
            self.img_flag == False
        else:
            img, baseline = self.generate_events(img, baseline)

        img = resize(img, (48, 64), mode='constant')
        
        # img = gray.astype('float64')
        # r, g, b = cv2.split(img)
        # img = np.asanyarray([r,g,b])

        img_left = cv2.cvtColor((self.cv_left_image * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
        img_left = cv2.bitwise_not(img_left)
        if self.img_left_flag == True:
            img_left, baseline_left = self.generate_events_left(img_left, self.baseline_left)
            self.img_left_flag == False
        else:
            img_left, baseline_left = self.generate_events_left(img_left, baseline_left)

        img_left = resize(img_left, (48, 64), mode='constant')
        # img_left = gray_left.astype('float64')
        #r, g, b = cv2.split(img_left)
        #img_left = np.asanyarray([r,g,b])

        img_right = cv2.cvtColor((self.cv_right_image * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
        img_right = cv2.bitwise_not(img_right)
        if self.img_right_flag == True:
            img_right, baseline_right = self.generate_events_right(img_right, self.baseline_right)
            self.img_right_flag == False
        else:
            img_right, baseline_right = self.generate_events_right(img_right, baseline_right)

        img_right = resize(img_right, (48, 64), mode='constant')
        # img_right = gray_right.astype('float64')
        #r, g, b = cv2.split(img_right)
        #img_right = np.asanyarray([r,g,b])
        ros_time = str(rospy.Time.now())

        if self.episode == 4000:
            self.learning = False
            self.dl.save(self.save_path)
            #self.dl.load(self.load_path)

        if self.episode == 8000:
            os.system('killall roslaunch')
            sys.exit()

        if self.learning:
            target_action = self.action
            distance = self.min_distance

            if self.mode == "manual":
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "zigzag":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0

            elif self.mode == "use_dl_output":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action


             
            elif self.mode == "change_dataset_balance":
                if distance < 0.05:
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                elif 0.05 <= distance < 0.1:
                    self.dl.make_dataset(img , target_action)
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                    lines = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(lines)
                   
                else:
                    self.dl.make_dataset(img , target_action)
                    self.dl.make_dataset(img , target_action)
                    self.dl.make_dataset(img , target_action)
                    self.dl.make_dataset(img , target_action)
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                    lines = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(lines)
                    with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(lines)


                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action


            # elif self.mode == "change_dataset_balancesss":
            #     if distance < 0.05:
            #         action, loss = self.dl.act_and_trains(img , target_action)
            #         if abs(target_action) < 0.1:
            #             action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
            #             action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
            #     elif 0.05 <= distance < 0.1:
            #         self.dl.make_dataset(img , target_action)
            #         action, loss = self.dl.act_and_trains(img , target_action)
            #         if abs(target_action) < 0.1:
            #             self.dl.make_dataset(img_left , target_action - 0.2)
            #             action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
            #             self.dl.make_dataset(img_right , target_action + 0.2)
            #             action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
            #         line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            #         with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
            #             writer = csv.writer(f, lineterminator='\n')
            #             writer.writerow(line)
            #     else:
            #         self.dl.make_dataset(img , target_action)
            #         self.dl.make_dataset(img , target_action)
            #         action, loss = self.dl.act_and_trains(img , target_action)
            #         if abs(target_action) < 0.1:
            #             self.dl.make_dataset(img_left , target_action - 0.2)
            #             self.dl.make_dataset(img_left , target_action - 0.2)
            #             action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
            #             self.dl.make_dataset(img_right , target_action + 0.2)
            #             self.dl.make_dataset(img_right , target_action + 0.2)
            #             action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
            #         line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            #         with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
            #             writer = csv.writer(f, lineterminator='\n')
            #             writer.writerow(line)
            #         with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
            #             writer = csv.writer(f, lineterminator='\n')
            #             writer.writerow(line)


            #     angle_error = abs(action - target_action)
            #     if distance > 0.1:
            #         self.select_dl = False
            #     elif distance < 0.05:
            #         self.select_dl = True
            #     if self.select_dl and self.episode >= 0:
            #         target_action = action

            elif self.mode == "follow_line":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "selected_training":
                action = self.dl.act(img )
                angle_error = abs(action - target_action)
                loss = 0
                if angle_error > 0.05:
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                
                if distance > 0.15 or angle_error > 0.3:
                    self.select_dl = False
                # if distance > 0.1:
                #     self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            # end mode

            self.episode += 1
            print(str(self.episode) + ", training, loss: " + str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance))
            # print(str(self.episode)  + ", distance: " + str(distance))
            # line = [str(self.episode), "training", str(loss), str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            target_action = self.dl.act(img)
            distance = self.min_distance
            print(str(self.episode) + ", test, angular:" + str(target_action) + ", distance: " + str(distance))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            # line = [str(self.episode), "test", "0", str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            line = [str(self.episode), "test", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()