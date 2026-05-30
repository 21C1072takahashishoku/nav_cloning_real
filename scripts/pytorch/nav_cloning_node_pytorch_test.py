#!/usr/bin/env python3
from __future__ import print_function
from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, SetBoolResponse
import copy

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.nav_pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/switch_segmentation', SetBool, self.callback_dl_switch)
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.navigation = True
        self.is_started = False
        self.switch = False
        self.flg = 0

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel

    def callback_dl_switch(self, data):
        resp = SetBoolResponse()
        self.switch = data.data
        if self.switch:
            self.flg += 1
        resp.message = "switch: " + str(self.switch)
        resp.success = True
        return resp

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return
        
        img = resize(self.cv_image, (48, 64), mode='constant')
        
        if self.switch and self.flg == 1:
            self.navigation = False
            self.dl.load("/home/yuzuki/model_gpu.pt")
         
        if self.switch and self.flg == 2:
            self.navigation = False
            self.dl.load("/home/yuzuki/Downloads/model_gpu.pt")

        if self.switch == False:
            self.navigation = True
           
        if self.navigation:
            target_action = self.action
            self.vel = target_action
            self.nav_pub.publish(self.vel)
            print("navigation")

        else:
            target_action = self.dl.act(img)
            print("dl_output")
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()