#!/usr/bin/env python3
from __future__ import print_function
from numpy import dtype
import numpy as np
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
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
from sensor_msgs.msg import Illuminance
from std_srvs.srv import TriggerResponse


class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "change_dataset_balance")

        # ===== Offline dataset logging (for offline training) =====
        # collect_teleop+lux.py と learning_surprise.py が期待する
        #   data/<DATASET_ID>/dataset/img/{episode}_{view}.npy
        #   data/<DATASET_ID>/dataset/vel/data.csv
        # 形式で保存する。
        #
        # 今回の追加：
        #   data/<DATASET_ID>/dataset/raw_img/center/{episode}_center.png
        #   data/<DATASET_ID>/dataset/raw_img/left/{episode}_left.png
        #   data/<DATASET_ID>/dataset/raw_img/right/{episode}_right.png
        # も保存する。
        # これはリサイズ前の 480x640 BGR画像であり、後処理セグメンテーション用。
        self.save_offline_dataset = rospy.get_param("/nav_cloning_node/save_offline_dataset", True)
        self.save_raw_images = rospy.get_param("/nav_cloning_node/save_raw_images", True)

        self.dataset_id = rospy.get_param("/nav_cloning_node/dataset_id", "")
        self.dataset_stride = int(rospy.get_param("/nav_cloning_node/dataset_stride", 1))
        self.dataset_action_offset = float(rospy.get_param("/nav_cloning_node/dataset_action_offset", 0.2))

        self.offline_episode = 0
        self.action_num = 1
        self.bridge = CvBridge()

        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)

        self.relay_nav_vel = rospy.get_param("/nav_cloning_node/relay_nav_vel", False)
        self.start_on_nav_vel = rospy.get_param("/nav_cloning_node/start_on_nav_vel", True)
        self.start_speed_threshold = float(
            rospy.get_param("/nav_cloning_node/start_speed_threshold", 0.01)
        )

        self.cmd_vel_mode = rospy.get_param("/nav_cloning_node/cmd_vel_mode", "mixed")
        self.cmd_vel_linear_source = rospy.get_param("/nav_cloning_node/cmd_vel_linear_source", "fixed")
        self.cmd_vel_linear_fixed = float(
            rospy.get_param("/nav_cloning_node/cmd_vel_linear_fixed", 0.2)
        )

        self.train_steps = int(rospy.get_param("/nav_cloning_node/train_steps", 10000))
        self.test_steps = int(rospy.get_param("/nav_cloning_node/test_steps", 3000))
        self.auto_test = rospy.get_param("/nav_cloning_node/auto_test", True)
        self.auto_stop_test = rospy.get_param("/nav_cloning_node/auto_stop_test", False)
        self.auto_save_model = rospy.get_param("/nav_cloning_node/auto_save_model", True)
        self.auto_load_model = rospy.get_param("/nav_cloning_node/auto_load_model", True)
        self.auto_save_on_shutdown = rospy.get_param("/nav_cloning_node/auto_save_on_shutdown", True)
        self.model_load_path = rospy.get_param("/nav_cloning_node/model_load_path", "")

        self.show_debug = rospy.get_param(
            "/nav_cloning_node/show_debug",
            bool(os.environ.get("DISPLAY"))
        )

        self.has_motion = False

        if self.relay_nav_vel:
            rospy.loginfo("[nav_cloning_node] relay_nav_vel enabled: /nav_vel -> /icart_mini/cmd_vel")

        # ===== Subscribers =====
        self.image_sub = rospy.Subscriber(
            "/camera_center/usb_cam/image_raw",
            Image,
            self.callback
        )
        self.image_left_sub = rospy.Subscriber(
            "/camera_left/usb_cam/image_raw",
            Image,
            self.callback_left_camera
        )
        self.image_right_sub = rospy.Subscriber(
            "/camera_right/usb_cam/image_raw",
            Image,
            self.callback_right_camera
        )

        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)

        self.pose_sub = rospy.Subscriber("/mcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.callback_path)

        # ===== State =====
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()

        # raw camera images: BGR uint8, 480x640x3
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_left_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_right_image = np.zeros((480, 640, 3), np.uint8)

        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")

        # Always switch to test after train_steps, regardless of auto_test.
        self.auto_test_step = self.train_steps
        self.test_end_step = (
            self.train_steps + self.test_steps
            if self.test_steps > 0
            else -1
        )

        self.auto_test_done = False
        self.stop_requested = False
        self.saved_model_path = ""

        # dataset_id が空なら start_time を使う
        if self.dataset_id is None or str(self.dataset_id).strip() == "":
            self.dataset_id = self.start_time

        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_' + str(self.mode) + '/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_' + str(self.mode) + '/'

        self.previous_reset_time = 0
        self.pos_x = None
        self.pos_y = None
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()

        os.makedirs(self.path + self.start_time, exist_ok=True)

        # ===== Offline dataset dirs =====
        self.dataset_base = (
            roslib.packages.get_pkg_dir('nav_cloning')
            + '/data/'
            + str(self.dataset_id)
            + '/dataset/'
        )

        # resized learning images
        self.img_store_path = os.path.join(self.dataset_base, 'img')

        # csv
        self.vel_store_path = os.path.join(self.dataset_base, 'vel')
        self.vel_csv_file = os.path.join(self.vel_store_path, 'data.csv')

        # raw images for offline segmentation
        self.raw_img_store_path = os.path.join(self.dataset_base, 'raw_img')
        self.raw_center_store_path = os.path.join(self.raw_img_store_path, 'center')
        self.raw_left_store_path = os.path.join(self.raw_img_store_path, 'left')
        self.raw_right_store_path = os.path.join(self.raw_img_store_path, 'right')

        os.makedirs(self.img_store_path, exist_ok=True)
        os.makedirs(self.vel_store_path, exist_ok=True)

        if self.save_raw_images:
            os.makedirs(self.raw_center_store_path, exist_ok=True)
            os.makedirs(self.raw_left_store_path, exist_ok=True)
            os.makedirs(self.raw_right_store_path, exist_ok=True)

        # lux topic
        self.latest_lux = float('nan')
        self.lux_sub = rospy.Subscriber("/lux", Illuminance, self.lux_callback)

        self.dl = deep_learning(n_action=self.action_num)

        if self.save_offline_dataset:
            rospy.loginfo(
                f"[nav_cloning_node] offline dataset save ON: "
                f"{self.dataset_base} "
                f"(stride={self.dataset_stride}, raw_images={self.save_raw_images})"
            )

        if self.model_load_path:
            if os.path.isfile(self.model_load_path):
                self.dl.load(self.model_load_path)
                rospy.loginfo(f"[nav_cloning_node] loaded model: {self.model_load_path}")
            else:
                rospy.logwarn(f"[nav_cloning_node] model_load_path not found: {self.model_load_path}")

        rospy.on_shutdown(self._on_shutdown)

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

    def lux_callback(self, msg):
        self.latest_lux = msg.illuminance

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        self.pos_x = pos.x
        self.pos_y = pos.y

        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x) ** 2 + (pos.y - path.y) ** 2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

        speed = abs(self.vel.linear.x) + abs(self.vel.angular.z)

        if speed >= self.start_speed_threshold:
            self.has_motion = True
            if self.start_on_nav_vel and not self.is_started:
                self.is_started = True

        if self.relay_nav_vel:
            self.nav_pub.publish(self.vel)

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def callback_model_save(self, data):
        save_dir = self.dl.save(self.save_path)
        return TriggerResponse(success=True, message=str(save_dir))

    def _on_shutdown(self):
        if not self.auto_save_on_shutdown:
            return
        if not self.auto_save_model:
            return
        if self.saved_model_path and os.path.isfile(self.saved_model_path):
            return

        try:
            save_dir = self.dl.save(self.save_path, tag=self.start_time)
            self.saved_model_path = os.path.join(save_dir, "model_gpu.pt")
            rospy.loginfo(f"[nav_cloning_node] saved model on shutdown: {self.saved_model_path}")
        except Exception as e:
            rospy.logwarn(f"[nav_cloning_node] failed to save model on shutdown: {e}")

    def _update_select_dl(self, distance):
        if self.mode != "change_dataset_balance":
            return False

        if distance > 0.1:
            self.select_dl = False
        elif distance < 0.05:
            self.select_dl = True

        return self.select_dl

    def _save_raw_images(self, ep, raw_img, raw_img_left, raw_img_right):
        """
        リサイズ前のカメラ画像を保存する。
        cv_bridgeでbgr8として受け取っているため、OpenCV BGR uint8のままPNG保存する。

        保存先：
            dataset/raw_img/center/{ep}_center.png
            dataset/raw_img/left/{ep}_left.png
            dataset/raw_img/right/{ep}_right.png
        """

        if not self.save_raw_images:
            return

        # 安全のためcopyして保存する
        raw_img = raw_img.copy()
        raw_img_left = raw_img_left.copy()
        raw_img_right = raw_img_right.copy()

        cv2.imwrite(
            os.path.join(self.raw_center_store_path, f"{ep}_center.png"),
            raw_img
        )
        cv2.imwrite(
            os.path.join(self.raw_left_store_path, f"{ep}_left.png"),
            raw_img_left
        )
        cv2.imwrite(
            os.path.join(self.raw_right_store_path, f"{ep}_right.png"),
            raw_img_right
        )

    def _maybe_save_offline_sample(
        self,
        img,
        img_left,
        img_right,
        raw_img,
        raw_img_left,
        raw_img_right,
        expert_action,
        distance,
        executed_action,
        policy_action,
        phase,
    ):
        """
        オフライン学習用に保存する。

        保存内容：
        1. 既存互換用：
           dataset/img/{episode}_{view}.npy
           float32 HWC (48,64,3)

        2. 追加：
           dataset/raw_img/center/{episode}_center.png
           dataset/raw_img/left/{episode}_left.png
           dataset/raw_img/right/{episode}_right.png
           uint8 BGR (480,640,3)

        3. CSV：
           dataset/vel/data.csv
        """

        if not self.save_offline_dataset:
            return

        if self.dataset_stride <= 0:
            return

        if (self.episode % self.dataset_stride) != 0:
            return

        # float32 で保存
        img = img.astype(np.float32, copy=False)
        img_left = img_left.astype(np.float32, copy=False)
        img_right = img_right.astype(np.float32, copy=False)

        ep = self.offline_episode

        # ===== 1. resized images for existing offline training =====
        np.save(os.path.join(self.img_store_path, f"{ep}_center.npy"), img)
        np.save(os.path.join(self.img_store_path, f"{ep}_left.npy"), img_left)
        np.save(os.path.join(self.img_store_path, f"{ep}_right.npy"), img_right)

        # ===== 2. raw images for segmentation =====
        self._save_raw_images(
            ep=ep,
            raw_img=raw_img,
            raw_img_left=raw_img_left,
            raw_img_right=raw_img_right,
        )

        # ===== 3. CSV =====
        file_exists = os.path.isfile(self.vel_csv_file)

        with open(self.vel_csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow([
                    'episode',
                    'center',
                    'left',
                    'right',
                    'lux',
                    'pose_x',
                    'pose_y',
                    'distance',
                    'expert',
                    'executed',
                    'policy',
                    'phase',
                ])

            try:
                lux_val = None if np.isnan(self.latest_lux) else round(float(self.latest_lux), 3)
            except Exception:
                lux_val = None

            center = float(expert_action)

            pos_x_val = None if self.pos_x is None else round(float(self.pos_x), 4)
            pos_y_val = None if self.pos_y is None else round(float(self.pos_y), 4)

            writer.writerow([
                ep,
                round(center, 4),
                round(center - self.dataset_action_offset, 4),
                round(center + self.dataset_action_offset, 4),
                lux_val,
                pos_x_val,
                pos_y_val,
                None if distance is None else round(float(distance), 4),
                round(float(expert_action), 4),
                round(float(executed_action), 4),
                None if policy_action is None else round(float(policy_action), 4),
                str(phase),
            ])

        self.offline_episode += 1

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return

        if self.start_on_nav_vel:
            if not self.has_motion:
                return
        else:
            speed = abs(self.vel.linear.x) + abs(self.vel.angular.z)
            if speed >= self.start_speed_threshold:
                self.is_started = True
            if self.is_started == False:
                return

        if (
            self.auto_stop_test
            and self.test_end_step >= 0
            and self.episode >= self.test_end_step
        ):
            if not self.stop_requested:
                self.stop_requested = True
                rospy.loginfo("[nav_cloning_node] test steps completed, stopping.")
                if not self.relay_nav_vel:
                    self.nav_pub.publish(Twist())
                rospy.signal_shutdown("test_steps completed")
            return

        # ===== raw image copy before resize =====
        raw_img = self.cv_image.copy()
        raw_img_left = self.cv_left_image.copy()
        raw_img_right = self.cv_right_image.copy()

        # ===== resized images for learning =====
        img = resize(self.cv_image, (48, 64), mode='constant').astype(np.float32)
        img_left = resize(self.cv_left_image, (48, 64), mode='constant').astype(np.float32)
        img_right = resize(self.cv_right_image, (48, 64), mode='constant').astype(np.float32)

        ros_time = str(rospy.Time.now())

        if (
            not self.auto_test_done
            and self.auto_test_step >= 0
            and self.episode >= self.auto_test_step
        ):
            self.auto_test_done = True

            if self.auto_save_model:
                save_dir = self.dl.save(self.save_path, tag=self.start_time)
                self.saved_model_path = os.path.join(save_dir, "model_gpu.pt")
                rospy.loginfo(f"[nav_cloning_node] saved model: {self.saved_model_path}")

            if self.auto_load_model and self.saved_model_path:
                if os.path.isfile(self.saved_model_path):
                    self.dl.load(self.saved_model_path)
                    rospy.loginfo(f"[nav_cloning_node] loaded model: {self.saved_model_path}")
                else:
                    rospy.logwarn(f"[nav_cloning_node] saved model missing: {self.saved_model_path}")

            self.learning = False

        if self.learning:
            expert_action = self.action
            target_action = expert_action
            distance = self.min_distance

            policy_action = target_action

            if self.mode == "change_dataset_balance":
                if distance < 0.05:
                    action, loss = self.dl.act_and_trains(img, target_action)
                    policy_action = action

                    if abs(target_action) < 0.1:
                        action_left, loss_left = self.dl.act_and_trains(img_left, target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)

                elif 0.05 <= distance < 0.1:
                    self.dl.make_dataset(img, target_action)
                    action, loss = self.dl.act_and_trains(img, target_action)
                    policy_action = action

                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left, target_action - 0.2)
                        action_left, loss_left = self.dl.act_and_trains(img_left, target_action - 0.2)

                        self.dl.make_dataset(img_right, target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)

                    line = [str(self.episode), "training", str(distance)]
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)

                else:
                    self.dl.make_dataset(img, target_action)
                    self.dl.make_dataset(img, target_action)

                    action, loss = self.dl.act_and_trains(img, target_action)
                    policy_action = action

                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left, target_action - 0.2)
                        self.dl.make_dataset(img_left, target_action - 0.2)
                        action_left, loss_left = self.dl.act_and_trains(img_left, target_action - 0.2)

                        self.dl.make_dataset(img_right, target_action + 0.2)
                        self.dl.make_dataset(img_right, target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)

                    line = [str(self.episode), "training", str(distance)]
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)

                # 学習は常に行い、mixed時のみ expert/policy を切り替える
                if self._update_select_dl(distance) and self.episode >= 0:
                    target_action = action

            else:
                action, loss = self.dl.act_and_trains(img, target_action)
                policy_action = action

            # ===== cmd action selection =====
            if self.cmd_vel_mode == "policy":
                cmd_action = policy_action
            elif self.cmd_vel_mode == "expert":
                cmd_action = expert_action
            else:
                cmd_action = target_action

            # ===== offline dataset save =====
            self._maybe_save_offline_sample(
                img=img,
                img_left=img_left,
                img_right=img_right,
                raw_img=raw_img,
                raw_img_left=raw_img_left,
                raw_img_right=raw_img_right,
                expert_action=self.action,
                distance=distance,
                executed_action=cmd_action,
                policy_action=policy_action,
                phase="training",
            )

            self.episode += 1

            print(str(self.episode) + ", training, distance: " + str(distance))

            line = [str(self.episode), "training", str(distance)]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)

            cmd_vel = Twist()

            if self.cmd_vel_linear_source == "nav_vel":
                cmd_vel.linear.x = self.vel.linear.x
            else:
                cmd_vel.linear.x = self.cmd_vel_linear_fixed

            cmd_vel.angular.z = cmd_action

            if not self.relay_nav_vel:
                self.nav_pub.publish(cmd_vel)

        else:
            expert_action = self.action
            policy_action = self.dl.act(img)
            distance = self.min_distance

            # Test must run with policy output only
            cmd_action = policy_action

            # ===== offline dataset save =====
            self._maybe_save_offline_sample(
                img=img,
                img_left=img_left,
                img_right=img_right,
                raw_img=raw_img,
                raw_img_left=raw_img_left,
                raw_img_right=raw_img_right,
                expert_action=expert_action,
                distance=distance,
                executed_action=cmd_action,
                policy_action=policy_action,
                phase="test",
            )

            print(str(self.episode) + ", test, angular:" + str(cmd_action) + ", distance: " + str(distance))

            self.episode += 1

            angle_error = abs(self.action - cmd_action)

            line = [str(self.episode), "test", str(distance)]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)

            cmd_vel = Twist()

            if self.cmd_vel_linear_source == "nav_vel":
                cmd_vel.linear.x = self.vel.linear.x
            else:
                cmd_vel.linear.x = self.cmd_vel_linear_fixed

            cmd_vel.angular.z = cmd_action

            if not self.relay_nav_vel:
                self.nav_pub.publish(cmd_vel)

        if self.show_debug:
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
