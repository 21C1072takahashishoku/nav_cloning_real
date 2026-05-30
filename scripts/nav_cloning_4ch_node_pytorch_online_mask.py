#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-robot nav_cloning node for 4ch input: BGR + Mask2Former ground mask.

This file is generated as a non-destructive addition.
It is based on nav_cloning_fullpic_node_pytorch_offline_collect.py.
Main differences:
    - deep_learning(n_channel=4)
    - subscribes /segmentation/ground_mask_{center,left,right}
    - converts each camera frame to float32 HWC=(48,64,4)
    - stores offline dataset as .npy with shape (48,64,4)

[修正] 2026-05-24: np.save / cv2.imwrite をメインループから切り離し、
       別スレッドのキューで非同期保存することで走行中の一時停止を解消。
[修正] 2026-05-30: マスク画像を raw_img と同じフォルダに保存するよう追加。
"""
from __future__ import print_function
from numpy import dtype
import numpy as np
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import threading
from queue import Queue

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_DIR = os.path.join(SCRIPT_DIR, "pytorch")
if PYTORCH_DIR not in sys.path:
    sys.path.insert(0, PYTORCH_DIR)

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

        self.save_offline_dataset = rospy.get_param("/nav_cloning_node/save_offline_dataset", True)
        self.save_raw_images = rospy.get_param("/nav_cloning_node/save_raw_images", True)

        self.dataset_id = rospy.get_param("/nav_cloning_node/dataset_id", "")
        self.dataset_stride = int(rospy.get_param("/nav_cloning_node/dataset_stride", 1))
        self.dataset_action_offset = float(rospy.get_param("/nav_cloning_node/dataset_action_offset", 0.2))

        self.offline_episode = 0
        self.action_num = 1
        self.bridge = CvBridge()

        self.mask_center_topic = rospy.get_param(
            "/nav_cloning_node/mask_center_topic",
            "/segmentation/ground_mask_center"
        )
        self.mask_left_topic = rospy.get_param(
            "/nav_cloning_node/mask_left_topic",
            "/segmentation/ground_mask_left"
        )
        self.mask_right_topic = rospy.get_param(
            "/nav_cloning_node/mask_right_topic",
            "/segmentation/ground_mask_right"
        )
        self.max_mask_age_sec = float(
            rospy.get_param("/nav_cloning_node/max_mask_age_sec", 0.5)
        )
        self.require_fresh_mask = bool(
            rospy.get_param("/nav_cloning_node/require_fresh_mask", True)
        )

        self.cv_mask_center = np.zeros((48, 64), np.uint8)
        self.cv_mask_left = np.zeros((48, 64), np.uint8)
        self.cv_mask_right = np.zeros((48, 64), np.uint8)

        self.got_mask_center = False
        self.got_mask_left = False
        self.got_mask_right = False

        self.mask_stamp_center = rospy.Time(0)
        self.mask_stamp_left = rospy.Time(0)
        self.mask_stamp_right = rospy.Time(0)

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

        # ===== 非同期保存キュー =====
        self._save_queue = Queue(maxsize=200)
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()

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

        self.mask_center_sub = rospy.Subscriber(
            self.mask_center_topic,
            Image,
            self.callback_mask_center,
            queue_size=1
        )
        self.mask_left_sub = rospy.Subscriber(
            self.mask_left_topic,
            Image,
            self.callback_mask_left,
            queue_size=1
        )
        self.mask_right_sub = rospy.Subscriber(
            self.mask_right_topic,
            Image,
            self.callback_mask_right,
            queue_size=1
        )

        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)

        self.pose_sub = rospy.Subscriber("/mcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.callback_path)

        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()

        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_left_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_right_image = np.zeros((480, 640, 3), np.uint8)

        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")

        self.auto_test_step = self.train_steps
        self.test_end_step = (
            self.train_steps + self.test_steps
            if self.test_steps > 0
            else -1
        )

        self.auto_test_done = False
        self.stop_requested = False
        self.saved_model_path = ""

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

        self.dataset_base = (
            roslib.packages.get_pkg_dir('nav_cloning')
            + '/data/'
            + str(self.dataset_id)
            + '/dataset/'
        )

        self.img_store_path = os.path.join(self.dataset_base, 'img')
        self.vel_store_path = os.path.join(self.dataset_base, 'vel')
        self.vel_csv_file = os.path.join(self.vel_store_path, 'data.csv')
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

        self.latest_lux = float('nan')
        self.lux_sub = rospy.Subscriber("/lux", Illuminance, self.lux_callback)

        self.dl = deep_learning(n_channel=4, n_action=self.action_num)

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

    # ===== 非同期保存ワーカー =====
    def _save_worker(self):
        """保存キューを別スレッドで処理。メインループをブロックしない。"""
        while True:
            task = self._save_queue.get()
            if task is None:
                break
            try:
                fn, args, kwargs = task
                fn(*args, **kwargs)
            except Exception as e:
                rospy.logwarn(f"[nav_cloning_node] save_worker error: {e}")
            finally:
                self._save_queue.task_done()

    def _enqueue_save(self, fn, *args, **kwargs):
        """保存タスクをキューに投入。キューが満杯なら警告してスキップ。"""
        try:
            self._save_queue.put_nowait((fn, args, kwargs))
        except Exception:
            rospy.logwarn_throttle(5.0, "[nav_cloning_node] save_queue full, skipping save")

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

    def _valid_stamp_or_now(self, stamp):
        if stamp is None or stamp.to_sec() <= 0.0:
            return rospy.Time.now()
        return stamp

    def callback_mask_center(self, data):
        try:
            self.cv_mask_center = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.mask_stamp_center = self._valid_stamp_or_now(data.header.stamp)
            self.got_mask_center = True
        except CvBridgeError as e:
            print(e)

    def callback_mask_left(self, data):
        try:
            self.cv_mask_left = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.mask_stamp_left = self._valid_stamp_or_now(data.header.stamp)
            self.got_mask_left = True
        except CvBridgeError as e:
            print(e)

    def callback_mask_right(self, data):
        try:
            self.cv_mask_right = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.mask_stamp_right = self._valid_stamp_or_now(data.header.stamp)
            self.got_mask_right = True
        except CvBridgeError as e:
            print(e)

    def _mask_is_ready(self):
        if not (self.got_mask_center and self.got_mask_left and self.got_mask_right):
            rospy.logwarn_throttle(2.0, "[nav_cloning_4ch] waiting for Mask2Former ground masks")
            return False

        if not self.require_fresh_mask:
            return True

        now = rospy.Time.now()
        ages = [
            (now - self.mask_stamp_center).to_sec(),
            (now - self.mask_stamp_left).to_sec(),
            (now - self.mask_stamp_right).to_sec(),
        ]
        if max(ages) > self.max_mask_age_sec:
            rospy.logwarn_throttle(
                2.0,
                "[nav_cloning_4ch] stale mask. ages center/left/right = %.3f/%.3f/%.3f sec, max=%.3f sec",
                ages[0], ages[1], ages[2], self.max_mask_age_sec
            )
            return False
        return True

    def to_4ch_hwc(self, bgr_img, mask_img_0_255, out_hw=(48, 64)):
        H, W = out_hw

        img = resize(
            bgr_img,
            (H, W),
            mode='constant',
            preserve_range=False
        ).astype(np.float32)

        if mask_img_0_255.shape[:2] != (H, W):
            mask_small = cv2.resize(
                mask_img_0_255,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_small = mask_img_0_255

        mask01 = (mask_small.astype(np.float32) / 255.0)[:, :, None]
        img4 = np.concatenate([img, mask01], axis=2).astype(np.float32)

        if img4.shape != (H, W, 4):
            raise RuntimeError("4ch shape error: {}".format(img4.shape))

        return img4

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
        # 保存キューを安全に終了
        self._save_queue.put(None)
        self._save_thread.join(timeout=10.0)

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

    def _save_raw_images_sync(self, ep, raw_img, raw_img_left, raw_img_right,
                               mask_center, mask_left, mask_right):
        """実際の保存処理（saveワーカースレッドから呼ばれる）。"""
        if not self.save_raw_images:
            return
        # 生カメラ画像
        cv2.imwrite(os.path.join(self.raw_center_store_path, f"{ep}_center.png"), raw_img)
        cv2.imwrite(os.path.join(self.raw_left_store_path, f"{ep}_left.png"), raw_img_left)
        cv2.imwrite(os.path.join(self.raw_right_store_path, f"{ep}_right.png"), raw_img_right)
        # [修正] マスク画像を追加保存
        cv2.imwrite(os.path.join(self.raw_center_store_path, f"{ep}_center_mask.png"), mask_center)
        cv2.imwrite(os.path.join(self.raw_left_store_path, f"{ep}_left_mask.png"), mask_left)
        cv2.imwrite(os.path.join(self.raw_right_store_path, f"{ep}_right_mask.png"), mask_right)

    def _save_npy_sync(self, path, arr):
        """np.saveをワーカースレッドから呼ぶ用。"""
        np.save(path, arr)

    def _save_csv_sync(self, row, file_exists):
        """CSV書き込みをワーカースレッドから呼ぶ用。"""
        with open(self.vel_csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow([
                    'episode', 'center', 'left', 'right', 'lux',
                    'pose_x', 'pose_y', 'distance', 'expert',
                    'executed', 'policy', 'phase',
                ])
            writer.writerow(row)

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
        rospy.loginfo_throttle(
            1.0,
            f"[4ch save check] episode={self.episode}, offline_episode={self.offline_episode}, "
            f"save_offline_dataset={self.save_offline_dataset}, "
            f"dataset_stride={self.dataset_stride}, "
            f"img_shape={getattr(img, 'shape', None)}, "
            f"left_shape={getattr(img_left, 'shape', None)}, "
            f"right_shape={getattr(img_right, 'shape', None)}, "
            f"img_store_path={self.img_store_path}"
        )

        if not self.save_offline_dataset:
            rospy.logwarn_throttle(1.0, "[4ch save skip] save_offline_dataset is False")
            return

        if self.dataset_stride <= 0:
            rospy.logwarn_throttle(1.0, f"[4ch save skip] dataset_stride={self.dataset_stride}")
            return

        if (self.episode % self.dataset_stride) != 0:
            return

        img = img.astype(np.float32, copy=False)
        img_left = img_left.astype(np.float32, copy=False)
        img_right = img_right.astype(np.float32, copy=False)

        ep = self.offline_episode

        rospy.loginfo_throttle(
            1.0,
            f"[4ch save DO] ep={ep}, center_shape={img.shape}, save_to={self.img_store_path}"
        )

        # ===== 非同期保存 =====
        self._enqueue_save(
            self._save_npy_sync,
            os.path.join(self.img_store_path, f"{ep}_center.npy"),
            img.copy()
        )
        self._enqueue_save(
            self._save_npy_sync,
            os.path.join(self.img_store_path, f"{ep}_left.npy"),
            img_left.copy()
        )
        self._enqueue_save(
            self._save_npy_sync,
            os.path.join(self.img_store_path, f"{ep}_right.npy"),
            img_right.copy()
        )
        # [修正] マスクのコピーを渡してスレッド競合を回避
        self._enqueue_save(
            self._save_raw_images_sync,
            ep,
            raw_img.copy(),
            raw_img_left.copy(),
            raw_img_right.copy(),
            self.cv_mask_center.copy(),
            self.cv_mask_left.copy(),
            self.cv_mask_right.copy()
        )

        # CSV のみ同期（軽量なのでブロック時間は無視できる）
        try:
            lux_val = None if np.isnan(self.latest_lux) else round(float(self.latest_lux), 3)
        except Exception:
            lux_val = None

        center = float(expert_action)
        pos_x_val = None if self.pos_x is None else round(float(self.pos_x), 4)
        pos_y_val = None if self.pos_y is None else round(float(self.pos_y), 4)

        row = [
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
        ]
        file_exists = os.path.isfile(self.vel_csv_file)
        self._enqueue_save(self._save_csv_sync, row, file_exists)

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

        raw_img = self.cv_image.copy()
        raw_img_left = self.cv_left_image.copy()
        raw_img_right = self.cv_right_image.copy()

        if not self._mask_is_ready():
            return

        img = self.to_4ch_hwc(self.cv_image, self.cv_mask_center)
        img_left = self.to_4ch_hwc(self.cv_left_image, self.cv_mask_left)
        img_right = self.to_4ch_hwc(self.cv_right_image, self.cv_mask_right)

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

                if self._update_select_dl(distance) and self.episode >= 0:
                    target_action = action

            else:
                action, loss = self.dl.act_and_trains(img, target_action)
                policy_action = action

            if self.cmd_vel_mode == "policy":
                cmd_action = policy_action
            elif self.cmd_vel_mode == "expert":
                cmd_action = expert_action
            else:
                cmd_action = target_action

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

            cmd_action = policy_action

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
            cv2.imshow("4ch Center Image BGR", copy.deepcopy(img[:, :, :3]))
            cv2.imshow("4ch Left Image BGR", copy.deepcopy(img_left[:, :, :3]))
            cv2.imshow("4ch Right Image BGR", copy.deepcopy(img_right[:, :, :3]))
            cv2.imshow("Ground Mask Center", copy.deepcopy(img[:, :, 3]))
            cv2.imshow("Ground Mask Left", copy.deepcopy(img_left[:, :, 3]))
            cv2.imshow("Ground Mask Right", copy.deepcopy(img_right[:, :, 3]))
            cv2.waitKey(1)


if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)

    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
