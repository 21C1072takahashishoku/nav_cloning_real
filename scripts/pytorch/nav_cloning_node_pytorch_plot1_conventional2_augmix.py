#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
# --- sensor_msgs の Image を ROSImage などにしてもOKですが、
#     ここでは衝突回避のため PIL を別名にします ---
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *  # deep_learning クラスなどが定義されている想定
from geometry_msgs.msg import Twist, PoseArray, PoseWithCovarianceStamped
from std_srvs.srv import Trigger, SetBool, SetBoolResponse
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32, Bool, String
import csv
import os
import copy
import sys
import tf
import time
import numpy as np

# --- PIL & AugMix ---
from PIL import Image as PILImage
current_dir = os.path.dirname(os.path.abspath(__file__))
augmix_dir = os.path.join(current_dir, '..', '..', 'augmix')
augmix_dir = os.path.abspath(augmix_dir)
sys.path.append(augmix_dir)
from augment_and_mix import augment_and_mix


def apply_augmix_bgr_uint8(cv_image, final_size=(48, 64)):
    """
    学習時に使う用:
      1) 入力は BGR, uint8(0..255)
      2) まず RGB, float(0..1) へ変換して AugMix
      3) AugMix結果(RGB, float)を BGR, uint8(0..255) に戻して返す
    """
    try:
        # 1. BGR -> RGB
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # shape=(H,W,3), uint8

        # 2. PIL へ変換してリサイズ (width=final_size[1], height=final_size[0])
        pil_img = PILImage.fromarray(img_rgb)
        pil_img = pil_img.resize((final_size[1], final_size[0]), PILImage.LANCZOS)

        # 3. NumPy float32, 0..1 に変換 → AugMix
        img_np = np.array(pil_img, dtype=np.float32) / 255.0  # shape=(h,w,3), float32, RGB
        aug_img = augment_and_mix(img_np)  # AugMix結果, shape=(h,w,3), float32, RGB
        aug_img = np.clip(aug_img, 0.0, 1.0)

        # 4. AugMix結果(RGB, [0..1] float) → BGR, uint8(0..255)
        aug_img_bgr = aug_img[..., ::-1]  # RGB -> BGR
        aug_img_bgr_uint8 = (aug_img_bgr * 255.0).astype(np.uint8)

        return aug_img_bgr_uint8

    except Exception as e:
        rospy.logerr(f"[apply_augmix_bgr_uint8] Error: {e}")
        return None


def resize_opencv_bgr(cv_image, final_size=(48, 64)):
    """
    テスト時用: BGR, uint8(0..255) のまま OpenCV でリサイズ
    """
    # cv2.resize() は引数が (width, height) の順
    resized = cv2.resize(
        cv_image,
        (final_size[1], final_size[0]),
        interpolation=cv2.INTER_AREA
    )
    return resized


class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        
        # 例: /nav_cloning_node/mode パラメータ (デフォルト: "change_dataset_balance")
        self.mode = rospy.get_param("/nav_cloning_node/mode", "change_dataset_balance")
        
        self.action_num = 1
        self.dl = deep_learning(n_action=self.action_num)
        self.bridge = CvBridge()

        # --- ROS イメージトピックのサブスクライバ ---
        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", ROSImage, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/usb_cam/image_raw", ROSImage, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/usb_cam/image_raw", ROSImage, self.callback_right_camera)

        # 外部ナビ (move_base 等) の出力を購読して教師データとして使う
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)

        # nav_cloning_node 側の /icart_mini/cmd_vel パブリッシャ
        self.nav_pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
        
        # サービス
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        
        # 自己位置やパスなど
        self.pose_sub = rospy.Subscriber("/mcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.callback_path)

        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        
        # カメラ画像バッファ (BGR, uint8)
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)

        # フラグ類
        self.learning = True
        self.select_dl = False
        
        # ログ保存用パス
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_night_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_night_'+str(self.mode)+'/'

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.previous_reset_time = 0
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        
        self.dir_pub = rospy.Publisher('/dir', String, queue_size=1)
        self.mode_pub = rospy.Publisher("/mode", Bool, queue_size=1)
        self.episode_csv_flg = 0
        
        self.tracker_sub = rospy.Subscriber("/waypoint_manager/waypoint/is_reached", Bool, self.callback_reached)
        self.waypoint_reach_flg = False
        self.waypoint_count = 0
        self.laps_pub = rospy.Publisher('/laps', Bool, queue_size=1)
        self.kill_flg = False
        self.kill_count = 0

    # ---------------- Callback methods ----------------
    def callback(self, data):
        """ メインカメラのコールバック (BGR, uint8) """
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def callback_left_camera(self, data):
        """ 左カメラのコールバック (BGR, uint8) """
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def callback_right_camera(self, data):
        """ 右カメラのコールバック (BGR, uint8) """
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            dist = np.sqrt((pos.x - path.x)**2 + (pos.y - path.y)**2)
            distance_list.append(dist)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_vel(self, data):
        # /nav_vel を購読→ロボット制御の教師データとして使う
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
        model_res.message = "model_save"
        model_res.success = True
        return model_res

    def callback_exit(self, data):
        self.exit_flg = data.data

    def callback_reached(self, data):
        self.waypoint_reach_flg = data.data
        if self.waypoint_reach_flg:
            self.waypoint_count += 1
        if self.waypoint_count == 8:
            self.waypoint_count = 0
            self.laps_pub.publish(True)

    # ---------------- Main loop ----------------
    def loop(self):
        # (1) カメラ画像が正しく入ってなければスキップ
        if self.cv_image.size != 640*480*3:
            return
        if self.cv_left_image.size != 640*480*3:
            return
        if self.cv_right_image.size != 640*480*3:
            return

        # (2) ロボットが動き始めるまでは待機
        #     ＝ /nav_vel で何かしら動きが出るのを待つ
        if self.vel.linear.x != 0:
            self.is_started = True
        if not self.is_started:
            return

        self.dir_pub.publish(self.path + self.start_time)

        # (3) 学習 or テスト で画像処理を分岐
        if self.learning:
            # --- 学習フェーズ: AugMix をかけて BGR, uint8 に戻す ---
            img       = apply_augmix_bgr_uint8(self.cv_image,       final_size=(48, 64))
            img_left  = apply_augmix_bgr_uint8(self.cv_left_image,  final_size=(48, 64))
            img_right = apply_augmix_bgr_uint8(self.cv_right_image, final_size=(48, 64))

            if img is None or img_left is None or img_right is None:
                rospy.logwarn("AugMix returned None. Skipping this loop.")
                return
        else:
            # --- テストフェーズ: OpenCV で BGRリサイズのみ ---
            img       = resize_opencv_bgr(self.cv_image,       final_size=(48, 64))
            img_left  = resize_opencv_bgr(self.cv_left_image,  final_size=(48, 64))
            img_right = resize_opencv_bgr(self.cv_right_image, final_size=(48, 64))

        # (4) エピソード数で学習->テストへ切り替え or 強制終了など
        if self.episode == 4000:
            self.learning = False
            self.dl.save(self.save_path)

        mode_bool = self.learning
        self.mode_pub.publish(mode_bool)

        if self.episode == 7000:
            os.system('killall roslaunch')
            sys.exit()

        # (5) 学習 or テスト本体
        distance = self.min_distance

        if self.learning:
            # ----------------
            # 学習時でも外部ナビ (/nav_vel) の速度を中継パブリッシュ
            # ----------------
            # 1) 教師データとして外部ナビの角速度を取得
            target_action = self.action

            # 2) たとえば "change_dataset_balance" モードの場合
            if self.mode == "change_dataset_balance":
                if distance < 0.05:
                    action, loss = self.dl.act_and_trains(img, target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left,  target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)
                elif 0.05 <= distance < 0.1:
                    self.dl.make_dataset(img, target_action)
                    action, loss = self.dl.act_and_trains(img, target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left,  target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left,  target_action - 0.2)
                        self.dl.make_dataset(img_right, target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)
                else:
                    for _ in range(4):
                        self.dl.make_dataset(img, target_action)
                    action, loss = self.dl.act_and_trains(img, target_action)
                    if abs(target_action) < 0.1:
                        for _ in range(4):
                            self.dl.make_dataset(img_left,  target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left,  target_action - 0.2)

                        for _ in range(4):
                            self.dl.make_dataset(img_right, target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right, target_action + 0.2)

            # 3) ログ書き
            self.episode += 1
            rospy.loginfo(f"{self.episode}, training, distance: {distance}")
            lines = [
                str(self.episode),
                "training",
                str(distance),
                str(self.pos_x),
                str(self.pos_y),
                str(self.pos_the)
            ]
            # もし angle_error を計算してCSVに書きたい場合は適宜追加
            with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(lines)

            # 4) 学習時にも「外部の /nav_vel をブリッジ」＝ ここで /icart_mini/cmd_vel に出力
            #    あるいは独自に self.vel.linear.x = 0.2 にしてもOK
            self.nav_pub.publish(self.vel)

        else:
            # テストフェーズ: nav_cloning_node が推論でロボットを動かす
            target_action = self.dl.act(img)
            rospy.loginfo(f"{self.episode}, test, distance: {distance}")
            self.episode += 1

            angle_error = abs(self.action - target_action)
            lines = [
                str(self.episode),
                "test",
                str(distance),
                str(self.pos_x),
                str(self.pos_y),
                str(self.pos_the)
            ]
            # ログ書き
            with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(lines)

            if angle_error < 0.05:
                with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(lines)

            # テスト時は nav_cloning_node が自律移動 (推論結果) でロボットを動かす
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        # (6) 画像表示 (BGR, uint8)
        if self.learning:
            disp_img       = copy.deepcopy(img)
            disp_img_left  = copy.deepcopy(img_left)
            disp_img_right = copy.deepcopy(img_right)
        else:
            disp_img       = copy.deepcopy(img)
            disp_img_left  = copy.deepcopy(img_left)
            disp_img_right = copy.deepcopy(img_right)

        cv2.imshow("Resized Center Image", disp_img)
        cv2.imshow("Resized Left Image", disp_img_left)
        cv2.imshow("Resized Right Image", disp_img_right)
        cv2.waitKey(1)


if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()

