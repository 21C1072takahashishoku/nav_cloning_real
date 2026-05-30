#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#4ch実機走行中に、topic周期、CPU、RAM、GPU、cmd_velをCSV保存する監視ノード

import os
import csv
import time
import subprocess
from collections import deque

import rospy
import psutil

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class TopicHzMonitor:
    def __init__(self, window_sec=5.0):
        self.window_sec = window_sec
        self.times = deque()

    def tick(self):
        now = time.time()
        self.times.append(now)
        while self.times and now - self.times[0] > self.window_sec:
            self.times.popleft()

    def hz(self):
        if len(self.times) < 2:
            return 0.0
        duration = self.times[-1] - self.times[0]
        if duration <= 0:
            return 0.0
        return (len(self.times) - 1) / duration


class PerfMonitor4ch:
    def __init__(self):
        rospy.init_node("perf_monitor_4ch", anonymous=False)

        self.dataset_id = rospy.get_param("~dataset_id", "real_4ch_short_test")
        self.log_rate_hz = float(rospy.get_param("~log_rate_hz", 1.0))

        self.base_dir = os.path.join(
            os.path.expanduser("~"),
            "/home/orne_beta/shoku_ws/src/nav_cloning/data",
            self.dataset_id,
            "perf"
        )
        os.makedirs(self.base_dir, exist_ok=True)

        self.csv_path = os.path.join(
            self.base_dir,
            time.strftime("perf_%Y%m%d_%H%M%S.csv")
        )

        self.monitors = {
            "camera_center_hz": TopicHzMonitor(),
            "camera_left_hz": TopicHzMonitor(),
            "camera_right_hz": TopicHzMonitor(),
            "mask_center_hz": TopicHzMonitor(),
            "mask_left_hz": TopicHzMonitor(),
            "mask_right_hz": TopicHzMonitor(),
            "nav_vel_hz": TopicHzMonitor(),
            "cmd_vel_hz": TopicHzMonitor(),
        }

        self.latest_nav_vel = Twist()
        self.latest_cmd_vel = Twist()

        rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.cb_camera_center, queue_size=1)
        rospy.Subscriber("/camera_left/usb_cam/image_raw", Image, self.cb_camera_left, queue_size=1)
        rospy.Subscriber("/camera_right/usb_cam/image_raw", Image, self.cb_camera_right, queue_size=1)

        rospy.Subscriber("/segmentation/ground_mask_center", Image, self.cb_mask_center, queue_size=1)
        rospy.Subscriber("/segmentation/ground_mask_left", Image, self.cb_mask_left, queue_size=1)
        rospy.Subscriber("/segmentation/ground_mask_right", Image, self.cb_mask_right, queue_size=1)

        rospy.Subscriber("/nav_vel", Twist, self.cb_nav_vel, queue_size=1)
        rospy.Subscriber("/icart_mini/cmd_vel", Twist, self.cb_cmd_vel, queue_size=1)

        self.header = [
            "time",
            "elapsed_sec",
            "cpu_percent",
            "ram_percent",
            "ram_used_gb",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "gpu_temp_c",
            "camera_center_hz",
            "camera_left_hz",
            "camera_right_hz",
            "mask_center_hz",
            "mask_left_hz",
            "mask_right_hz",
            "nav_vel_hz",
            "cmd_vel_hz",
            "nav_vel_linear_x",
            "nav_vel_angular_z",
            "cmd_vel_linear_x",
            "cmd_vel_angular_z",
            "npy_file_count",
        ]

        self.start_time = time.time()

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

        rospy.loginfo("[perf_monitor_4ch] started")
        rospy.loginfo("[perf_monitor_4ch] log: %s", self.csv_path)

    def cb_camera_center(self, msg):
        self.monitors["camera_center_hz"].tick()

    def cb_camera_left(self, msg):
        self.monitors["camera_left_hz"].tick()

    def cb_camera_right(self, msg):
        self.monitors["camera_right_hz"].tick()

    def cb_mask_center(self, msg):
        self.monitors["mask_center_hz"].tick()

    def cb_mask_left(self, msg):
        self.monitors["mask_left_hz"].tick()

    def cb_mask_right(self, msg):
        self.monitors["mask_right_hz"].tick()

    def cb_nav_vel(self, msg):
        self.monitors["nav_vel_hz"].tick()
        self.latest_nav_vel = msg

    def cb_cmd_vel(self, msg):
        self.monitors["cmd_vel_hz"].tick()
        self.latest_cmd_vel = msg

    def get_gpu_info(self):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            parts = [x.strip() for x in out.split(",")]
            return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            return -1.0, -1.0, -1.0, -1.0

    def count_npy_files(self):
        img_dir = os.path.join(
            os.path.expanduser("~"),
            "shoku_ws/src/nav_cloning/data",
            self.dataset_id,
            "dataset/img"
        )
        if not os.path.isdir(img_dir):
            return 0
        count = 0
        for name in os.listdir(img_dir):
            if name.endswith(".npy"):
                count += 1
        return count

    def run(self):
        rate = rospy.Rate(self.log_rate_hz)

        while not rospy.is_shutdown():
            now = time.time()
            elapsed = now - self.start_time

            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used_gb = ram.used / (1024 ** 3)

            gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = self.get_gpu_info()

            row = [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(elapsed, 3),
                cpu_percent,
                ram_percent,
                round(ram_used_gb, 3),
                gpu_util,
                gpu_mem_used,
                gpu_mem_total,
                gpu_temp,
                round(self.monitors["camera_center_hz"].hz(), 3),
                round(self.monitors["camera_left_hz"].hz(), 3),
                round(self.monitors["camera_right_hz"].hz(), 3),
                round(self.monitors["mask_center_hz"].hz(), 3),
                round(self.monitors["mask_left_hz"].hz(), 3),
                round(self.monitors["mask_right_hz"].hz(), 3),
                round(self.monitors["nav_vel_hz"].hz(), 3),
                round(self.monitors["cmd_vel_hz"].hz(), 3),
                round(self.latest_nav_vel.linear.x, 4),
                round(self.latest_nav_vel.angular.z, 4),
                round(self.latest_cmd_vel.linear.x, 4),
                round(self.latest_cmd_vel.angular.z, 4),
                self.count_npy_files(),
            ]

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            rospy.loginfo_throttle(
                5.0,
                "[perf] cam=%.1f/%.1f/%.1fHz mask=%.1f/%.1f/%.1fHz GPU=%.0f%% mem=%.0f/%.0fMB cmd=(%.3f, %.3f) npy=%d",
                self.monitors["camera_center_hz"].hz(),
                self.monitors["camera_left_hz"].hz(),
                self.monitors["camera_right_hz"].hz(),
                self.monitors["mask_center_hz"].hz(),
                self.monitors["mask_left_hz"].hz(),
                self.monitors["mask_right_hz"].hz(),
                gpu_util,
                gpu_mem_used,
                gpu_mem_total,
                self.latest_cmd_vel.linear.x,
                self.latest_cmd_vel.angular.z,
                self.count_npy_files(),
            )

            rate.sleep()


if __name__ == "__main__":
    node = PerfMonitor4ch()
    node.run()
