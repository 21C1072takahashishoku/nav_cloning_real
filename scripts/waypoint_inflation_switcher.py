#!/usr/bin/env python3
import re
import rospy
from std_msgs.msg import Bool
from dynamic_reconfigure.client import Client as DynClient
from waypoint_manager_msgs.msg import Waypoint


class WaypointInflationSwitcher:
    def __init__(self):
        self.mode = rospy.get_param("~mode", "count")
        self.topic = rospy.get_param("~topic", "/waypoint_manager/waypoint/is_reached")
        self.waypoint_topic = rospy.get_param("~waypoint_topic", "/waypoint_manager/waypoint")
        self.identity_regex = rospy.get_param("~identity_regex", r"^wp_(\d+)$")
        if "\\\\" in self.identity_regex:
            # Normalize any double-escaped patterns from launch files.
            self.identity_regex = self.identity_regex.replace("\\\\", "\\")
        self.wp_small = int(rospy.get_param("~wp_small", 10))
        self.wp_large = int(rospy.get_param("~wp_large", 15))
        self.index_small = int(rospy.get_param("~index_small", 9))
        self.index_large = int(rospy.get_param("~index_large", 0))
        self.radius_small = float(rospy.get_param("~radius_small", 0.2))
        self.radius_large = float(rospy.get_param("~radius_large", 0.6))
        self.global_layer = rospy.get_param(
            "~global_layer", "/move_base/global_costmap/inflation_layer"
        )
        self.local_layer = rospy.get_param(
            "~local_layer", "/move_base/local_costmap/inflation_layer"
        )
        self.retry_hz = float(rospy.get_param("~retry_hz", 1.0))

        self.count = 0
        self.last_index = None
        self.last_reached = False
        self.target_radius = None

        self._global_client = None
        self._local_client = None

        if self.mode == "identity":
            rospy.Subscriber(self.waypoint_topic, Waypoint, self._waypoint_cb, queue_size=10)
        else:
            rospy.Subscriber(self.topic, Bool, self._reached_cb, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / self.retry_hz), self._timer_cb)

        rospy.loginfo(
            "waypoint_inflation_switcher: mode=%s topic=%s waypoint_topic=%s "
            "wp_small=%d wp_large=%d index_small=%d index_large=%d identity_regex=%s "
            "radius_small=%.3f radius_large=%.3f",
            self.mode,
            self.topic,
            self.waypoint_topic,
            self.wp_small,
            self.wp_large,
            self.index_small,
            self.index_large,
            self.identity_regex,
            self.radius_small,
            self.radius_large,
        )

    def _ensure_clients(self):
        if self._global_client is None:
            self._global_client = DynClient(self.global_layer, timeout=2.0)
        if self._local_client is None:
            self._local_client = DynClient(self.local_layer, timeout=2.0)

    def _apply_radius(self, radius):
        try:
            self._ensure_clients()
            self._global_client.update_configuration({"inflation_radius": radius})
            self._local_client.update_configuration({"inflation_radius": radius})
            rospy.loginfo("inflation_radius set to %.3f", radius)
            return True
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0, "failed to set inflation_radius: %s", str(exc)
            )
            self._global_client = None
            self._local_client = None
            return False

    def _reached_cb(self, msg):
        if msg.data and not self.last_reached:
            self.count += 1
            rospy.loginfo("waypoint reached count=%d", self.count)
            if self.count == self.wp_small:
                self.target_radius = self.radius_small
            elif self.count == self.wp_large:
                self.target_radius = self.radius_large
        self.last_reached = msg.data

    def _extract_index(self, identity):
        match = re.match(self.identity_regex, identity)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _waypoint_cb(self, msg):
        idx = self._extract_index(msg.identity)
        if idx is None:
            return
        if self.last_index == idx:
            return
        self.last_index = idx
        rospy.loginfo("current waypoint index=%d (identity=%s)", idx, msg.identity)
        if idx == self.index_small:
            self.target_radius = self.radius_small
        elif idx == self.index_large:
            self.target_radius = self.radius_large

    def _timer_cb(self, _event):
        if self.target_radius is None:
            return
        if self._apply_radius(self.target_radius):
            self.target_radius = None


def main():
    rospy.init_node("waypoint_inflation_switcher", anonymous=False)
    WaypointInflationSwitcher()
    rospy.spin()


if __name__ == "__main__":
    main()
