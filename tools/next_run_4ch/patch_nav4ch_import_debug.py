#!/usr/bin/env python3
"""
Patch nav_cloning_4ch_node_pytorch_online_mask.py after moving it to scripts/.
Adds scripts/pytorch to sys.path and injects save-debug logs.
"""
from pathlib import Path

TARGET = Path('/home/orne_beta/shoku_ws/src/nav_cloning/scripts/nav_cloning_4ch_node_pytorch_online_mask.py')

IMPORT_PATCH = '''import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_DIR = os.path.join(SCRIPT_DIR, "pytorch")
if PYTORCH_DIR not in sys.path:
    sys.path.insert(0, PYTORCH_DIR)

from nav_cloning_pytorch import *'''

OLD_IMPORT = 'from nav_cloning_pytorch import *'

SAVE_TARGET = '''        if not self.save_offline_dataset:
            return

        if self.dataset_stride <= 0:
            return

        if (self.episode % self.dataset_stride) != 0:
            return
'''

SAVE_REPL = '''        rospy.loginfo_throttle(
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
            rospy.logwarn_throttle(1.0, f"[4ch save skip] episode={self.episode}, stride={self.dataset_stride}")
            return
'''

DO_TARGET = '''        # ===== 1. resized images for existing offline training =====
        np.save(os.path.join(self.img_store_path, f"{ep}_center.npy"), img)
'''

DO_REPL = '''        rospy.loginfo_throttle(1.0, f"[4ch save DO] ep={ep}, center_shape={img.shape}, save_to={self.img_store_path}")

        # ===== 1. resized images for existing offline training =====
        np.save(os.path.join(self.img_store_path, f"{ep}_center.npy"), img)
'''

def main():
    p = TARGET
    if not p.exists():
        raise FileNotFoundError(p)
    s = p.read_text()
    if OLD_IMPORT in s and 'PYTORCH_DIR = os.path.join(SCRIPT_DIR, "pytorch")' not in s:
        s = s.replace(OLD_IMPORT, IMPORT_PATCH, 1)
    if SAVE_TARGET in s and '[4ch save check]' not in s:
        s = s.replace(SAVE_TARGET, SAVE_REPL, 1)
    if DO_TARGET in s and '[4ch save DO]' not in s:
        s = s.replace(DO_TARGET, DO_REPL, 1)
    p.write_text(s)
    print(f'patched: {p}')

if __name__ == '__main__':
    main()
