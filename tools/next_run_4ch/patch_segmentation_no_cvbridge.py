#!/usr/bin/env python3
"""
Patch segmentation_colorserver_real.py to avoid cv_bridge inside conda Python.
Reason: conda libffi and ROS cv_bridge can conflict with:
  undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
"""
from pathlib import Path

TARGET = Path('/home/orne_beta/shoku_ws/src/nav_cloning/scripts/segmentation_colorserver_real.py')

helper = r'''
def numpy_to_imgmsg_no_cvbridge(arr, encoding="mono8", stamp=None, frame_id=""):
    import numpy as np
    from sensor_msgs.msg import Image
    arr = np.asarray(arr)
    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    enc = encoding.lower()
    if enc in ("mono8", "8uc1"):
        arr = arr.astype(np.uint8, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"mono8 image must be HxW, got shape={arr.shape}")
        msg.step = int(msg.width)
        msg.data = arr.tobytes()
        return msg
    if enc in ("mono16", "16uc1"):
        arr = arr.astype(np.uint16, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"mono16 image must be HxW, got shape={arr.shape}")
        msg.step = int(msg.width * 2)
        msg.data = arr.tobytes()
        return msg
    if enc in ("bgr8", "rgb8"):
        arr = arr.astype(np.uint8, copy=False)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{encoding} image must be HxWx3, got shape={arr.shape}")
        msg.step = int(msg.width * 3)
        msg.data = arr.tobytes()
        return msg
    raise ValueError(f"Unsupported encoding without cv_bridge: {encoding}")

def imgmsg_to_numpy_no_cvbridge(msg, desired_encoding="bgr8"):
    import numpy as np
    enc = msg.encoding.lower()
    h = int(msg.height)
    w = int(msg.width)
    if enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr.reshape((h, int(msg.step)))
        arr = arr[:, :w * 3].reshape((h, w, 3))
        if desired_encoding == enc:
            return arr.copy()
        if desired_encoding == "bgr8" and enc == "rgb8":
            return arr[:, :, ::-1].copy()
        if desired_encoding == "rgb8" and enc == "bgr8":
            return arr[:, :, ::-1].copy()
        return arr.copy()
    if enc in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr.reshape((h, int(msg.step)))
        return arr[:, :w].copy()
    raise ValueError(f"Unsupported image encoding without cv_bridge: {msg.encoding}")
'''

def main():
    p = TARGET
    if not p.exists():
        raise FileNotFoundError(p)
    s = p.read_text()
    if 'def imgmsg_to_numpy_no_cvbridge' not in s:
        marker = 'import rospy\n'
        if marker in s:
            s = s.replace(marker, marker + helper + '\n', 1)
        else:
            s = helper + '\n' + s
    replacements = {
        'self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")': 'imgmsg_to_numpy_no_cvbridge(msg, desired_encoding="bgr8")',
        'self.bridge.imgmsg_to_cv2(msg, "bgr8")': 'imgmsg_to_numpy_no_cvbridge(msg, "bgr8")',
        'self.bridge.imgmsg_to_cv2(data, "bgr8")': 'imgmsg_to_numpy_no_cvbridge(data, "bgr8")',
        'self.bridge.cv2_to_imgmsg(mask_small, encoding="mono8")': 'numpy_to_imgmsg_no_cvbridge(mask_small, encoding="mono8", stamp=common_stamp)',
        'self.bridge.cv2_to_imgmsg(pred.astype(np.uint16), encoding="mono16")': 'numpy_to_imgmsg_no_cvbridge(pred.astype(np.uint16), encoding="mono16", stamp=common_stamp)',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    p.write_text(s)
    print(f'patched: {p}')
    print('Check with: grep -n "cv2_to_imgmsg\\|imgmsg_to_cv2"', p)

if __name__ == '__main__':
    main()
