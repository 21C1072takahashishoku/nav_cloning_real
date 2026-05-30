#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Illuminance
import serial
import time
import csv
from datetime import datetime

# シリアルポートの設定
PORT = '/dev/ttyACM2'  # ポート名を確認済み
BAUD_RATE = 38400
TIMEOUT = 10  # タイムアウトを10秒に延長

# シリアルポートを開く
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
except serial.SerialException as e:
    rospy.logerr("Error opening serial port: %s", e)
    exit(1)

def send_command(command, retry=3):
    """コマンドをデバイスに送信し、応答を取得する"""
    for _ in range(retry):
        ser.write((command + '\r\n').encode())
        time.sleep(1)  # コマンド送信後に1秒待つ
        response = ser.read_all().decode().strip()
        if response:
            rospy.loginfo("Sent: %s, Received: %s", command, response)
            return response
        else:
            rospy.logwarn("Sent: %s, No response received, retrying...", command)
            time.sleep(2)  # 再試行前に2秒待つ
    return None

if __name__ == '__main__':
    # ROSノードの初期化
    rospy.init_node('lux_sensor_node', anonymous=True)
    # sensor_msgs/Illuminance 型のパブリッシャーを作成（トピック名は "lux" とする）
    pub = rospy.Publisher('lux', Illuminance, queue_size=10)
    rate = rospy.Rate(1)  # 1Hzでパブリッシュ

    # デバイスの初期化
    rospy.loginfo("Initializing device...")
    init_response = send_command(':SYST:INIT')
    if not init_response:
        rospy.logerr("Failed to initialize device.")
        ser.close()
        exit(1)
    time.sleep(3)  # 初期化後に3秒待つ

    # デバイスモデルの確認
    model = send_command('QPID')
    rospy.loginfo("Device Model: %s", model)

    # ゼロ調整
    zero_adjustment = send_command(':0ADJUST')
    if zero_adjustment == "CAP ERR":
        rospy.logerr("Zero Adjustment failed: Ensure the sensor cap is properly attached.")
    elif zero_adjustment == "OK":
        rospy.loginfo("Zero Adjustment successful.")
    else:
        rospy.loginfo("Zero Adjustment response: %s", zero_adjustment)

    # キャップを外して再度測定
    input("Remove the sensor cap and press Enter to continue...")

    # 日付時刻を付与したログファイル名を準備
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f'lux_log_{timestamp_str}.csv'
    # CSVファイルのヘッダーを書き込み
    with open(logfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Lux'])

    rospy.loginfo("Starting to monitor lux values...")

    # 継続モニタリングループ
    while not rospy.is_shutdown():
        lux_value_str = send_command(':MEAS?')
        if lux_value_str:
            try:
                # 文字列から数値（float）に変換
                lux_value = float(lux_value_str)
            except ValueError:
                rospy.logwarn("Received invalid lux value: %s", lux_value_str)
                rate.sleep()
                continue

            # CSVログへの記録
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(logfile, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, lux_value])
            rospy.loginfo("Logged at %s: %f", timestamp, lux_value)

            # ROSメッセージの作成とパブリッシュ
            msg = Illuminance()
            msg.header.stamp = rospy.Time.now()
            msg.illuminance = lux_value
            msg.variance = 0.0  # 分散が不明な場合は0に設定
            pub.publish(msg)
        else:
            rospy.logwarn("Failed to get Lux Value.")

        rate.sleep()

    # ループ終了後、シリアルポートを閉じる
    ser.close()


