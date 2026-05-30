# nav_cloning_real

視覚と行動のend-to-end学習による経路追従行動のオンライン模倣（実機用）．  
LiDARベースのナビゲーション行動を，カメラ画像のみを入力とする行動にオンラインで模倣学習する．  
4チャンネル（4ch）版では Mask2Former による地面セグメンテーションマスクを4ch目に追加し，未知障害物の回避を実現する．

---

## 目次

- [システム概要](#システム概要)
- [動作環境](#動作環境)
- [インストール](#インストール)
- [パッケージ構成](#パッケージ構成)
- [データ保存構造](#データ保存構造)
- [実機走行手順](#実機走行手順)
- [主要パラメータ](#主要パラメータ)
- [トラブルシューティング](#トラブルシューティング)

---

## システム概要

```
実機カメラ（中央 / 左 / 右）
        │  /camera_*/usb_cam/image_raw
        ▼
segmentation_colorserver_real.py
        │  Mask2Former → 48×64 mono8 地面マスク
        │  /segmentation/ground_mask_*
        ▼
nav_cloning_4ch_node_pytorch_online_mask.py
        │  BGR(3ch) + 地面マスク(1ch) → 4ch入力(48×64×4)
        │  オンライン模倣学習 → 推論
        ▼
/icart_mini/cmd_vel（角速度指令）
```

---

## 動作環境

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 20.04 |
| ROS | Noetic |
| Python | 3.8以上 |
| PyTorch | 1.x以上 |
| GPU | NVIDIA（VRAM 8GB以上推奨） |
| ロボット | orne-gamma（icart_mini） |
| カメラ | USBカメラ×3（中央 / 左 / 右） |

---

## インストール

### 1. ワークスペースの作成とクローン

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
sudo apt install python3-vcstool
vcs import < nav_cloning_real/nav_cloning.install
```

### 2. ビルド

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

> `catkin_make` が依存関係で止まっても，Pythonノード単体の動作確認は可能．[トラブルシューティング](#トラブルシューティング)を参照．

### 3. Python依存パッケージ

```bash
pip install torch torchvision scikit-image opencv-python
```

### 4. Mask2Former（MMSegmentation）のセットアップ

```bash
pyenv install 3.10.x
pyenv virtualenv 3.10.x mmseg310
pyenv activate mmseg310
pip install mmengine mmcv mmsegmentation
```

> モデルweightsは `segmentation_colorserver_real.py` の設定に従って配置すること．

### 5. インストール確認

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
rospack find nav_cloning
# 期待結果：~/catkin_ws/src/nav_cloning_real のパスが返る
```

---

## パッケージ構成

```
nav_cloning_real/
├── nav_cloning.install                          # vcstool依存リスト
├── launch/
│   ├── camera_bringup.launch                   # USBカメラ起動
│   ├── nav_cloning_4ch_only.launch             # 確認用（走行なし）
│   └── nav_cloning_4ch_all.launch              # 実機走行用
├── scripts/
│   ├── segmentation_colorserver_real.py        # Mask2Former地面マスク配信
│   ├── start_mmseg.sh                          # Mask2Former環境起動
│   ├── perf_monitor_4ch.py                     # CPU/GPU/Hz性能記録
│   └── pytorch/
│       └── nav_cloning_4ch_node_pytorch_online_mask.py  # 4ch模倣学習ノード
├── experiments/
│   └── experiment_4ch_mask2former_offline_collect.sh   # 実験実行スクリプト
└── tools/
    ├── check_4ch_dataset_shape.py              # .npyのshape確認
    ├── make_ground_mask_mmseg.py               # オフラインセグメンテーション
    └── monitor_4ch_topics.sh                   # トピック・GPU確認
```

---

## データ保存構造

全データは `data/` 以下に保存される（`.gitignore` でGit管理対象外）．

```
data/
├── result_<mode>/<タイムスタンプ>/
│   └── training.csv              # ステップ・フェーズ・距離のログ
├── model_<mode>/<タイムスタンプ>/
│   └── model_gpu.pt              # 学習済みモデルweights
└── <DATASET_ID>/dataset/
    ├── img/
    │   ├── 0_center.npy          # 48×64×4 float32（BGR＋マスク）
    │   ├── 0_left.npy
    │   └── 0_right.npy
    ├── raw_img/
    │   ├── center/
    │   │   ├── 0_center.png      # 480×640×3 uint8 BGR（生カメラ画像）
    │   │   └── 0_center_mask.png # 地面マスク画像
    │   ├── left/
    │   └── right/
    └── vel/
        └── data.csv              # 操舵角・pose・距離・フェーズのラベル
```

---

## 実機走行手順

### 事前確認

走行前に以下をすべて確認すること．

| 確認項目 | コマンド | 合格条件 |
|---|---|---|
| パッケージ認識 | `rospack find nav_cloning` | 正しいパスが返る |
| カメラトピック | `rostopic hz /camera_*/usb_cam/image_raw` | 各1Hz以上・encoding=`bgr8` |
| 地面マスク配信 | `rostopic hz /segmentation/ground_mask_center` | 1Hz以上 |
| マスク形式 | `rostopic echo -n 1 /segmentation/ground_mask_center/encoding` | `mono8`・height=48・width=64 |
| GPUメモリ | `nvidia-smi` | メモリ枯渇なし |
| 4ch入力shape | `python3 tools/check_4ch_dataset_shape.py <パス>` | `(48, 64, 4) float32` |

---

### Step 0：ロボットの電源を入れる

電源スイッチを OFF → ON にする．

---

### Step 1：環境設定（全端末共通）

ターミナルを5つ開き，それぞれで実行する：

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$HOME/catkin_ws/src
```

---

### Step 2：ロボット本体の起動【端末1】

```bash
roslaunch orne_bringup orne_gamma.launch
```

---

### Step 3：カメラの起動【端末2】

```bash
roslaunch nav_cloning camera_bringup.launch
```

> カメラが認識されない場合は `launch/camera_bringup.launch` でカメラIDを修正し，再起動する．

---

### Step 4：4ch実験プログラムの起動【端末3】

> **`conda activate` しないこと**（ROS環境と競合する）．

```bash
cd ~/catkin_ws/src/nav_cloning/experiments
./experiment_4ch_mask2former_offline_collect.sh <DATASET_ID>

# 例（本番）：
./experiment_4ch_mask2former_offline_collect.sh real_4ch_train5000_test2000_20260528_03

# 例（短時間テスト）：
TRAIN_STEPS=300 TEST_STEPS=100 ./experiment_4ch_mask2former_offline_collect.sh real_4ch_short_test
```

---

### Step 5：トピック確認【端末4】

```bash
rostopic hz /segmentation/ground_mask_center
rostopic hz /nav_vel
rostopic echo -n 5 /icart_mini/cmd_vel
```

---

### Step 6：性能記録の起動【端末5】

```bash
rosrun nav_cloning perf_monitor_4ch.py \
  _dataset_id:=<DATASET_ID> \
  _log_rate_hz:=1.0
```

---

### Step 7：RVizで自己位置合わせ → 緊急停止解除

1. RVizで **2D Pose Estimate** により初期位置を設定する
2. 自己位置推定が正しいことを確認する
3. 緊急停止を解除する → 経路追従開始

> **初回走行時** は `cmd_vel_linear_fixed=0.05` m/s（通常の1/4速）で行うこと．

---

## 主要パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `train_steps` | `10000` | 模倣学習ステップ数 |
| `test_steps` | `3000` | 推論のみのステップ数 |
| `dataset_id` | タイムスタンプ | データセット識別子 |
| `dataset_stride` | `1` | Nステップごとに保存（短時間テストは`5`推奨） |
| `save_raw_images` | `true` | 480×640 PNG画像を保存する |
| `cmd_vel_linear_fixed` | `0.2` | 前進速度 [m/s] |
| `max_mask_age_sec` | `0.5` | マスクの最大許容遅延 [秒] |
| `rate_hz`（Mask2Former） | `2.0` | マスク推論レート [Hz] |

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `rospack find` が失敗 | `ROS_PACKAGE_PATH` 未設定 | Step 1の環境設定コマンドを再実行 |
| `catkin_make` が止まる | 既存パッケージの依存問題 | `python3 -m py_compile` で構文確認のみ実施 |
| ground_maskトピックが出ない | カメラトピック名の不一致 | `rostopic list \| grep camera` で実名確認しlaunchを修正 |
| ground_maskのHzが低い | Mask2FormerのGPU過負荷 | `rate_hz` を下げる・`nvidia-smi` で負荷確認 |
| 4ch shapeエラー | HWC/CHW混同 | 入力は必ず `HWC=(48,64,4)` |
| 3chモデルロードで落ちる | `conv1` 形状不一致 | 3chモデルの流用不可．4ch用に新規学習する |
| ロボットが危険な挙動 | マスク遅延・誤分類 | 即座に緊急停止．速度を下げマスク可視化を確認 |
| ROS起動後にエラー | conda環境の競合 | 4chノードのターミナルでは `conda activate` しない |

---

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照．
