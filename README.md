# identify_faces

## 概要
このプロジェクトは、InsightFace（ArcFace）とOpenCVを使用したリアルタイム顔認識システムです。  
カメラからの顔検出・認識、学習データ管理、GUI操作によるデータ整理機能を提供します。

## 主な機能

- **リアルタイム顔検出・認識**: カメラからの映像で人物を自動識別
- **自動学習データ更新**: 検出した顔を自動的に学習データに追加
- **GUI管理ツール**: 検出された顔画像の整理・統合・削除
- **人物名前登録**: IDに対応する名前の管理
- **学習データ再作成**: 既存画像から学習データを再構築

## 必要な環境・ライブラリ

- Python 3.8 以上
- OpenCV (`opencv-python`)
- insightface
- onnxruntime
- numpy
- tkinter (GUI用)
- Pillow (日本語表示用)

### インストール手順

#### 1. 仮想環境の作成・アクティベート

```sh
# Conda環境の場合
conda create -n identify_faces python=3.8
conda activate identify_faces

# venv環境の場合
python -m venv identify_faces
# Windows
identify_faces\Scripts\activate
# Linux/Mac
source identify_faces/bin/activate
```

#### 2. 必要ライブラリのインストール

```sh
# 基本ライブラリ
pip install opencv-python numpy insightface onnxruntime

# GUI・画像処理用
pip install pillow

# または一括インストール
pip install opencv-python numpy insightface onnxruntime pillow
```

#### 3. InsightFaceモデルの準備
初回実行時に`buffalo_l`モデルが自動的にダウンロードされます（約600MB）。

### 推奨システム要件

- CPU: Intel Core i5以上
- RAM: 8GB以上
- ストレージ: 2GB以上の空き容量（モデル・データ保存用）
- Webカメラ: 顔認識用

## ファイル構成

### メインプログラム
- `det_on_camera/det_on_camera.py` : カメラ顔検出メインスクリプト
- `det_on_camera/person_detection_system.py` : 顔検出・認識システム
- `det_on_camera/sort_out_person.py` : GUI管理ツール

### 従来の動画処理（レガシー）
- `identify_arcface.py` : 顔特徴抽出・識別メインスクリプト
- `identify_arcface_cv2tracker.py` : CV2トラッカー使用版

### データフォルダ
- `det_on_camera/training_data/` : 学習データ（.npyファイル）
- `det_on_camera/detected_faces/` : 検出された顔画像
- `det_on_camera/detected_faces/del_img/` : 削除済み画像
- `movie/` : サンプル動画ファイル

## 使い方

### 1. カメラを使った顔認識システム

```sh
# メインシステムの起動
cd det_on_camera
python det_on_camera.py
```

**操作方法:**
- カメラ映像で顔を自動検出・認識
- 新しい人物は自動的に新規IDを採番
- `q`キーで終了

### 2. GUI管理ツール

```sh
# データ管理ツールの起動
cd det_on_camera
python sort_out_person.py
```

**機能:**
- 検出された顔画像の一覧表示
- 人物の統合（複数IDを1つに統合）
- 不要な画像・IDの削除
- 名前の登録・管理
- 学習データの再作成

### 3. 従来の動画処理（レガシー機能）

```sh
# 動画から顔識別
python identify_arcface.py
```

## 詳細な使用手順

### 初回セットアップ
1. 仮想環境の作成・アクティベート
2. 必要ライブラリのインストール
3. `det_on_camera.py`を実行してInsightFaceモデルをダウンロード

### 日常的な使用
1. `det_on_camera.py`でカメラ検出を実行
2. 定期的に`sort_out_person.py`でデータを整理
3. 必要に応じて人物名を登録

## システム設定

### カメラ設定
- デフォルト: カメラインデックス 0
- 別のカメラを使用する場合は`det_on_camera.py`内の`camera_index`を変更

### 顔認識設定
- 類似度閾値: 0.5（デフォルト）
- バウンディングボックスマージン: 25%（デフォルト）
- 検出サイズ: 640x640

### データ管理
- 学習データ: `training_data/person_{ID}.npy`
- 人物情報: `training_data/person_info.json`
- 顔画像: `detected_faces/{ID}_{日時}.jpg`

## トラブルシューティング

### よくある問題

#### 1. カメラが開けない
```
カメラ(index: 0)を開けませんでした
```
**解決策:**
- 他のアプリケーションがカメラを使用していないか確認
- カメラインデックスを1, 2などに変更
- カメラドライバーの再インストール

#### 2. InsightFaceモデルのダウンロード失敗
**解決策:**
- インターネット接続を確認
- ファイアウォール設定を確認
- 手動でモデルをダウンロード

#### 3. 日本語が文字化けする
**解決策:**
- Pillowライブラリがインストールされているか確認
- Windowsフォントが正常に読み込まれているか確認

#### 4. メモリ不足エラー
**解決策:**
- 学習データが大きすぎる場合は古いデータを削除
- RAMを増設
- バッチサイズを小さくする

### ログの確認
- システムのログはコンソールに出力されます
- エラーの詳細はログメッセージを確認してください

## コード概要

### PersonDetectionSystem クラス
- `load_training_data()`: 学習データの読み込み
- `find_matching_person()`: 人物マッチング
- `save_detected_face()`: 検出顔画像の保存
- `update_training_data()`: 学習データの更新
- `create_new_person()`: 新規人物の登録

### PersonSortOutApp クラス
- GUI による画像管理機能
- 人物の統合・削除機能
- 名前登録機能
- 学習データ再作成機能

## 注意事項

- GPU利用の場合は`providers=['CUDAExecutionProvider']`に変更してください
- カメラ使用時はプライバシーに注意してください
- 大量の学習データは定期的に整理することを推奨します
- バックアップは定期的に取得してください

## ライセンス

このプロジェクトは個人使用・学習目的で作成されています。
商用利用の場合は、使用しているライブラリのライセンスを確認してください。
