# identify_faces

## 概要
このプロジェクトは、動画から顔特徴ベクトルを抽出し、別の動画で同一人物を識別するPythonスクリプトです。  
`insightface`（ArcFace+ResNet）とOpenCVを利用しています。

## 必要な環境・ライブラリ

- Python 3.8 以上
- OpenCV (`opencv-python`)
- insightface
- onnxruntime
- numpy

### インストール例（conda環境の場合）

```sh
conda create -n identify_faces python=3.8
conda activate identify_faces
conda install -c conda-forge opencv numpy insightface onnxruntime
```

またはpipの場合

```sh
pip install opencv-python numpy insightface onnxruntime
```

## ファイル構成

- `identify_arcface.py` : 顔特徴抽出・識別メインスクリプト
- `movie/boy01.mp4` : 学習用動画
- `movie/movie01.mp4` : 識別対象動画

## 使い方

1. `movie`フォルダに学習用動画(`boy01.mp4`)と識別対象動画(`movie01.mp4`)を配置してください。
2. ターミナルでPython環境をアクティブにします。
3. スクリプトを実行します。

```sh
python identify_arcface.py
```

- `boy01.mp4`から顔特徴ベクトルを抽出し、`movie01.mp4`で同一人物を識別します。
- 顔枠と一致度（コサイン類似度）が表示されます。
- ウィンドウで`q`キーを押すと終了します。

## コード概要

- `extract_features(video_path)`  
  学習用動画から顔特徴ベクトルを抽出し、平均ベクトルを返します。
- `identify_person(video_path, learned_feature, threshold=0.6)`  
  識別対象動画で顔を検出し、学習済み特徴ベクトルとのコサイン類似度で同一人物か判定します。

## 注意事項

- 動画ファイルパスは必要に応じて変更してください。
- GPU利用の場合は`providers`引数を変更してください（例: `providers=['CUDAExecutionProvider']`）。
