import cv2
from insightface.app import FaceAnalysis
import numpy as np

# --- 動画ファイルのパス定義 ---
VIDEO_PATH_LEARN = 'movie/boy03.mp4'
VIDEO_PATH_IDENTIFY = 'movie/movie01.mp4'
# --- フレームスキップ数の定義 ---
SKIP_FRAMES_LEARN = 3  # 学習用動画でスキップするフレーム数
SKIP_FRAMES_IDENTIFY = 3  # 識別用動画でスキップするフレーム数

# --- InsightFaceの初期化（ArcFace+ResNet） ---
#app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- (1) 学習用動画から顔特徴ベクトル抽出 ---

def extract_features(video_path, skip_frames=0):
    video = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue
        # 顔検出
        faces = app.get(frame)
        # 顔検出
        faces = app.get(frame)
        # 顔枠描画
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
            # 顔特徴ベクトル抽出
            features.append(face.embedding)
        # ウィンドウサイズを1366x768にリサイズ
        disp_frame = cv2.resize(frame, (1366, 768))
        cv2.imshow('Learn (boy01)', disp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if features:
        # 平均特徴ベクトル（複数フレームの場合）
        return np.mean(features, axis=0)
    else:
        return None

# --- (2)-(3) boy01.mp4から特徴ベクトル抽出 ---
learned_feature = extract_features(VIDEO_PATH_LEARN, skip_frames=SKIP_FRAMES_LEARN)
if learned_feature is None:
    print('学習用動画から顔が検出できませんでした。')
    exit()

# --- (4)-(5) movie01.mp4で同一人物認識 ---
def identify_person(video_path, learned_feature, threshold=0.6, skip_frames=0):
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue
        faces = app.get(frame)
        for face in faces:
            # コサイン類似度で判定
            similarity = np.dot(learned_feature, face.embedding) / (np.linalg.norm(learned_feature) * np.linalg.norm(face.embedding))
            if similarity > threshold:
                # 顔枠描画
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                cv2.putText(frame, f"Match: {similarity:.2f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        # ウィンドウサイズを1366x768にリサイズ
        disp_frame = cv2.resize(frame, (1366, 768))
        cv2.imshow('Identify', disp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    video.release()
    cv2.destroyAllWindows()

identify_person(VIDEO_PATH_IDENTIFY, learned_feature, skip_frames=SKIP_FRAMES_IDENTIFY)
