import cv2
from insightface.app import FaceAnalysis
import numpy as np
from collections import defaultdict
import math

# --- 機能フラグ定義 ---
ENABLE_FACE_DIRECTION = True  # 顔の向き認識機能を有効にするかどうか
USE_3D_BOX_VISUALIZATION = False  # 3Dボックス表示を使用するかどうか（Falseの場合は3D矢印を使用）
ENABLE_LANDMARK_DISPLAY = True  # 顔のランドマーク（目、鼻、口）表示を有効にするかどうか

# --- 動画ファイルのパス定義 ---
#VIDEO_PATH_LEARN = 'movie/boy03.mp4'
#VIDEO_PATH_IDENTIFY = 'movie/movie01.mp4'

#VIDEO_PATH_LEARN = 'movie/movie01.mp4'
#VIDEO_PATH_IDENTIFY = 'movie/boy03.mp4'

VIDEO_PATH_LEARN = 'movie/movie01.mp4'
VIDEO_PATH_IDENTIFY = 'movie/movie02.mp4'


# --- フレームスキップ数の定義 ---
SKIP_FRAMES_LEARN = 5  # 学習用動画でスキップするフレーム数
SKIP_FRAMES_IDENTIFY = 5  # 識別用動画でスキップするフレーム数
# --- フレーム数上限 ---
FRAMES_COUNT_LEARN = 30  # 学習用動画で処理するフレーム数上限
# --- 類似度閾値 ---
SIMILARITY_THRESHOLD = 0.5  # 同一人物判定の閾値
TRACKER_THRESHOLD = 0.7  # トラッカー用の高い閾値

# --- InsightFaceの初期化（ArcFace+ResNet） ---
#app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- 顔の向き認識関数 ---
def get_face_pose_description(pose):
    """顔の向きを英語で説明"""
    yaw, pitch, roll = pose
    
    # ヨー角（左右の向き）
    if yaw > 30:
        horizontal = "Right"
    elif yaw < -30:
        horizontal = "Left"
    else:
        horizontal = "Front"
    
    # ピッチ角（上下の向き）
    if pitch > 20:
        vertical = "Up"
    elif pitch < -20:
        vertical = "Down"
    else:
        vertical = ""
    
    # ロール角（傾き）
    if abs(roll) > 15:
        tilt = f"Tilt{roll:.0f}deg"
    else:
        tilt = ""
    
    # 組み合わせて説明文を作成
    description = horizontal
    if vertical:
        description += f"-{vertical}"
    if tilt:
        description += f"-{tilt}"
    
    return description

def draw_face_direction_arrow(frame, bbox, pose, face=None):
    """顔の向きを3D矢印で表示（ランドマークを意識した位置から描画）"""
    yaw, pitch, roll = pose
    
    # ランドマークが利用可能な場合は、ランドマークから顔の中心を計算
    if face is not None and hasattr(face, 'kps') and face.kps is not None:
        landmarks = face.kps
        if len(landmarks) >= 3:
            # 両目と鼻から顔の中心を計算
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # 目の中点と鼻の中点を計算して、より正確な顔の中心を求める
            eye_center = (left_eye + right_eye) / 2
            face_center = (eye_center + nose) / 2
            center_x = int(face_center[0])
            center_y = int(face_center[1])
        else:
            # ランドマークが不十分な場合はバウンディングボックスの中心を使用
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
    else:
        # ランドマークが利用できない場合はバウンディングボックスの中心を使用
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
    
    # 矢印の長さ（顔のサイズに応じて調整）
    face_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    arrow_length = max(40, face_size // 2)
    
    # 3D方向ベクトルを計算（オイラー角から方向ベクトルへ変換）
    yaw_rad = math.radians(-yaw)    # 左右の回転
    pitch_rad = math.radians(-pitch)  # 上下の回転
    
    # 3D方向ベクトル（Z軸が前方向）
    direction_x = math.sin(yaw_rad) * math.cos(pitch_rad)
    direction_y = math.sin(pitch_rad)
    direction_z = math.cos(yaw_rad) * math.cos(pitch_rad)
    
    # 2D投影（パースペクティブ効果を追加）
    # Z成分が正の場合（前向き）は矢印を長く、負の場合（後ろ向き）は短く
    perspective_factor = max(0.3, 1.0 + direction_z * 0.5)
    
    end_x = int(center_x + arrow_length * direction_x * perspective_factor)
    end_y = int(center_y - arrow_length * direction_y * perspective_factor)  # Y軸は反転
    
    # 矢印の色を方向によって変更
    if direction_z > 0.5:  # 前向き
        arrow_color = (0, 255, 255)  # 黄色
    elif direction_z > 0:  # やや前向き
        arrow_color = (0, 255, 0)    # 緑色
    elif direction_z > -0.5:  # やや後ろ向き
        arrow_color = (0, 165, 255)  # オレンジ色
    else:  # 後ろ向き
        arrow_color = (0, 0, 255)    # 赤色
    
    # 矢印の太さも距離感で調整
    arrow_thickness = max(2, int(3 * perspective_factor))
    
    # メイン矢印を描画
    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                    arrow_color, arrow_thickness, tipLength=0.3)
    
    # 顔の中心点をマーキング（ランドマークベースの場合は特別な色で）
    if face is not None and hasattr(face, 'kps') and face.kps is not None:
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)  # シアン色
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 0), 1)
    else:
        cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)  # 白色
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 0), 1)
    
    # 3D効果のための補助線（ロール角を表現）
    if abs(roll) > 10:  # 顔の傾きが大きい場合
        roll_rad = math.radians(roll)
        roll_length = 30
        
        # ロール線の端点を計算
        roll_end_x = int(center_x + roll_length * math.cos(roll_rad))
        roll_end_y = int(center_y + roll_length * math.sin(roll_rad))
        
        # ロール表示線（点線風）
        cv2.line(frame, (center_x, center_y), (roll_end_x, roll_end_y), 
                 (255, 0, 255), 2)  # マゼンタ色
    
    # 3D座標系の参考線（デバッグ用 - オプション）
    if ENABLE_FACE_DIRECTION:
        # 小さな3D座標軸を表示
        axis_length = 25
        
        # X軸（赤）- 右方向
        x_end = int(center_x + axis_length)
        cv2.line(frame, (center_x, center_y), (x_end, center_y), (0, 0, 255), 1)
        
        # Y軸（緑）- 上方向
        y_end = int(center_y - axis_length)
        cv2.line(frame, (center_x, center_y), (center_x, y_end), (0, 255, 0), 1)
        
        # 3D情報をテキストで表示
        info_text = f"3D:({direction_x:.2f},{direction_y:.2f},{direction_z:.2f})"
        cv2.putText(frame, info_text, (bbox[0], bbox[3]+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def draw_3d_face_orientation_box(frame, bbox, pose):
    """顔の向きを3Dボックスで表現（より高度な3D表現）"""
    yaw, pitch, roll = pose
    
    # バウンディングボックスの中心と基本サイズ
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    box_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 4
    
    # 3D立方体の頂点を定義（ローカル座標）
    vertices_3d = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 後面
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]   # 前面
    ]) * box_size / 2
    
    # 回転行列を作成
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    
    # 各軸の回転行列
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                   [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    
    Ry = np.array([[math.cos(yaw_rad), 0, math.sin(yaw_rad)],
                   [0, 1, 0],
                   [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]])
    
    Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0],
                   [math.sin(roll_rad), math.cos(roll_rad), 0],
                   [0, 0, 1]])
    
    # 合成回転行列
    R = Rz @ Ry @ Rx
    
    # 頂点を回転
    vertices_rotated = (R @ vertices_3d.T).T
    
    # 2D投影（簡単な正射影）
    vertices_2d = vertices_rotated[:, :2] + np.array([center_x, center_y])
    vertices_2d = vertices_2d.astype(int)
    
    # 立方体の線を描画
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 後面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 前面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 接続線
    ]
    
    for edge in edges:
        pt1 = tuple(vertices_2d[edge[0]])
        pt2 = tuple(vertices_2d[edge[1]])
        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
    
    # 前面を強調表示（半透明効果）
    front_face = [vertices_2d[4], vertices_2d[5], vertices_2d[6], vertices_2d[7]]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(front_face)], (0, 255, 255))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    return frame

def draw_face_landmarks(frame, face):
    """顔のランドマーク（目、鼻、口）を描画"""
    if not ENABLE_LANDMARK_DISPLAY:
        return frame
        
    if hasattr(face, 'kps') and face.kps is not None:
        landmarks = face.kps  # [左目, 右目, 鼻, 左口角, 右口角]
        
        # ランドマークの色定義
        colors = [
            (255, 0, 0),    # 左目 - 青
            (255, 0, 0),    # 右目 - 青
            (0, 255, 0),    # 鼻 - 緑
            (0, 0, 255),    # 左口角 - 赤
            (0, 0, 255),    # 右口角 - 赤
        ]
        
        # ランドマークのラベル
        labels = ['L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth']
        
        # 各ランドマークを描画
        for i, (landmark, color, label) in enumerate(zip(landmarks, colors, labels)):
            x, y = int(landmark[0]), int(landmark[1])
            
            # ランドマークポイントを円で描画
            cv2.circle(frame, (x, y), 3, color, -1)
            
            # ラベルを描画（小さなフォントで）
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 目と目を結ぶ線を描画
        if len(landmarks) >= 2:
            left_eye = landmarks[0].astype(int)
            right_eye = landmarks[1].astype(int)
            cv2.line(frame, tuple(left_eye), tuple(right_eye), (255, 255, 0), 1)
        
        # 口角を結ぶ線を描画
        if len(landmarks) >= 5:
            left_mouth = landmarks[3].astype(int)
            right_mouth = landmarks[4].astype(int)
            cv2.line(frame, tuple(left_mouth), tuple(right_mouth), (255, 0, 255), 1)
        
        # 鼻と目の中点を結ぶ線を描画（顔の中心軸）
        if len(landmarks) >= 3:
            nose = landmarks[2].astype(int)
            if len(landmarks) >= 2:
                eye_center = ((landmarks[0] + landmarks[1]) / 2).astype(int)
                cv2.line(frame, tuple(eye_center), tuple(nose), (0, 255, 255), 1)
    
    return frame

# --- 人物管理クラス ---
class PersonManager:
    def __init__(self):
        self.persons = {}  # person_id: {'features': [特徴ベクトルのリスト], 'tracker': tracker_object}
        self.next_id = 1
        
    def add_person(self, feature_vector, bbox):
        """新しい人物を追加"""
        person_id = self.next_id
        self.persons[person_id] = {
            'features': [feature_vector],
            'tracker': None
        }
        self.next_id += 1
        return person_id
    
    def find_matching_person(self, feature_vector, threshold=SIMILARITY_THRESHOLD):
        """既存の人物との類似度を計算して最も類似した人物IDを返す"""
        best_match_id = None
        best_similarity = 0
        
        for person_id, person_data in self.persons.items():
            # その人物の平均特徴ベクトルと比較
            avg_feature = np.mean(person_data['features'], axis=0)
            similarity = np.dot(feature_vector, avg_feature) / (
                np.linalg.norm(feature_vector) * np.linalg.norm(avg_feature)
            )
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match_id = person_id
                
        return best_match_id, best_similarity
    
    def add_feature_to_person(self, person_id, feature_vector):
        """既存の人物に新しい特徴ベクトルを追加（追加学習）"""
        if person_id in self.persons:
            self.persons[person_id]['features'].append(feature_vector)
    
    def get_person_feature(self, person_id):
        """指定された人物の平均特徴ベクトルを取得"""
        if person_id in self.persons:
            return np.mean(self.persons[person_id]['features'], axis=0)
        return None
    
    def init_tracker(self, person_id, frame, bbox):
        """指定された人物のトラッカーを初期化"""
        if person_id in self.persons:
            try:
                # OpenCV 4.12以降の新しいトラッカーAPI
                tracker = cv2.TrackerMIL_create()
                
                # バウンディングボックスを(x, y, w, h)形式に変換
                x, y, x2, y2 = bbox
                tracker_bbox = (x, y, x2-x, y2-y)
                success = tracker.init(frame, tracker_bbox)
                
                if success:
                    self.persons[person_id]['tracker'] = tracker
                    print(f"Initialized tracker for ID {person_id}")
                else:
                    print(f"Warning: Failed to initialize tracker for ID {person_id}")
                    self.persons[person_id]['tracker'] = None
                    
            except Exception as e:
                print(f"Warning: Error occurred during tracker initialization: {e}")
                self.persons[person_id]['tracker'] = None
    
    def update_tracker(self, person_id, frame):
        """指定された人物のトラッカーを更新"""
        if person_id in self.persons and self.persons[person_id]['tracker'] is not None:
            success, bbox = self.persons[person_id]['tracker'].update(frame)
            if success:
                return bbox
        return None

# --- (1) 学習用動画から顔特徴ベクトル抽出（複数人対応） ---

def extract_features_multipeople(video_path, skip_frames=0):
    """複数人の顔を学習し、トラッカーで同一人物を管理"""
    video = cv2.VideoCapture(video_path)
    person_manager = PersonManager()
    frame_count = 0
    learn_count = 0
    
    print("Starting learning phase...")
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue
            
        if learn_count >= FRAMES_COUNT_LEARN:
            break

        # 顔検出
        faces = app.get(frame)
        
        # 検出された各顔について処理
        for face in faces:
            feature_vector = face.embedding
            bbox = face.bbox.astype(int)
            
            # 顔の向き情報を取得（フラグが有効な場合のみ）
            pose = None
            pose_description = "Unknown"
            
            if ENABLE_FACE_DIRECTION and hasattr(face, 'pose') and face.pose is not None:
                pose = face.pose
                pose_description = get_face_pose_description(pose)
                print(f"Face direction: {pose_description} (Yaw:{pose[0]:.1f}deg, Pitch:{pose[1]:.1f}deg, Roll:{pose[2]:.1f}deg)")
            
            # 既存の人物との類似度をチェック
            matched_id, similarity = person_manager.find_matching_person(
                feature_vector, threshold=TRACKER_THRESHOLD
            )
            
            if matched_id is not None:
                # 既存の人物に追加学習
                person_manager.add_feature_to_person(matched_id, feature_vector)
                print(f"Added features to ID {matched_id} (similarity: {similarity:.3f})")
            else:
                # 新しい人物として登録
                new_id = person_manager.add_person(feature_vector, bbox)
                person_manager.init_tracker(new_id, frame, bbox)
                print(f"Registered new person ID {new_id}")
            
            # バウンディングボックスとIDを描画
            current_id = matched_id if matched_id is not None else person_manager.next_id - 1
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {current_id}", (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 顔のランドマークを描画
            frame = draw_face_landmarks(frame, face)
            
            # 顔の向き情報を表示（フラグが有効な場合のみ）
            if ENABLE_FACE_DIRECTION:
                cv2.putText(frame, pose_description, (bbox[0], bbox[3]+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 顔の向きを3D表現で表示
                if pose is not None:
                    if USE_3D_BOX_VISUALIZATION:
                        frame = draw_3d_face_orientation_box(frame, bbox, pose)
                    else:
                        frame = draw_face_direction_arrow(frame, bbox, pose, face)
        
        # トラッカーの更新（検出されなかった人物の追跡）
        for person_id in list(person_manager.persons.keys()):
            tracked_bbox = person_manager.update_tracker(person_id, frame)
            if tracked_bbox is not None:
                # トラッカーで追跡中の人物の枠を描画（異なる色）
                x, y, w, h = [int(v) for v in tracked_bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, f"Track ID: {person_id}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ウィンドウサイズを1366x768にリサイズ
        disp_frame = cv2.resize(frame, (1366, 768))
        if ENABLE_FACE_DIRECTION:
            if USE_3D_BOX_VISUALIZATION:
                window_title = 'Learn (Multiple People with 3D Box Direction)'
            else:
                window_title = 'Learn (Multiple People with 3D Arrow Direction)'
        else:
            window_title = 'Learn (Multiple People)'
        cv2.imshow(window_title, disp_frame)

        learn_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video.release()
    cv2.destroyAllWindows()
    
    print(f"Learning completed: {len(person_manager.persons)} people learned")
    for person_id, person_data in person_manager.persons.items():
        print(f"  ID {person_id}: {len(person_data['features'])} feature vectors")
    
    return person_manager

# --- (2) 識別用動画での人物認識（複数人対応） ---

def identify_people(video_path, person_manager, skip_frames=0):
    """学習した複数人を識別"""
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("Starting identification phase...")
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue
            
        faces = app.get(frame)
        
        for face in faces:
            feature_vector = face.embedding
            bbox = face.bbox.astype(int)
            
            # 顔の向き情報を取得（フラグが有効な場合のみ）
            pose = None
            pose_description = "Unknown"
            
            if ENABLE_FACE_DIRECTION and hasattr(face, 'pose') and face.pose is not None:
                pose = face.pose
                pose_description = get_face_pose_description(pose)
            
            # 学習した人物との類似度をチェック
            matched_id, similarity = person_manager.find_matching_person(
                feature_vector, threshold=SIMILARITY_THRESHOLD
            )
            
            if matched_id is not None:
                # 一致した人物の場合
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {matched_id} ({similarity:.2f})", 
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 顔のランドマークを描画
                frame = draw_face_landmarks(frame, face)
                
                # 顔の向き情報を表示（フラグが有効な場合のみ）
                if ENABLE_FACE_DIRECTION:
                    cv2.putText(frame, pose_description, (bbox[0], bbox[3]+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 顔の向きを3D表現で表示
                    if pose is not None:
                        if USE_3D_BOX_VISUALIZATION:
                            frame = draw_3d_face_orientation_box(frame, bbox, pose)
                        else:
                            frame = draw_face_direction_arrow(frame, bbox, pose, face)
            else:
                # 未知の人物の場合
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 顔のランドマークを描画
                frame = draw_face_landmarks(frame, face)
                
                # 顔の向き情報を表示（フラグが有効な場合のみ）
                if ENABLE_FACE_DIRECTION:
                    cv2.putText(frame, pose_description, (bbox[0], bbox[3]+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 顔の向きを3D表現で表示
                    if pose is not None:
                        if USE_3D_BOX_VISUALIZATION:
                            frame = draw_3d_face_orientation_box(frame, bbox, pose)
                        else:
                            frame = draw_face_direction_arrow(frame, bbox, pose, face)
        
        # ウィンドウサイズを1366x768にリサイズ
        disp_frame = cv2.resize(frame, (1366, 768))
        if ENABLE_FACE_DIRECTION:
            if USE_3D_BOX_VISUALIZATION:
                window_title = 'Identify (Multiple People with 3D Box Direction)'
            else:
                window_title = 'Identify (Multiple People with 3D Arrow Direction)'
        else:
            window_title = 'Identify (Multiple People)'
        cv2.imshow(window_title, disp_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
        
    video.release()
    cv2.destroyAllWindows()

# --- メイン処理 ---
if __name__ == "__main__":
    # 学習フェーズ: 複数人の顔特徴を学習
    person_manager = extract_features_multipeople(VIDEO_PATH_LEARN, skip_frames=SKIP_FRAMES_LEARN)
    
    if len(person_manager.persons) == 0:
        print('No faces detected from learning video.')
        exit()
    
    # 識別フェーズ: 学習した人物を識別
    identify_people(VIDEO_PATH_IDENTIFY, person_manager, skip_frames=SKIP_FRAMES_IDENTIFY)
