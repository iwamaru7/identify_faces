import cv2
from insightface.app import FaceAnalysis
import numpy as np
from collections import defaultdict

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
                    print(f"ID {person_id}のトラッカーを初期化しました")
                else:
                    print(f"警告: ID {person_id}のトラッカー初期化に失敗しました")
                    self.persons[person_id]['tracker'] = None
                    
            except Exception as e:
                print(f"警告: トラッカーの初期化でエラーが発生しました: {e}")
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
    
    print("学習フェーズを開始します...")
    
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
            
            # 既存の人物との類似度をチェック
            matched_id, similarity = person_manager.find_matching_person(
                feature_vector, threshold=TRACKER_THRESHOLD
            )
            
            if matched_id is not None:
                # 既存の人物に追加学習
                person_manager.add_feature_to_person(matched_id, feature_vector)
                print(f"ID {matched_id}に特徴を追加学習 (類似度: {similarity:.3f})")
            else:
                # 新しい人物として登録
                new_id = person_manager.add_person(feature_vector, bbox)
                person_manager.init_tracker(new_id, frame, bbox)
                print(f"新しい人物ID {new_id}を登録")
            
            # バウンディングボックスとIDを描画
            current_id = matched_id if matched_id is not None else person_manager.next_id - 1
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {current_id}", (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
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
        cv2.imshow('Learn (Multiple People)', disp_frame)

        learn_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video.release()
    cv2.destroyAllWindows()
    
    print(f"学習完了: {len(person_manager.persons)}人を学習しました")
    for person_id, person_data in person_manager.persons.items():
        print(f"  ID {person_id}: {len(person_data['features'])}個の特徴ベクトル")
    
    return person_manager

# --- (2) 識別用動画での人物認識（複数人対応） ---

def identify_people(video_path, person_manager, skip_frames=0):
    """学習した複数人を識別"""
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("識別フェーズを開始します...")
    
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
            
            # 学習した人物との類似度をチェック
            matched_id, similarity = person_manager.find_matching_person(
                feature_vector, threshold=SIMILARITY_THRESHOLD
            )
            
            if matched_id is not None:
                # 一致した人物の場合
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {matched_id} ({similarity:.2f})", 
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # 未知の人物の場合
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # ウィンドウサイズを1366x768にリサイズ
        disp_frame = cv2.resize(frame, (1366, 768))
        cv2.imshow('Identify (Multiple People)', disp_frame)
        
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
        print('学習用動画から顔が検出できませんでした。')
        exit()
    
    # 識別フェーズ: 学習した人物を識別
    identify_people(VIDEO_PATH_IDENTIFY, person_manager, skip_frames=SKIP_FRAMES_IDENTIFY)
