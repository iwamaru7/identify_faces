import cv2
import numpy as np
import os
import time
from datetime import datetime
import glob
import json
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
from voice_announcer import VoiceAnnouncer

# 顔画像保存時のマージン設定（バウンディングボックスからの拡張率）
FACE_MARGIN_RATIO = 0.25  # マージン(%)

class PersonDetectionSystem:
    def __init__(self, logger, training_data_dir, detected_faces_dir, similarity_threshold=0.5):
        self.logger = logger
        self.training_data_dir = training_data_dir
        self.detected_faces_dir = detected_faces_dir
        self.similarity_threshold = similarity_threshold
        self.training_data = {}  # {person_id: feature_vectors_list}
        self.person_names = {}  # {person_id: name}
        self.next_id = 1
        self.person_info_file = os.path.join(self.training_data_dir, "person_info.json")
        
        # 音声読み上げシステムの初期化
        self.voice_announcer = VoiceAnnouncer(logger)
        
        # 最後に挨拶した人物を記録（重複防止用）
        self.last_greeted_persons = {}  # {person_id: last_greeted_time}
        self.greeting_cooldown = 30  # 30秒間は同じ人に再度挨拶しない

        # InsightFaceの初期化
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.logger.info("InsightFaceアプリケーションを初期化しました")

    def load_training_data(self):
        """学習データを全ID分ファイルから読み込む"""
        self.logger.info("学習データの読み込みを開始します")

        # training_dataフォルダ内のperson_*.npyファイルを検索
        pattern = os.path.join(self.training_data_dir, "person_*.npy")
        training_files = glob.glob(pattern)

        loaded_count = 0
        for file_path in training_files:
            try:
                # ファイル名からIDを抽出 (person_123.npy -> 123)
                filename = os.path.basename(file_path)
                id_str = filename.replace("person_", "").replace(".npy", "")
                person_id = int(id_str)

                # 特徴ベクトルを読み込み
                features = np.load(file_path)
                self.training_data[person_id] = features

                self.logger.info(f"ID {person_id}の学習データを読み込みました (特徴ベクトル数: {len(features)})")
                loaded_count += 1

                # 次のIDを更新
                if person_id >= self.next_id:
                    self.next_id = person_id + 1

            except Exception as e:
                self.logger.error(f"学習データの読み込みに失敗しました: {file_path}, エラー: {e}")

        self.logger.info(f"学習データの読み込み完了: {loaded_count}人分のデータを読み込みました")
        
        # 人物名情報も読み込み
        self.load_person_names()

    def load_person_names(self):
        """person_info.jsonから人物名情報を読み込む"""
        self.person_names = {}
        self.logger.info(f"人物名情報ファイルパス: {self.person_info_file}")
        
        if os.path.exists(self.person_info_file):
            try:
                with open(self.person_info_file, 'r', encoding='utf-8') as f:
                    person_info = json.load(f)
                    self.logger.info(f"読み込んだJSON内容: {person_info}")
                    
                    # 文字列キーを整数IDに変換
                    for id_str, name in person_info.items():
                        try:
                            person_id = int(id_str)
                            self.person_names[person_id] = name
                            self.logger.info(f"ID {person_id} に名前 '{name}' を登録")
                        except ValueError:
                            self.logger.warning(f"無効なID形式をスキップ: {id_str}")
                            continue
                            
                self.logger.info(f"人物名情報を読み込みました: {len(self.person_names)}人分")
                self.logger.info(f"登録された名前: {self.person_names}")
                
            except Exception as e:
                self.logger.error(f"人物名情報の読み込みに失敗しました: {e}")
        else:
            self.logger.warning(f"person_info.jsonが見つかりません: {self.person_info_file}")

    def should_greet_person(self, person_id):
        """指定した人物に挨拶すべきかチェックする"""
        current_time = time.time()
        
        if person_id not in self.last_greeted_persons:
            return True
        
        time_since_last_greeting = current_time - self.last_greeted_persons[person_id]
        return time_since_last_greeting >= self.greeting_cooldown
    
    def greet_person(self, person_id, person_name=None):
        """人物に音声で挨拶する"""
        current_time = time.time()
        
        if self.should_greet_person(person_id):
            # 挨拶を実行
            if person_name:
                self.voice_announcer.announce_person(person_name)
                self.logger.info(f"音声挨拶: {person_name}さん、こんにちは")
            else:
                pass
                # self.voice_announcer.announce_person_with_id(person_id)
                # self.logger.info(f"音声挨拶: ID {person_id}の方、こんにちは")
            
            # 挨拶時刻を記録
            self.last_greeted_persons[person_id] = current_time
        else:
            self.logger.debug(f"ID {person_id}: クールダウン中のため挨拶をスキップ")

    def get_person_display_name(self, person_id):
        """IDに対応する表示名を取得（名前があれば名前、なければIDのみ）"""
        self.logger.debug(f"表示名取得: ID {person_id}, 登録名: {self.person_names}")
        
        if person_id in self.person_names:
            name = self.person_names[person_id]
            display_name = f"ID:{person_id} {name}"
            self.logger.debug(f"名前付き表示: {display_name}")
            return display_name
        else:
            display_name = f"ID:{person_id}"
            self.logger.debug(f"ID のみ表示: {display_name}")
            return display_name
    
    def draw_japanese_text(self, img, text, position, font_size=30, color=(0, 255, 0)):
        """日本語テキストを画像に描画する"""
        try:
            # OpenCV画像をPIL画像に変換
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # フォントを設定（Windowsの標準日本語フォント）
            try:
                # Windows標準の日本語フォントを試行
                font_paths = [
                    "C:/Windows/Fonts/msgothic.ttc",  # MS Gothic
                    "C:/Windows/Fonts/meiryo.ttc",    # Meiryo
                    "C:/Windows/Fonts/YuGothM.ttc",   # Yu Gothic Medium
                    "C:/Windows/Fonts/arial.ttf"      # Arial (英語フォールバック)
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                            break
                        except Exception:
                            continue
                
                if font is None:
                    # デフォルトフォントを使用
                    font = ImageFont.load_default()
                    
            except Exception:
                font = ImageFont.load_default()
            
            # RGB色をPIL用に変換
            pil_color = (color[2], color[1], color[0])  # BGR -> RGB
            
            # テキストを描画
            draw.text(position, text, font=font, fill=pil_color)
            
            # PIL画像をOpenCV画像に戻す
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
            
        except Exception as e:
            self.logger.error(f"日本語テキスト描画エラー: {e}")
            # エラー時は元の画像をそのまま返す
            return img

    def find_matching_person(self, feature_vector):
        """検出した人と学習データの人物が一致しているか確認する"""
        best_match_id = None
        best_similarity = 0

        for person_id, features_list in self.training_data.items():
            # 平均特徴ベクトルと比較
            if len(features_list) > 0:
                avg_feature = np.mean(features_list, axis=0)
                similarity = np.dot(feature_vector, avg_feature) / (
                    np.linalg.norm(feature_vector) * np.linalg.norm(avg_feature)
                )

                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = person_id

        return best_match_id, best_similarity

    def save_detected_face(self, frame, bbox, person_id):
        """検出した人の画像(バウンディングボックスの範囲+マージン)をIDと時間をファイル名に含めて保存する"""
        try:
            # バウンディングボックスの座標を取得
            x1, y1, x2, y2 = bbox.astype(int)
            
            # バウンディングボックスのサイズを計算
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # マージンを計算
            margin_x = int(bbox_width * FACE_MARGIN_RATIO)
            margin_y = int(bbox_height * FACE_MARGIN_RATIO)
            
            # マージンを含めた切り抜き範囲を計算
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(frame.shape[1], x2 + margin_x)
            crop_y2 = min(frame.shape[0], y2 + margin_y)
            
            # 顔画像を切り抜き（マージン付き）
            face_image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # ファイル名を生成 (ID_YYYYMMDD_HHMMSS_milliseconds.jpg)
            now = datetime.now()
            filename = f"{person_id}_{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond//1000:03d}.jpg"
            filepath = os.path.join(self.detected_faces_dir, filename)

            # 画像を保存
            cv2.imwrite(filepath, face_image)
            
            # デバッグ情報をログ出力
            self.logger.info(f"検出した顔画像を保存しました: {filename}")
            self.logger.debug(f"元のバウンディングボックス: ({x1},{y1})-({x2},{y2})")
            self.logger.debug(f"マージン付き切り抜き範囲: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
            self.logger.debug(f"保存画像サイズ: {face_image.shape[1]}x{face_image.shape[0]}")

        except Exception as e:
            self.logger.error(f"顔画像の保存に失敗しました: {e}")

    def update_training_data(self, person_id, feature_vector):
        """対象の学習データに今回の人の特徴ベクトルをマージする"""
        try:
            if person_id in self.training_data:
                # 既存のデータに追加
                self.training_data[person_id] = np.vstack([self.training_data[person_id], feature_vector])
            else:
                # 新規作成
                self.training_data[person_id] = np.array([feature_vector])

            # ファイルに保存
            filename = f"person_{person_id}.npy"
            filepath = os.path.join(self.training_data_dir, filename)
            np.save(filepath, self.training_data[person_id])

            self.logger.info(f"ID {person_id}の学習データを更新しました")

        except Exception as e:
            self.logger.error(f"学習データの更新に失敗しました: {e}")

    def create_new_person(self, feature_vector):
        """新規のIDを採番し、学習データファイルを新規作成する"""
        try:
            # 仮のIDを設定（まだ採番しない）
            temp_id = self.next_id

            # 新規学習データを作成
            temp_training_data = np.array([feature_vector])

            # ファイルに保存
            filename = f"person_{temp_id}.npy"
            filepath = os.path.join(self.training_data_dir, filename)
            np.save(filepath, temp_training_data)

            # ファイルが正常に保存されたかを確認
            if os.path.exists(filepath):
                try:
                    # 保存されたファイルを読み込んで検証
                    loaded_data = np.load(filepath)
                    if loaded_data.shape == temp_training_data.shape:
                        # 保存成功時のみIDを正式に採番
                        new_id = self.next_id
                        self.next_id += 1
                        
                        # メモリ上のデータも更新
                        self.training_data[new_id] = temp_training_data
                        
                        self.logger.info(f"新規ID {new_id}を採番し、学習データファイルを作成しました")
                        return new_id
                    else:
                        self.logger.error(f"保存されたファイルのデータが不正です: {filepath}")
                        # 不正なファイルを削除
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        return None
                except Exception as verify_error:
                    self.logger.error(f"保存ファイルの検証に失敗しました: {verify_error}")
                    # エラーのあるファイルを削除
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return None
            else:
                self.logger.error(f"学習データファイルの保存確認に失敗しました: {filepath}")
                return None

        except Exception as e:
            self.logger.error(f"新規ID作成に失敗しました: {e}")
            return None

    def process_detected_faces(self, frame, faces):
        """検出された顔を処理する"""
        for i, face in enumerate(faces):
            try:
                # 特徴ベクトルを抽出
                feature_vector = face.embedding
                bbox = face.bbox

                # 学習データとの照合
                matched_id, similarity = self.find_matching_person(feature_vector)

                if matched_id is not None:
                    # 一致した場合の処理
                    # 顔画像を保存（バウンディングボックス描画前の元フレームから）
                    self.save_detected_face(frame, bbox, matched_id)
                    
                    # 音声で挨拶（名前があれば名前で、なければIDで）
                    person_name = self.person_names.get(matched_id)
                    self.greet_person(matched_id, person_name)
                    
                    # バウンディングボックスとIDを画像に追加
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 表示テキストを作成（名前があれば名前も表示）
                    display_text = f"{self.get_person_display_name(matched_id)} ({similarity:.2f})"
                    
                    # 日本語対応でテキストを描画
                    frame = self.draw_japanese_text(frame, display_text, (x1, y1-35), 
                                                   font_size=25, color=(0, 255, 0))

                    # ログ出力
                    self.logger.info(f"ID {matched_id}の人物が検出されました (類似度: {similarity:.3f})")

                    # 学習データを更新
                    self.update_training_data(matched_id, feature_vector)

                else:
                    # 一致しない場合の処理
                    # 新規IDを採番
                    new_id = self.create_new_person(feature_vector)

                    if new_id is not None:
                        # 顔画像を保存（バウンディングボックス描画前の元フレームから）
                        self.save_detected_face(frame, bbox, new_id)
                        
                        # バウンディングボックスとIDを画像に追加
                        x1, y1, x2, y2 = bbox.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # 新規IDの表示テキスト
                        display_text = f"NEW {self.get_person_display_name(new_id)}"
                        
                        # 日本語対応でテキストを描画
                        frame = self.draw_japanese_text(frame, display_text, (x1, y1-35), 
                                                       font_size=25, color=(0, 0, 255))

                        # ログ出力
                        self.logger.info(f"新規人物が検出され、ID {new_id}を採番しました")

            except Exception as e:
                self.logger.error(f"顔処理中にエラーが発生しました: {e}")

        return frame

    def run_camera_detection(self, camera_index=0):
        """カメラからの人物検出メインループ"""
        self.logger.info("カメラによる人物検出を開始します")

        # 人物名情報を最新の状態に更新
        self.load_person_names()

        # カメラを初期化
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self.logger.error(f"カメラ(index: {camera_index})を開けませんでした")
            return

        self.logger.info("カメラが正常に初期化されました。'q'キーで終了してください。")

        try:
            while True:
                # フレームを読み込み
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("フレームの読み込みに失敗しました")
                    break

                # 人を検出
                faces = self.app.get(frame)

                if len(faces) > 0:
                    # 人が検出された場合
                    self.logger.info(f"{len(faces)}人の顔が検出されました")

                    # 検出された顔を処理
                    frame = self.process_detected_faces(frame, faces)

                # フレームを表示
                cv2.imshow('Camera Face Detection', frame)

                # 'q'キーが押されたら終了
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("終了キーが押されました")
                    break

        except KeyboardInterrupt:
            self.logger.info("プログラムが中断されました")
        except Exception as e:
            self.logger.error(f"カメラ処理中にエラーが発生しました: {e}")
        finally:
            # リソースを解放
            cap.release()
            cv2.destroyAllWindows()
            
            # 音声システムを停止
            self.voice_announcer.stop()
            
            self.logger.info("カメラリソースを解放しました")