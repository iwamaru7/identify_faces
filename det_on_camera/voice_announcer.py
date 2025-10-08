import pyttsx3
import threading
import queue
import time
from datetime import datetime
from typing import Optional

class VoiceAnnouncer:
    """音声読み上げを管理するクラス"""
    
    def __init__(self, logger):
        self.logger = logger
        self.engine = None
        self.voice_queue = queue.Queue()
        self.is_speaking = False
        self.worker_thread = None
        self.should_stop = False
        
        # 音声エンジンの初期化
        self._initialize_engine()
        
        # ワーカースレッドの開始
        self._start_worker_thread()
    
    def _initialize_engine(self):
        """音声エンジンを初期化する"""
        try:
            self.engine = pyttsx3.init()
            
            # 音声設定
            voices = self.engine.getProperty('voices')
            if voices:
                # 日本語音声があれば使用（Windowsの場合）
                for voice in voices:
                    if 'japanese' in voice.name.lower() or 'japan' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        self.logger.info(f"日本語音声を設定しました: {voice.name}")
                        break
                else:
                    # デフォルト音声を使用
                    self.engine.setProperty('voice', voices[0].id)
                    self.logger.info(f"デフォルト音声を設定しました: {voices[0].name}")
            
            # 音声速度設定（やや速め）
            self.engine.setProperty('rate', 150)
            
            # 音量設定
            self.engine.setProperty('volume', 0.8)
            
            self.logger.info("音声エンジンの初期化が完了しました")
            
        except Exception as e:
            self.logger.error(f"音声エンジンの初期化に失敗しました: {e}")
            self.engine = None
    
    def _start_worker_thread(self):
        """ワーカースレッドを開始する"""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.logger.info("音声読み上げワーカースレッドを開始しました")
    
    def _worker(self):
        """音声読み上げを処理するワーカースレッド"""
        while not self.should_stop:
            try:
                # キューからメッセージを取得（タイムアウト付き）
                message = self.voice_queue.get(timeout=1.0)
                
                if message is None:  # 終了シグナル
                    break
                
                # 音声読み上げを実行
                self._speak_text(message)
                
                # タスク完了をマーク
                self.voice_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"音声読み上げワーカーでエラーが発生しました: {e}")
    
    def _speak_text(self, text: str):
        """実際に音声を読み上げる"""
        if self.engine is None:
            self.logger.warning("音声エンジンが利用できません")
            return
        
        try:
            self.is_speaking = True
            self.logger.info(f"音声読み上げ開始: {text}")
            
            # 音声読み上げを実行
            self.engine.say(text)
            self.engine.runAndWait()
            
            self.logger.info("音声読み上げ完了")
            
        except Exception as e:
            self.logger.error(f"音声読み上げ中にエラーが発生しました: {e}")
        finally:
            self.is_speaking = False
    
    def get_time_based_greeting(self):
        """現在時刻に応じた挨拶を取得する"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 11:
            return "おはようございます"
        elif 18 <= current_hour or current_hour < 5:
            return "こんばんは"
        else:
            return "こんにちは"
    
    def announce_person(self, person_name: str):
        """人物の名前を音声で読み上げる（時間帯に応じた挨拶）"""
        greeting = self.get_time_based_greeting()
        
        if person_name:
            message = f"{person_name}さん、{greeting}"
        else:
            message = greeting
        
        self.add_to_queue(message)
    
    def announce_person_with_id(self, person_id: int, person_name: Optional[str] = None):
        """IDと名前を使って音声で読み上げる（時間帯に応じた挨拶）"""
        greeting = self.get_time_based_greeting()
        
        if person_name:
            message = f"{person_name}さん、{greeting}"
        else:
            message = f"ID {person_id}の方、{greeting}"
        
        self.add_to_queue(message)
    
    def add_to_queue(self, message: str):
        """音声読み上げメッセージをキューに追加する"""
        try:
            # 既に同じメッセージが読み上げ中でない場合のみ追加
            if not self.is_speaking:
                self.voice_queue.put(message)
                self.logger.debug(f"音声メッセージをキューに追加しました: {message}")
            else:
                self.logger.debug("音声読み上げ中のため、メッセージをスキップしました")
                
        except Exception as e:
            self.logger.error(f"音声メッセージのキュー追加に失敗しました: {e}")
    
    def is_engine_available(self) -> bool:
        """音声エンジンが利用可能かチェックする"""
        return self.engine is not None
    
    def stop(self):
        """音声読み上げシステムを停止する"""
        self.logger.info("音声読み上げシステムを停止しています...")
        
        # 停止フラグを設定
        self.should_stop = True
        
        # 終了シグナルをキューに追加
        self.voice_queue.put(None)
        
        # ワーカースレッドの終了を待機
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        
        # エンジンを停止
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                self.logger.warning(f"音声エンジンの停止中にエラーが発生しました: {e}")
        
        self.logger.info("音声読み上げシステムが停止しました")