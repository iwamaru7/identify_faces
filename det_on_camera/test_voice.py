#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声読み上げ機能のテスト
"""

import sys
import os
import logging
import time

# パスを追加
sys.path.append(os.path.dirname(__file__))

from voice_announcer import VoiceAnnouncer

def test_voice_announcer():
    """音声読み上げ機能をテストする"""
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("音声読み上げテストを開始します...")
    
    try:
        # VoiceAnnouncerを初期化
        voice_announcer = VoiceAnnouncer(logger)
        
        if not voice_announcer.is_engine_available():
            print("音声エンジンが利用できません。")
            return
        
        print("音声エンジンが正常に初期化されました。")
        
        # 現在の時間帯の挨拶を確認
        from datetime import datetime
        current_greeting = voice_announcer.get_time_based_greeting()
        current_hour = datetime.now().hour
        print(f"現在時刻: {current_hour}時 - 挨拶: {current_greeting}")
        
        # テスト1: 名前付きの挨拶
        print("テスト1: 名前付きの挨拶（時間帯対応）")
        voice_announcer.announce_person("田中")
        time.sleep(4)
        
        # テスト2: IDのみの挨拶
        print("テスト2: IDのみの挨拶（時間帯対応）")
        voice_announcer.announce_person_with_id(123)
        time.sleep(4)
        
        # テスト3: 時間帯テスト
        print("テスト3: 異なる時間帯の挨拶テスト")
        print("  - 朝の挨拶（5:00-10:59）")
        voice_announcer.add_to_queue("田中さん、おはようございます")
        time.sleep(3)
        
        print("  - 昼の挨拶（11:00-17:59）")
        voice_announcer.add_to_queue("田中さん、こんにちは")
        time.sleep(3)
        
        print("  - 夜の挨拶（18:00-4:59）")
        voice_announcer.add_to_queue("田中さん、こんばんは")
        time.sleep(3)
        
        # テスト4: カスタムメッセージ
        print("テスト4: カスタムメッセージ")
        voice_announcer.add_to_queue("システムが正常に動作しています")
        time.sleep(3)
        
        print("テスト完了")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        logger.error(f"テスト中にエラー: {e}")
    
    finally:
        # 音声システムを停止
        if 'voice_announcer' in locals():
            voice_announcer.stop()
        print("音声システムを停止しました")

if __name__ == "__main__":
    test_voice_announcer()