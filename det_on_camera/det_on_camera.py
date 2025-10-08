import os
import logging
from datetime import datetime
from person_detection_system import PersonDetectionSystem

# --- 設定 ---
SIMILARITY_THRESHOLD = 0.5  # 同一人物判定の閾値
CAMERA_INDEX = 0  # カメラのインデックス (0が通常のデフォルトカメラ)

# --- フォルダパス ---
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
DETECTED_FACES_DIR = os.path.join(os.path.dirname(__file__), "detected_faces")

# --- ログ設定 ---
def setup_logging():
    """ログ設定を行う"""
    today = datetime.now().strftime("%Y%m%d")
    log_filename = f"detect_{today}.log"
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # ログフォーマット
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()  # コンソールにも出力
        ]
    )

    return logging.getLogger(__name__)

def main():
    """メイン関数"""
    # ログ設定
    logger = setup_logging()
    logger.info("=== 人物検出システムを開始します ===")

    try:
        # 人物検出システムを初期化
        detection_system = PersonDetectionSystem(
            logger=logger,
            training_data_dir=TRAINING_DATA_DIR,
            detected_faces_dir=DETECTED_FACES_DIR,
            similarity_threshold=SIMILARITY_THRESHOLD
        )

        # 学習データを読み込み
        detection_system.load_training_data()

        # カメラによる検出を実行
        detection_system.run_camera_detection(CAMERA_INDEX)

    except Exception as e:
        logger.error(f"システム実行中にエラーが発生しました: {e}")
    finally:
        logger.info("=== 人物検出システムを終了します ===")

if __name__ == "__main__":
    main()