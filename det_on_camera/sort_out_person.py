import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import glob
import json
import numpy as np
import shutil
from datetime import datetime
from PIL import Image, ImageTk
import cv2
from insightface.app import FaceAnalysis

class PersonSortOutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人物データ整理ツール")
        self.root.geometry("1200x800")
        
        # フォルダパス
        self.script_dir = os.path.dirname(__file__)
        self.training_data_dir = os.path.join(self.script_dir, "training_data")
        self.detected_faces_dir = os.path.join(self.script_dir, "detected_faces")
        self.del_img_dir = os.path.join(self.detected_faces_dir, "del_img")
        self.person_info_file = os.path.join(self.training_data_dir, "person_info.json")
        
        # del_imgフォルダが存在しない場合は作成
        os.makedirs(self.del_img_dir, exist_ok=True)
        
        # InsightFaceの初期化
        self.app = None
        self.init_insightface()
        
        # データ
        self.person_data = {}  # {person_id: {'images': [paths], 'checkbox_var': tk.BooleanVar()}}
        self.person_info = {}  # {person_id: name}
        
        # GUI作成
        self.create_widgets()
        
        # データ読み込み
        self.load_data()
    
    def init_insightface(self):
        """InsightFaceを初期化"""
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFaceアプリケーションを初期化しました")
        except Exception as e:
            print(f"InsightFace初期化エラー: {e}")
            self.app = None
    
    def create_widgets(self):
        """GUI要素を作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上部：コントロールパネル
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 統合機能
        integrate_frame = ttk.LabelFrame(control_frame, text="統合機能")
        integrate_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Label(integrate_frame, text="統合先ID:").pack(anchor=tk.W)
        self.target_id_var = tk.StringVar()
        self.target_id_combo = ttk.Combobox(integrate_frame, textvariable=self.target_id_var, state="readonly")
        self.target_id_combo.pack(pady=2)
        
        ttk.Button(integrate_frame, text="統合", command=self.integrate_persons).pack(pady=5)
        
        # 削除機能
        delete_frame = ttk.LabelFrame(control_frame, text="削除機能")
        delete_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Button(delete_frame, text="削除", command=self.delete_persons).pack(pady=5)
        
        # 名前登録機能
        name_frame = ttk.LabelFrame(control_frame, text="名前登録")
        name_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Label(name_frame, text="ID:").pack(anchor=tk.W)
        self.name_id_var = tk.StringVar()
        self.name_id_combo = ttk.Combobox(name_frame, textvariable=self.name_id_var, state="readonly")
        self.name_id_combo.pack(pady=2)
        
        ttk.Label(name_frame, text="名前:").pack(anchor=tk.W)
        self.name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.name_var).pack(pady=2)
        
        ttk.Button(name_frame, text="登録", command=self.register_name).pack(pady=5)
        
        # 学習データ再作成機能
        recreate_frame = ttk.LabelFrame(control_frame, text="学習データ再作成")
        recreate_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        ttk.Button(recreate_frame, text="選択IDの学習データ再作成", command=self.recreate_selected_training_data).pack(pady=5)
        
        # 更新ボタン
        ttk.Button(control_frame, text="データ更新", command=self.load_data).pack(side=tk.RIGHT, pady=5)
        
        # 下部：スクロール可能な人物表示エリア
        self.canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # マウスホイールでスクロール
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """マウスホイールでスクロール"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def load_data(self):
        """データを読み込み"""
        # 既存のデータをクリア
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.person_data.clear()
        
        # 人物情報を読み込み
        self.load_person_info()
        
        # 全ての人物IDを収集（学習データファイルと画像ファイルの両方から）
        all_person_ids = set()
        
        # 学習データファイルからIDを取得
        pattern = os.path.join(self.training_data_dir, "person_*.npy")
        training_files = glob.glob(pattern)
        for file_path in training_files:
            filename = os.path.basename(file_path)
            person_id = int(filename.replace("person_", "").replace(".npy", ""))
            all_person_ids.add(person_id)
        
        # 画像ファイルからIDを取得
        image_pattern = os.path.join(self.detected_faces_dir, "*_*.jpg")
        image_files = glob.glob(image_pattern)
        for img_path in image_files:
            filename = os.path.basename(img_path)
            try:
                # ファイル名からIDを抽出 (ID_YYYYMMDD_HHMMSS_milliseconds.jpg)
                person_id = int(filename.split('_')[0])
                all_person_ids.add(person_id)
            except (ValueError, IndexError):
                # ファイル名が期待する形式でない場合はスキップ
                continue
        
        if not all_person_ids:
            ttk.Label(self.scrollable_frame, text="人物データが見つかりません").pack(pady=20)
            return
        
        # 各IDの画像を収集
        for person_id in all_person_ids:
            # 対応する画像ファイルを取得
            id_image_pattern = os.path.join(self.detected_faces_dir, f"{person_id}_*.jpg")
            image_files = glob.glob(id_image_pattern)
            
            if image_files:  # 画像ファイルが存在する場合のみ表示
                # 作成日時でソートして、離れた5ファイルを選択
                selected_images = self.select_representative_images(image_files, 5)
                
                # データを保存
                checkbox_var = tk.BooleanVar()
                self.person_data[person_id] = {
                    'images': selected_images,
                    'checkbox_var': checkbox_var
                }
        
        # GUIを更新
        self.update_display()
        self.update_comboboxes()
    
    def select_representative_images(self, image_files, max_count):
        """作成日時が離れた代表的な画像を選択"""
        if len(image_files) <= max_count:
            return image_files
        
        # ファイルを作成日時でソート
        files_with_time = []
        for file_path in image_files:
            mtime = os.path.getmtime(file_path)
            files_with_time.append((file_path, mtime))
        
        files_with_time.sort(key=lambda x: x[1])
        
        # 等間隔で選択
        selected = []
        total_files = len(files_with_time)
        interval = total_files / max_count
        
        for i in range(max_count):
            index = int(i * interval)
            if index < total_files:
                selected.append(files_with_time[index][0])
        
        return selected
    
    def update_display(self):
        """人物表示を更新"""
        for row, (person_id, data) in enumerate(sorted(self.person_data.items())):
            # 行フレーム
            row_frame = ttk.Frame(self.scrollable_frame)
            row_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # チェックボックス
            checkbox = ttk.Checkbutton(row_frame, variable=data['checkbox_var'])
            checkbox.pack(side=tk.LEFT, padx=(0, 10))
            
            # ID表示
            id_label = ttk.Label(row_frame, text=f"ID: {person_id}", font=("Arial", 12, "bold"))
            id_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # 名前表示
            name = self.person_info.get(str(person_id), "未登録")
            name_label = ttk.Label(row_frame, text=f"名前: {name}", font=("Arial", 10))
            name_label.pack(side=tk.LEFT, padx=(0, 20))
            
            # 画像枚数表示
            total_images = len(glob.glob(os.path.join(self.detected_faces_dir, f"{person_id}_*.jpg")))
            count_label = ttk.Label(row_frame, text=f"画像数: {total_images}枚", font=("Arial", 9))
            count_label.pack(side=tk.LEFT, padx=(0, 20))
            
            # 画像表示フレーム
            image_frame = ttk.Frame(row_frame)
            image_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # 画像を表示
            for img_path in data['images']:
                try:
                    # 画像を読み込み、リサイズ
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (80, 80))
                        
                        # PIL Imageに変換してTkinterで表示
                        pil_img = Image.fromarray(img)
                        tk_img = ImageTk.PhotoImage(pil_img)
                        
                        img_label = ttk.Label(image_frame, image=tk_img)
                        img_label.image = tk_img  # 参照を保持
                        
                        # ダブルクリックで詳細ウィンドウを開く
                        img_label.bind("<Double-Button-1>", lambda e, pid=person_id: self.open_image_viewer(pid))
                        
                        img_label.pack(side=tk.LEFT, padx=2)
                except Exception as e:
                    print(f"画像読み込みエラー: {img_path}, {e}")
            
            # 「全画像表示」ボタンを追加
            view_all_btn = ttk.Button(
                row_frame, 
                text="全画像表示", 
                command=lambda pid=person_id: self.open_image_viewer(pid)
            )
            view_all_btn.pack(side=tk.RIGHT, padx=10)
            
            # 区切り線
            ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=5)
    
    def update_comboboxes(self):
        """コンボボックスを更新"""
        person_ids = [str(pid) for pid in sorted(self.person_data.keys())]
        
        self.target_id_combo['values'] = person_ids
        self.name_id_combo['values'] = person_ids
        
        if person_ids:
            if not self.target_id_var.get():
                self.target_id_var.set(person_ids[0])
            if not self.name_id_var.get():
                self.name_id_var.set(person_ids[0])
    
    def integrate_persons(self):
        """選択された人物を統合"""
        # 選択されたIDを取得
        selected_ids = []
        for person_id, data in self.person_data.items():
            if data['checkbox_var'].get():
                selected_ids.append(person_id)
        
        if not selected_ids:
            messagebox.showwarning("警告", "統合するIDを選択してください")
            return
        
        target_id_str = self.target_id_var.get()
        if not target_id_str:
            messagebox.showwarning("警告", "統合先IDを選択してください")
            return
        
        target_id = int(target_id_str)
        
        if target_id in selected_ids:
            messagebox.showwarning("警告", "統合先IDは選択対象から除外してください")
            return
        
        try:
            # 統合先の学習データを読み込み
            target_file = os.path.join(self.training_data_dir, f"person_{target_id}.npy")
            if os.path.exists(target_file):
                target_features = np.load(target_file)
            else:
                messagebox.showerror("エラー", f"統合先ID {target_id} の学習データが見つかりません")
                return
            
            # 選択されたIDの特徴量をマージ
            all_features = [target_features]
            
            for selected_id in selected_ids:
                source_file = os.path.join(self.training_data_dir, f"person_{selected_id}.npy")
                if os.path.exists(source_file):
                    features = np.load(source_file)
                    all_features.append(features)
            
            # 特徴量を結合
            merged_features = np.vstack(all_features)
            
            # 統合先に保存
            np.save(target_file, merged_features)
            
            # 画像ファイル名を変更
            for selected_id in selected_ids:
                self.rename_image_files(selected_id, target_id)
                
                # 学習ファイルを削除
                source_file = os.path.join(self.training_data_dir, f"person_{selected_id}.npy")
                if os.path.exists(source_file):
                    os.remove(source_file)
                
                # 人物情報から削除
                self.remove_person_info(selected_id)
            
            messagebox.showinfo("完了", f"ID {selected_ids} を ID {target_id} に統合しました")
            self.load_data()
            
        except Exception as e:
            messagebox.showerror("エラー", f"統合処理でエラーが発生しました: {e}")
    
    def rename_image_files(self, source_id, target_id):
        """画像ファイル名を変更"""
        pattern = os.path.join(self.detected_faces_dir, f"{source_id}_*.jpg")
        source_files = glob.glob(pattern)
        
        for source_path in source_files:
            filename = os.path.basename(source_path)
            # ID部分を置換
            new_filename = filename.replace(f"{source_id}_", f"{target_id}_", 1)
            new_path = os.path.join(self.detected_faces_dir, new_filename)
            
            # 重複チェック
            counter = 1
            while os.path.exists(new_path):
                name, ext = os.path.splitext(new_filename)
                new_filename = f"{name}_{counter}{ext}"
                new_path = os.path.join(self.detected_faces_dir, new_filename)
                counter += 1
            
            # ファイル名変更
            shutil.move(source_path, new_path)
    
    def delete_persons(self):
        """選択された人物を削除"""
        # 選択されたIDを取得
        selected_ids = []
        for person_id, data in self.person_data.items():
            if data['checkbox_var'].get():
                selected_ids.append(person_id)
        
        if not selected_ids:
            messagebox.showwarning("警告", "削除するIDを選択してください")
            return
        
        # 確認ダイアログ
        if not messagebox.askyesno("確認", f"ID {selected_ids} を削除しますか？"):
            return
        
        try:
            for selected_id in selected_ids:
                # 学習ファイルを削除
                training_file = os.path.join(self.training_data_dir, f"person_{selected_id}.npy")
                if os.path.exists(training_file):
                    os.remove(training_file)
                
                # 画像ファイルをdel_imgフォルダに移動
                pattern = os.path.join(self.detected_faces_dir, f"{selected_id}_*.jpg")
                image_files = glob.glob(pattern)
                
                for img_path in image_files:
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(self.del_img_dir, filename)
                    
                    # 重複チェック
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        new_filename = f"{name}_{counter}{ext}"
                        dest_path = os.path.join(self.del_img_dir, new_filename)
                        counter += 1
                    
                    shutil.move(img_path, dest_path)
                
                # 人物情報から削除
                self.remove_person_info(selected_id)
            
            messagebox.showinfo("完了", f"ID {selected_ids} を削除しました")
            self.load_data()
            
        except Exception as e:
            messagebox.showerror("エラー", f"削除処理でエラーが発生しました: {e}")
    
    def register_name(self):
        """名前を登録"""
        person_id_str = self.name_id_var.get()
        name = self.name_var.get().strip()
        
        if not person_id_str:
            messagebox.showwarning("警告", "IDを選択してください")
            return
        
        if not name:
            messagebox.showwarning("警告", "名前を入力してください")
            return
        
        try:
            person_id = int(person_id_str)
            
            # 人物情報を更新
            self.person_info[person_id_str] = name
            self.save_person_info()
            
            messagebox.showinfo("完了", f"ID {person_id} に名前 '{name}' を登録しました")
            
            # 表示を更新
            self.load_data()
            
            # 入力欄をクリア
            self.name_var.set("")
            
        except Exception as e:
            messagebox.showerror("エラー", f"名前登録でエラーが発生しました: {e}")
    
    def recreate_selected_training_data(self):
        """選択されたIDの学習データを再作成"""
        # 選択されたIDを取得
        selected_ids = []
        for person_id, data in self.person_data.items():
            if data['checkbox_var'].get():
                selected_ids.append(person_id)
        
        if not selected_ids:
            messagebox.showwarning("警告", "学習データを再作成するIDを選択してください")
            return
        
        if self.app is None:
            messagebox.showerror("エラー", "InsightFaceが初期化されていません")
            return
        
        # 確認ダイアログ
        if not messagebox.askyesno("確認", f"ID {selected_ids} の学習データを再作成しますか？\n既存の学習データは上書きされます。"):
            return
        
        try:
            results = []
            total_processed = 0
            total_images = 0
            
            # プログレス表示用のウィンドウを作成
            progress_window = tk.Toplevel(self.root)
            progress_window.title("学習データ再作成中")
            progress_window.geometry("400x150")
            progress_window.resizable(False, False)
            
            # プログレス表示
            progress_label = ttk.Label(progress_window, text="処理中...", font=("Arial", 12))
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, mode='determinate')
            progress_bar.pack(pady=10, padx=20, fill=tk.X)
            progress_bar['maximum'] = len(selected_ids)
            
            detail_label = ttk.Label(progress_window, text="", font=("Arial", 10))
            detail_label.pack(pady=5)
            
            # ウィンドウを更新
            progress_window.update()
            
            for i, person_id in enumerate(selected_ids):
                try:
                    progress_label.config(text=f"ID {person_id} の学習データを再作成中...")
                    detail_label.config(text=f"進行状況: {i+1}/{len(selected_ids)}")
                    progress_window.update()
                    
                    processed_count, total_count = self.recreate_training_data_for_person(person_id)
                    results.append(f"ID {person_id}: {processed_count}/{total_count}枚処理")
                    total_processed += processed_count
                    total_images += total_count
                    
                    progress_bar['value'] = i + 1
                    progress_window.update()
                    
                except Exception as e:
                    results.append(f"ID {person_id}: エラー - {e}")
                    print(f"ID {person_id} の学習データ再作成エラー: {e}")
            
            # プログレスウィンドウを閉じる
            progress_window.destroy()
            
            # 結果を表示
            result_text = f"学習データ再作成完了\n\n"
            result_text += f"全体: {total_processed}/{total_images}枚の画像を処理\n\n"
            result_text += "詳細:\n" + "\n".join(results)
            
            messagebox.showinfo("完了", result_text)
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("エラー", f"学習データ再作成でエラーが発生しました: {e}")
    
    def load_person_info(self):
        """人物情報を読み込み"""
        self.person_info = {}
        if os.path.exists(self.person_info_file):
            try:
                with open(self.person_info_file, 'r', encoding='utf-8') as f:
                    self.person_info = json.load(f)
            except Exception as e:
                print(f"人物情報読み込みエラー: {e}")
    
    def save_person_info(self):
        """人物情報を保存"""
        try:
            with open(self.person_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.person_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"人物情報保存エラー: {e}")
    
    def remove_person_info(self, person_id):
        """人物情報から指定IDを削除"""
        person_id_str = str(person_id)
        if person_id_str in self.person_info:
            del self.person_info[person_id_str]
            self.save_person_info()
    
    def open_image_viewer(self, person_id):
        """指定IDの全画像を表示するウィンドウを開く"""
        # 対応する全画像ファイルを取得
        image_pattern = os.path.join(self.detected_faces_dir, f"{person_id}_*.jpg")
        all_image_files = glob.glob(image_pattern)
        
        if not all_image_files:
            messagebox.showinfo("情報", f"ID {person_id} の画像が見つかりません")
            return
        
        # 作成日時でソート
        image_files_with_time = []
        for img_path in all_image_files:
            try:
                mtime = os.path.getmtime(img_path)
                image_files_with_time.append((img_path, mtime))
            except Exception as e:
                print(f"ファイル時刻取得エラー: {img_path}, {e}")
                continue
        
        # 時刻順にソート（新しい順）
        image_files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # 画像ビューワーウィンドウを作成
        self.create_image_viewer_window(person_id, image_files_with_time)
    
    def create_image_viewer_window(self, person_id, image_files_with_time):
        """画像ビューワーウィンドウを作成"""
        # 新しいウィンドウを作成
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title(f"ID {person_id} の全画像 ({len(image_files_with_time)}枚)")
        viewer_window.geometry("1000x700")
        
        # 人物名を取得
        person_name = self.person_info.get(str(person_id), "未登録")
        
        # ヘッダー情報フレーム
        header_frame = ttk.Frame(viewer_window)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 情報ラベル
        info_label = ttk.Label(
            header_frame, 
            text=f"ID: {person_id} | 名前: {person_name} | 画像数: {len(image_files_with_time)}枚",
            font=("Arial", 14, "bold")
        )
        info_label.pack(side=tk.LEFT)
        
        # 選択・削除ボタンフレーム
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # 画像選択状態を保存する辞書
        image_selection = {}
        
        # 全選択/全解除ボタン
        def toggle_all_selection():
            all_selected = all(var.get() for var in image_selection.values())
            new_state = not all_selected
            for var in image_selection.values():
                var.set(new_state)
        
        toggle_btn = ttk.Button(button_frame, text="全選択/解除", command=toggle_all_selection)
        toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # 選択画像削除ボタン
        def delete_selected_images():
            selected_images = [path for path, var in image_selection.items() if var.get()]
            if not selected_images:
                messagebox.showwarning("警告", "削除する画像を選択してください")
                return
            
            if messagebox.askyesno("確認", f"{len(selected_images)}枚の画像を削除しますか？"):
                try:
                    moved_count = 0
                    for img_path in selected_images:
                        filename = os.path.basename(img_path)
                        dest_path = os.path.join(self.del_img_dir, filename)
                        
                        # 重複チェック
                        counter = 1
                        while os.path.exists(dest_path):
                            name, ext = os.path.splitext(filename)
                            new_filename = f"{name}_{counter}{ext}"
                            dest_path = os.path.join(self.del_img_dir, new_filename)
                            counter += 1
                        
                        # ファイル移動
                        shutil.move(img_path, dest_path)
                        moved_count += 1
                    
                    messagebox.showinfo("完了", f"{moved_count}枚の画像を削除しました")
                    viewer_window.destroy()
                    self.load_data()  # メイン画面を更新
                    
                except Exception as e:
                    messagebox.showerror("エラー", f"画像削除でエラーが発生しました: {e}")
        
        delete_btn = ttk.Button(button_frame, text="選択画像削除", command=delete_selected_images)
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        # 学習データ再作成ボタン
        def recreate_training_data():
            if messagebox.askyesno("確認", f"ID {person_id} の学習データを全画像から再作成しますか？\n既存の学習データは上書きされます。"):
                try:
                    self.recreate_training_data_for_person(person_id)
                    messagebox.showinfo("完了", f"ID {person_id} の学習データを再作成しました")
                except Exception as e:
                    messagebox.showerror("エラー", f"学習データ再作成でエラーが発生しました: {e}")
        
        recreate_btn = ttk.Button(button_frame, text="学習データ再作成", command=recreate_training_data)
        recreate_btn.pack(side=tk.LEFT, padx=5)
        
        # スクロール可能なフレームを作成
        canvas = tk.Canvas(viewer_window, bg="white")
        scrollbar_v = ttk.Scrollbar(viewer_window, orient="vertical", command=canvas.yview)
        scrollbar_h = ttk.Scrollbar(viewer_window, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # 画像を格子状に配置
        images_per_row = 5  # 1行あたりの画像数
        current_row = 0
        current_col = 0
        
        for i, (img_path, mtime) in enumerate(image_files_with_time):
            try:
                # 画像を読み込み、リサイズ
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (150, 150))  # ビューワーでは大きめに表示
                
                # PIL Imageに変換
                pil_img = Image.fromarray(img)
                tk_img = ImageTk.PhotoImage(pil_img)
                
                # 画像用のフレーム
                img_container = ttk.Frame(scrollable_frame, relief='ridge', borderwidth=1)
                img_container.grid(row=current_row, column=current_col, padx=5, pady=5)
                
                # 選択用チェックボックス
                selection_var = tk.BooleanVar()
                image_selection[img_path] = selection_var
                
                checkbox = ttk.Checkbutton(img_container, variable=selection_var)
                checkbox.pack(anchor=tk.W)
                
                # 画像ラベル
                img_label = ttk.Label(img_container, image=tk_img)
                img_label.image = tk_img  # 参照を保持
                img_label.pack()
                
                # ファイル名ラベル
                filename = os.path.basename(img_path)
                filename_label = ttk.Label(
                    img_container, 
                    text=filename, 
                    font=("Arial", 8),
                    wraplength=150
                )
                filename_label.pack()
                
                # 作成日時ラベル
                try:
                    datetime_str = datetime.fromtimestamp(mtime).strftime("%Y/%m/%d %H:%M:%S")
                    time_label = ttk.Label(
                        img_container, 
                        text=datetime_str, 
                        font=("Arial", 8),
                        foreground="gray"
                    )
                    time_label.pack()
                except Exception as e:
                    print(f"日時表示エラー: {e}")
                
                # 画像をクリックで拡大表示
                img_label.bind("<Button-1>", lambda e, path=img_path: self.show_large_image(path))
                
                # チェックボックスをクリックしたときの視覚的フィードバック
                def on_checkbox_change(container=img_container, var=selection_var):
                    if var.get():
                        container.configure(relief='solid', borderwidth=3)
                    else:
                        container.configure(relief='ridge', borderwidth=1)
                
                selection_var.trace('w', lambda *args, container=img_container, var=selection_var: on_checkbox_change(container, var))
                
                # 次の位置を計算
                current_col += 1
                if current_col >= images_per_row:
                    current_col = 0
                    current_row += 1
                    
            except Exception as e:
                print(f"画像処理エラー: {img_path}, {e}")
                continue
        
        # スクロールバーとキャンバスを配置
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        
        # マウスホイールでスクロール
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # ウィンドウが閉じられたときのクリーンアップ
        def on_closing():
            canvas.unbind_all("<MouseWheel>")
            viewer_window.destroy()
        
        viewer_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    def show_large_image(self, img_path):
        """画像を拡大表示するウィンドウを開く"""
        try:
            # 新しいウィンドウを作成
            large_window = tk.Toplevel(self.root)
            large_window.title(f"画像表示: {os.path.basename(img_path)}")
            
            # 画像を読み込み
            img = cv2.imread(img_path)
            if img is None:
                messagebox.showerror("エラー", "画像を読み込めませんでした")
                large_window.destroy()
                return
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 画面サイズに合わせてリサイズ
            screen_width = large_window.winfo_screenwidth()
            screen_height = large_window.winfo_screenheight()
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8)
            
            h, w = img.shape[:2]
            scale = min(max_width / w, max_height / h, 1.0)
            
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # PIL Imageに変換
            pil_img = Image.fromarray(img)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # ウィンドウサイズを画像に合わせる
            large_window.geometry(f"{img.shape[1]}x{img.shape[0]+50}")
            
            # 画像ラベル
            img_label = ttk.Label(large_window, image=tk_img)
            img_label.image = tk_img  # 参照を保持
            img_label.pack(padx=10, pady=10)
            
            # ファイル情報
            try:
                mtime = os.path.getmtime(img_path)
                datetime_str = datetime.fromtimestamp(mtime).strftime("%Y/%m/%d %H:%M:%S")
                info_text = f"ファイル: {os.path.basename(img_path)} | 作成日時: {datetime_str}"
                info_label = ttk.Label(large_window, text=info_text, font=("Arial", 10))
                info_label.pack(pady=5)
            except Exception as e:
                print(f"ファイル情報取得エラー: {e}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"画像表示でエラーが発生しました: {e}")
    
    def recreate_training_data_for_person(self, person_id):
        """指定IDの全画像から学習データを再作成"""
        if self.app is None:
            raise Exception("InsightFaceが初期化されていません")
        
        # 対応する全画像ファイルを取得
        image_pattern = os.path.join(self.detected_faces_dir, f"{person_id}_*.jpg")
        all_image_files = glob.glob(image_pattern)
        
        if not all_image_files:
            raise Exception(f"ID {person_id} の画像が見つかりません")
        
        print(f"ID {person_id}: {len(all_image_files)}枚の画像から学習データを再作成します")
        
        # 特徴ベクトルを抽出
        feature_vectors = []
        processed_count = 0
        
        for img_path in all_image_files:
            try:
                # 画像を読み込み
                img = cv2.imread(img_path)
                if img is None:
                    print(f"画像読み込み失敗: {img_path}")
                    continue
                
                # InsightFaceで顔検出・特徴抽出
                faces = self.app.get(img)
                
                if len(faces) == 0:
                    print(f"顔が検出されませんでした: {img_path}")
                    continue
                
                # 最初に検出された顔の特徴ベクトルを使用
                # （基本的に1つの顔画像なので最初の顔を使用）
                feature_vector = faces[0].embedding
                feature_vectors.append(feature_vector)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"処理済み: {processed_count}/{len(all_image_files)}")
                
            except Exception as e:
                print(f"画像処理エラー: {img_path}, {e}")
                continue
        
        if len(feature_vectors) == 0:
            raise Exception("有効な特徴ベクトルが抽出できませんでした")
        
        # 特徴ベクトルをnumpy配列に変換
        training_data = np.vstack(feature_vectors)
        
        # 学習データファイルを保存
        filename = f"person_{person_id}.npy"
        filepath = os.path.join(self.training_data_dir, filename)
        np.save(filepath, training_data)
        
        print(f"学習データを保存しました: {filepath}")
        print(f"処理完了: {processed_count}/{len(all_image_files)}枚の画像から特徴ベクトルを抽出")
        
        return processed_count, len(all_image_files)

def main():
    root = tk.Tk()
    app = PersonSortOutApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()