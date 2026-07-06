import sys
import os
import threading

# 嘗試導入必要套件，若缺少則顯示圖形化錯誤提示
try:
    import torch
    from PIL import Image, ImageTk
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError as e:
    import tkinter as tk
    from tkinter import messagebox
    
    # 建立一個隱藏的主視窗用於顯示錯誤對話框
    root = tk.Tk()
    root.withdraw()
    
    missing_packages = []
    error_msg = str(e)
    if "PIL" in error_msg or "Pillow" in error_msg:
        missing_packages.append("pillow")
    if "torch" in error_msg and "torchvision" not in error_msg:
        missing_packages.append("torch")
    if "torchvision" in error_msg:
        missing_packages.append("torchvision")
    
    if not missing_packages:
        missing_packages = ["torch", "torchvision", "pillow"]
        
    cmd = f"pip install {' '.join(missing_packages)}"
    msg = (
        f"偵測到缺少必要的 Python 套件！\n\n"
        f"錯誤訊息: {error_msg}\n"
        f"缺少套件: {', '.join(missing_packages)}\n\n"
        f"請在終端機中執行以下指令安裝套件：\n"
        f"{cmd}\n\n"
        f"安裝完成後請重新執行此程式。"
    )
    messagebox.showerror("缺少必要套件", msg)
    sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox

class CatDogClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 貓狗影像辨識系統")
        self.root.geometry("500x650")
        self.root.configure(bg="#0F172A") # Slate-900 質感深藍灰色底色
        self.root.resizable(False, False)
        
        self.model = None
        self.categories = None
        self.image_path = None
        self.tk_image = None
        
        # 設定 UI 介面
        self.setup_ui()
        
        # 非同步載入 AI 模型，避免 GUI 凍結
        self.async_load_model()
        
    def setup_ui(self):
        # 1. 頂部標題列
        title_label = tk.Label(
            self.root, 
            text="🐱 AI 貓狗影像辨識系統 🐶", 
            font=("Microsoft JhengHei", 18, "bold"),
            bg="#0F172A",
            fg="#F8FAFC"
        )
        title_label.pack(pady=20)
        
        # 2. 圖片預覽區域 (外框與預設提示)
        self.preview_frame = tk.Frame(
            self.root,
            width=340,
            height=340,
            bg="#1E293B", # Slate-800 卡片背景
            highlightbackground="#334155", # Slate-700 邊框
            highlightthickness=2
        )
        self.preview_frame.pack_propagate(False)
        self.preview_frame.pack(pady=10)
        
        self.preview_label = tk.Label(
            self.preview_frame,
            text="📸\n\n點選下方「選擇圖片」按鈕開始",
            font=("Microsoft JhengHei", 12),
            bg="#1E293B",
            fg="#94A3B8"
        )
        self.preview_label.pack(expand=True, fill="both")
        
        # 3. 按鈕控制區域
        self.btn_frame = tk.Frame(self.root, bg="#0F172A")
        self.btn_frame.pack(pady=20)
        
        # 選擇圖片按鈕 (藍色)
        self.select_btn = tk.Button(
            self.btn_frame,
            text="選擇圖片",
            font=("Microsoft JhengHei", 12, "bold"),
            bg="#3B82F6", # Blue-500
            fg="white",
            activebackground="#2563EB",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            state="disabled" # 在模型載入前先禁用
        )
        self.select_btn.pack(side="left", padx=8)
        self.select_btn.bind("<Enter>", lambda e: self.on_btn_hover(self.select_btn, "#2563EB", True))
        self.select_btn.bind("<Leave>", lambda e: self.on_btn_hover(self.select_btn, "#3B82F6", True))
        self.select_btn.config(command=self.select_image)
        
        # 下載 Kaggle 數據集按鈕 (紫色)
        self.download_btn = tk.Button(
            self.btn_frame,
            text="下載 Kaggle 數據集",
            font=("Microsoft JhengHei", 12, "bold"),
            bg="#8B5CF6", # Purple-500
            fg="white",
            activebackground="#7C3AED",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            state="disabled" # 在模型載入前先禁用
        )
        self.download_btn.pack(side="left", padx=8)
        self.download_btn.bind("<Enter>", lambda e: self.on_btn_hover(self.download_btn, "#7C3AED", True))
        self.download_btn.bind("<Leave>", lambda e: self.on_btn_hover(self.download_btn, "#8B5CF6", True))
        self.download_btn.config(command=self.async_download_kaggle)
        
        # 開始辨識按鈕 (綠色)
        self.predict_btn = tk.Button(
            self.btn_frame,
            text="開始辨識",
            font=("Microsoft JhengHei", 12, "bold"),
            bg="#10B981", # Emerald-500
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            state="disabled" # 在載入圖片前先禁用
        )
        self.predict_btn.pack(side="left", padx=8)
        self.predict_btn.bind("<Enter>", lambda e: self.on_btn_hover(self.predict_btn, "#059669"))
        self.predict_btn.bind("<Leave>", lambda e: self.on_btn_hover(self.predict_btn, "#10B981"))
        self.predict_btn.config(command=self.async_predict)
        
        # 4. 結果與進度顯示區域
        self.result_frame = tk.Frame(
            self.root,
            bg="#1E293B",
            padx=15,
            pady=15,
            highlightbackground="#334155",
            highlightthickness=1
        )
        self.result_frame.pack(fill="x", padx=40, pady=10)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="系統狀態：正在載入 AI 模型中...",
            font=("Microsoft JhengHei", 11),
            bg="#1E293B",
            fg="#F1F5F9",
            wraplength=380,
            justify="center"
        )
        self.result_label.pack()
        
        # 置信度自訂進度條 (Canvas)
        self.score_canvas = tk.Canvas(
            self.result_frame,
            width=380,
            height=12,
            bg="#334155",
            highlightthickness=0,
            bd=0
        )
        # 預設先隱藏進度條
        
        self.score_text = tk.Label(
            self.result_frame,
            text="",
            font=("Microsoft JhengHei", 9),
            bg="#1E293B",
            fg="#94A3B8"
        )
        # 預設先隱藏詳細文字描述

    def on_btn_hover(self, btn, color, check_model_loading=False):
        """處理按鈕懸停的高亮視覺效果"""
        if check_model_loading and self.model is None:
            return
        if btn["state"] == "normal":
            btn["background"] = color
            
    def async_load_model(self):
        """開啟新執行緒載入模型，防止 GUI 介面凍結"""
        thread = threading.Thread(target=self.load_model_worker)
        thread.daemon = True
        thread.start()
        
    def load_model_worker(self):
        try:
            # 使用 ResNet18 模型並下載官方預訓練權重 (首次執行需要網路連線)
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            self.model.eval()
            self.categories = weights.meta["categories"]
            
            # 載入成功，透過 Tkinter 的 after 方法安全地在主執行緒更新 UI
            self.root.after(0, self.model_loaded_callback)
        except Exception as e:
            # 載入失敗，更新 UI 並彈出視窗
            self.root.after(0, lambda: self.model_load_failed_callback(e))
            
    def model_loaded_callback(self):
        self.select_btn.config(state="normal")
        self.download_btn.config(state="normal")
        self.result_label.config(
            text="✅ AI 模型已成功載入！請選擇圖片或下載 Kaggle 數據集。",
            fg="#10B981"
        )
        
    def model_load_failed_callback(self, error):
        self.result_label.config(
            text=f"❌ AI 模型載入失敗！\n錯誤: {str(error)}\n請確認網路連線是否正常。",
            fg="#EF4444"
        )
        messagebox.showerror("模型載入失敗", f"無法載入預訓練模型。\n請確認您有網路連線（首次執行需要下載約 44MB 的模型權重）。\n\n錯誤詳情：\n{str(error)}")
        
    def select_image(self):
        """讓使用者選擇電腦中的影像檔案並顯示預覽"""
        file_path = filedialog.askopenfilename(
            filetypes=[("圖片檔案", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return
            
        self.image_path = file_path
        
        try:
            # 載入圖片並進行尺寸縮放以符合 320x320 預覽框，保留長寬比
            img = Image.open(self.image_path)
            img.thumbnail((320, 320))
            
            self.tk_image = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.tk_image, text="")
            
            # 啟用辨識按鈕
            self.predict_btn.config(state="normal")
            self.result_label.config(text="圖片載入成功！點擊「開始辨識」按鈕進行辨識。", fg="#F1F5F9")
            
            # 隱藏上一次的評估進度條與文字
            self.score_canvas.pack_forget()
            self.score_text.pack_forget()
        except Exception as e:
            messagebox.showerror("讀取圖片失敗", f"無法開啟此圖片檔案。\n錯誤: {str(e)}")
            
    def async_download_kaggle(self):
        """非同步下載 Kaggle 數據集，避免 GUI 凍結"""
        self.select_btn.config(state="disabled")
        self.download_btn.config(state="disabled")
        self.predict_btn.config(state="disabled")
        self.result_label.config(text="🌐 正在下載 Kaggle 數據集 (約 218MB)，請稍候...", fg="#F1F5F9")
        
        thread = threading.Thread(target=self.download_kaggle_worker)
        thread.daemon = True
        thread.start()
        
    def download_kaggle_worker(self):
        try:
            # 檢查是否有安裝 kaggle 庫
            try:
                import kaggle
            except ImportError:
                self.root.after(0, lambda: messagebox.showerror(
                    "缺少必要套件", 
                    "偵測到尚未安裝 kaggle Python 套件！\n\n請在終端機中執行：\npip install kaggle\n\n安裝完成後請重啟程式。"
                ))
                self.root.after(0, self.reset_buttons_callback)
                return
                
            # 驗證 Kaggle API 憑證
            try:
                kaggle.api.authenticate()
            except Exception as auth_err:
                msg = (
                    "Kaggle API 憑證驗證失敗！\n\n"
                    "請確認您已完成以下步驟：\n"
                    "1. 登入 Kaggle 並至 Account 頁面點選 'Create New API Token' 下載 kaggle.json。\n"
                    "2. 將 kaggle.json 檔案放置於您的使用者路徑，路徑通常為：\n"
                    "   C:\\Users\\您的使用者名稱\\.kaggle\\kaggle.json\n\n"
                    f"詳細錯誤訊息：\n{str(auth_err)}"
                )
                self.root.after(0, lambda: messagebox.showerror("Kaggle 憑證錯誤", msg))
                self.root.after(0, self.reset_buttons_callback)
                return
                
            # 下載 tongpython/cat-and-dog 公共貓狗數據集並自動解壓縮
            temp_dir = os.path.dirname(os.path.abspath(__file__))
            dest_dir = os.path.join(temp_dir, "kaggle_dataset")
            os.makedirs(dest_dir, exist_ok=True)
            
            # 使用 kaggle api 下載並自動解壓縮
            kaggle.api.dataset_download_files('tongpython/cat-and-dog', path=dest_dir, unzip=True)
            
            self.root.after(0, lambda: self.kaggle_download_success_callback(dest_dir))
            
        except Exception as e:
            self.root.after(0, lambda: self.kaggle_download_failed_callback(e))
            
    def kaggle_download_success_callback(self, dest_dir):
        self.select_btn.config(state="normal")
        self.download_btn.config(state="normal")
        self.result_label.config(text="✅ Kaggle 數據集下載完成！請點擊「選擇圖片」進行選取。", fg="#10B981")
        
        # 彈出成功提示
        messagebox.showinfo(
            "下載成功", 
            f"Kaggle 貓狗數據集已下載並解壓縮完成！\n"
            f"儲存路徑為：\n{dest_dir}\n\n"
            f"您現在可以點選「選擇圖片」，進入此資料夾選取圖片進行辨識。"
        )
        
    def kaggle_download_failed_callback(self, error):
        self.select_btn.config(state="normal")
        self.download_btn.config(state="normal")
        self.result_label.config(text="❌ Kaggle 數據集下載失敗！", fg="#EF4444")
        messagebox.showerror("下載失敗", f"無法下載 Kaggle 數據集。\n請確認您的網路連線與憑證是否正確。\n\n錯誤詳情：\n{str(error)}")
        
    def reset_buttons_callback(self):
        self.select_btn.config(state="normal")
        self.download_btn.config(state="normal")
        self.result_label.config(text="請選擇圖片或點選「下載 Kaggle 數據集」開始。", fg="#F1F5F9")
            
    def async_predict(self):
        """開啟新執行緒執行 AI 模型推論，防止 UI 凍結"""
        if not self.image_path or not self.model:
            return
            
        # 暫時禁用按鈕，避免重疊執行
        self.predict_btn.config(state="disabled")
        self.select_btn.config(state="disabled")
        self.result_label.config(text="🔮 正在進行 AI 分類辨識，請稍候...", fg="#F1F5F9")
        
        thread = threading.Thread(target=self.predict_worker)
        thread.daemon = True
        thread.start()
        
    def predict_worker(self):
        try:
            # 1. 影像預處理：符合 ResNet-18 標準輸入格式 (224x224, 影像標準化)
            img = Image.open(self.image_path).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0) # 加上 batch 維度 (1, 3, 224, 224)
            
            # 2. 模型推論
            with torch.no_grad():
                output = self.model(input_batch)
                
            # 3. 計算機率分布 (Softmax)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 4. 定義 ImageNet 中的狗與貓類別編號
            # 狗的類別主要集中在 151 到 268 ( domestic dogs )，以及部分狼/胡狼等野生物種 (269-275)
            DOG_CLASSES = set(range(151, 276))
            # 貓的類別集中在 281 到 285 ( domestic cats )，加上 286 (美洲獅)、287 (山貓) 等貓科
            CAT_CLASSES = set(range(281, 288))
            
            # 累加所有的狗類別機率與貓類別機率，比單一亞種機率更準確
            prob_dog = sum(probabilities[idx].item() for idx in DOG_CLASSES)
            prob_cat = sum(probabilities[idx].item() for idx in CAT_CLASSES)
            
            # 取得全類別最高預測值 (用於判定其他非貓狗類別)
            top_prob, top_idx = torch.max(probabilities, 0)
            top_idx = top_idx.item()
            top_prob = top_prob.item()
            
            # 5. 進行邏輯判定
            # 若貓狗累加機率都偏低 (例如 < 15%)，則回傳最可能的 ImageNet 類別名稱
            if prob_dog > prob_cat and prob_dog > 0.15:
                pred_class = "狗 (Dog)"
                confidence = prob_dog
                details = f"判定為犬科或狗狗品種 (累計信心度: {prob_dog:.1%})"
                color = "#3B82F6" # 藍色
            elif prob_cat > prob_dog and prob_cat > 0.15:
                pred_class = "貓 (Cat)"
                confidence = prob_cat
                details = f"判定為貓科或貓咪品種 (累計信心度: {prob_cat:.1%})"
                color = "#10B981" # 綠色
            else:
                class_name = self.categories[top_idx] if self.categories else f"Class {top_idx}"
                pred_class = "非貓狗物體"
                confidence = top_prob
                details = f"這可能不是貓狗。\n預測最接近：{class_name} (信心度: {top_prob:.1%})"
                color = "#EAB308" # 黃色
                
            # 回傳主執行緒更新 UI
            self.root.after(0, lambda: self.prediction_success_callback(pred_class, confidence, details, color))
            
        except Exception as e:
            self.root.after(0, lambda: self.prediction_failed_callback(e))
            
    def prediction_success_callback(self, pred_class, confidence, details, color):
        # 恢復按鈕狀態
        self.predict_btn.config(state="normal")
        self.select_btn.config(state="normal")
        
        # 顯示主要預測結果
        self.result_label.config(
            text=f"🎯 辨識結果：{pred_class}",
            fg=color
        )
        
        # 顯示並更新進度條
        self.score_canvas.pack(pady=(10, 0))
        self.score_canvas.delete("all")
        
        # 計算進度條寬度 (滿分為 380px)
        fill_width = int(380 * confidence)
        self.score_canvas.create_rectangle(
            0, 0, fill_width, 12,
            fill=color,
            width=0
        )
        
        # 顯示詳細文字
        self.score_text.pack(pady=(5, 0))
        self.score_text.config(text=details, fg="#94A3B8")
        
    def prediction_failed_callback(self, error):
        self.predict_btn.config(state="normal")
        self.select_btn.config(state="normal")
        self.result_label.config(text="❌ 辨識過程中發生錯誤！", fg="#EF4444")
        messagebox.showerror("辨識失敗", f"處理圖片時出錯。\n錯誤訊息: {str(error)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogClassifierApp(root)
    root.mainloop()
