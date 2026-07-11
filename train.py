ㄍimport os
import sys
import urllib.request
import zipfile
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. 確保數據集存在，若不存在則自動下載
def prepare_dataset(dest_dir):
    # 檢查是否已存在任何一個數據集目錄
    for folder in ["cats_and_dogs_filtered", "dogscats"]:
        dataset_path = os.path.join(dest_dir, folder)
        if os.path.exists(dataset_path):
            print(f"[Info] 偵測到數據集 {folder} 已存在，跳過下載步驟。")
            return dataset_path

    os.makedirs(dest_dir, exist_ok=True)
    
    # 嘗試下載第一源：Google 託管的精簡版數據集 (68MB)
    url_google = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_path = os.path.join(dest_dir, "cats_and_dogs_filtered.zip")
    
    try:
        print("[Download] 正在下載貓狗數據集精簡版 (約 68MB)...")
        req = urllib.request.Request(url_google, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024 * 64
            with open(zip_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        sys.stdout.write(f"\r下載進度: {percent}%")
                        sys.stdout.flush()
        print("\n[Extract] 下載完成！正在解壓縮數據集...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        try:
            os.remove(zip_path)
        except Exception:
            pass
        print("[Success] 數據集準備就緒！")
        return os.path.join(dest_dir, "cats_and_dogs_filtered")
        
    except Exception as e:
        print(f"\n[Warning] 精簡版下載失敗 (可能是網路阻擋): {e}")
        print("[Download] 正在切換至備用下載點 (完整版約 839MB，請耐心等候)...")
        
        url_backup = "http://files.fast.ai/data/examples/dogscats.tgz"
        tgz_path = os.path.join(dest_dir, "dogscats.tgz")
        
        try:
            req = urllib.request.Request(url_backup, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                block_size = 1024 * 64
                with open(tgz_path, 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            sys.stdout.write(f"\r下載進度: {percent}%")
                            sys.stdout.flush()
            print("\n[Extract] 下載完成！正在解壓 .tgz 數據集...")
            with tarfile.open(tgz_path, 'r:gz') as tar_ref:
                tar_ref.extractall(dest_dir)
            try:
                os.remove(tgz_path)
            except Exception:
                pass
            print("[Success] 備用數據集準備就緒！")
            return os.path.join(dest_dir, "dogscats")
        except Exception as err:
            print(f"\n[Error] 備用下載點也下載失敗: {err}")
            raise err

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(script_dir, "kaggle_dataset")
    
    # 準備數據集
    dataset_path = prepare_dataset(dest_dir)
    
    train_dir = os.path.join(dataset_path, "train")
    
    # 根據不同數據集決定驗證集路徑
    if os.path.exists(os.path.join(dataset_path, "validation")):
        val_dir = os.path.join(dataset_path, "validation")
    elif os.path.exists(os.path.join(dataset_path, "valid")):
        val_dir = os.path.join(dataset_path, "valid")
    else:
        val_dir = os.path.join(dataset_path, "validation")
    
    # 2. 數據增強與載入
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("[Data] 載入圖片數據...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"訓練樣本數: {len(train_dataset)}，類別: {train_dataset.classes}")
    print(f"驗證樣本數: {len(val_dataset)}，類別: {val_dataset.classes}")
    print("類別對應標籤：0 -> 貓 (cats), 1 -> 狗 (dogs)")
    
    # 3. 建立 ResNet-18 模型並進行微調 (Transfer Learning)
    print("[Model] 載入 ResNet-18 預訓練模型並修改分類器...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # 凍結所有特徵提取層的參數，不參與梯度更新
    for param in model.parameters():
        param.requires_grad = False
        
    # 將最後的 Fully Connected 層改為輸出 2 類別
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    
    # 4. 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    # 僅優化 model.fc 層的參數，加速訓練且保持特徵層不變
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 5. 開始訓練迴圈
    epochs = 3
    print("[Train] 開始訓練模型 (共 3 個 Epoch)...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # 驗證步驟
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
              
    # 6. 儲存模型權重
    model_save_path = os.path.join(script_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"[Save] 自訂模型訓練完成！權重已儲存至：{model_save_path}")

if __name__ == "__main__":
    main()
