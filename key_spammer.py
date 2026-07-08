import time
import threading
import sys
import keyboard

# 定義狀態常數
STATE_IDLE = 0             # 閒置，等待按下 't' 開始偵測
STATE_WAITING_FOR_W = 1     # 已按 't'，等待按下 'w'
STATE_MEASURING_W = 2       # 已按 'w'，正在計時中（直到放開 'w'）
STATE_RUNNING = 3           # 計時結束，自動化循環運行中

# 全局控制變數
current_state = STATE_IDLE
running = False
exiting = False
start_time = 0.0
duration = 0.0

# 鎖，用以保護狀態存取
status_lock = threading.Lock()

print("==================================================")
print("             鍵盤自動化按鍵模擬程式 (高精度自訂時間版)")
print("==================================================")
print("使用流程：")
print("  1. 開啟程式後，系統處於「閒置狀態」")
print("  2. 在任意視窗按下 't' 鍵 -> 進入「準備偵測狀態」")
print("  3. 按住 'w' 鍵 -> 開始計時")
print("  4. 放開 'w' 鍵 -> 結束計時")
print("  5. 系統將自動執行：")
print("     - 用剛才按住 'w' 的相同時間長度，自動按住 's'，再按住 'w'，循環往復")
print("     - 同時在背景不斷快速連點 'e'")
print("控制熱鍵：")
print("  - [F8]  ：停止自動化並重置回「閒置狀態」")
print("  - [ESC] ：結束並關閉程式")
print("==================================================")
print("系統狀態：已就緒。請按 [t] 鍵開始偵測。")

def release_all_keys():
    """安全釋放所有模擬按鍵，避免卡鍵"""
    try:
        keyboard.release('w')
        keyboard.release('s')
        keyboard.release('e')
    except Exception:
        pass

def reset_to_idle():
    """重置回閒置狀態，停止自動化"""
    global current_state, running
    with status_lock:
        if exiting:
            return
        current_state = STATE_IDLE
        running = False
        release_all_keys()
        print("\n[狀態] 已停止並重置。系統回到閒置狀態。請重新按 [t] 開始偵測。")

def exit_script():
    """安全結束程式"""
    global running, exiting, current_state
    with status_lock:
        exiting = True
        running = False
        current_state = STATE_IDLE
        print("\n[狀態] 正在關閉程式，釋放按鍵並登出...")
        release_all_keys()
    time.sleep(0.2)
    sys.exit(0)

def on_key_event(event):
    """處理鍵盤事件，用來偵測 t 和 w"""
    global current_state, start_time, duration, running
    
    # 效能優化：如果已經在自動化運行中，直接跳過，不佔用鎖與 CPU
    if current_state == STATE_RUNNING:
        return

    with status_lock:
        if exiting:
            return
            
        key_name = event.name.lower() if event.name else ""
            
        # 1. 處於閒置狀態，等待按 't'
        if current_state == STATE_IDLE:
            if key_name == 't' and event.event_type == 'down':
                current_state = STATE_WAITING_FOR_W
                print("\n[偵測] 偵測到按下 't'！進入準備偵測狀態。")
                print("       請在您需要的目標視窗中，按下並「按住」 'w' 鍵開始計時測量...")
                
        # 2. 準備偵測狀態，等待按住 'w'
        elif current_state == STATE_WAITING_FOR_W:
            if key_name == 'w' and event.event_type == 'down':
                start_time = time.perf_counter()
                current_state = STATE_MEASURING_W
                print("[偵測] 'w' 鍵已按下，計時中...")
                
        # 3. 正在計時狀態，等待放開 'w'
        elif current_state == STATE_MEASURING_W:
            if key_name == 'w' and event.event_type == 'up':
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                # 避免極短按的誤判（小於 0.1 秒視為無效）
                if duration < 0.1:
                    current_state = STATE_WAITING_FOR_W
                    print("[偵測] 按下時間過短，請重新按住 'w' 鍵...")
                else:
                    current_state = STATE_RUNNING
                    running = True
                    print(f"[偵測] 'w' 鍵已放開！計時結束。")
                    print(f"[系統] 測量時間為：{duration:.4f} 秒。")
                    print(f"[系統] 開始自動執行循環：按 s ({duration:.4f} 秒) -> 按 w ({duration:.4f} 秒)...")
                    print("       背景已同步啟動連點 'e'。鍵盤隨時可按 [F8] 停止並重置。")

def precise_sleep(target_duration):
    """高精度睡眠控制，解決 time.sleep() 在 Windows 上的累積誤差與不精準問題"""
    start = time.perf_counter()
    while True:
        if current_state != STATE_RUNNING or not running or exiting:
            return False
        
        elapsed = time.perf_counter() - start
        rem = target_duration - elapsed
        if rem <= 0:
            break
        
        # 如果剩餘時間大於 15ms，則進行短暫睡眠讓出 CPU 資源
        if rem > 0.015:
            time.sleep(0.005)
        else:
            # 剩餘時間極短時，使用微秒級自旋鎖（Spin Lock）以確保絕對精準度
            pass
            
    return True

def e_spammer_thread():
    """自動連點 'e' 執行緒"""
    while not exiting:
        if current_state == STATE_RUNNING and running:
            keyboard.press('e')
            time.sleep(0.05)
            keyboard.release('e')
            time.sleep(0.1)  # 連點間隔
        else:
            time.sleep(0.1)

def movement_thread():
    """控制自動 w/s 循環的執行緒"""
    global current_state, running
    while not exiting:
        if current_state == STATE_RUNNING and running:
            # --- 1. 自動按住 's' 鍵 ---
            if current_state != STATE_RUNNING or not running or exiting:
                continue
            print(f"[動作] 自動按住 's' 鍵 {duration:.4f} 秒...")
            keyboard.press('s')
            
            # 使用高精度睡眠
            success = precise_sleep(duration)
            
            keyboard.release('s')
            print("[動作] 釋放 's' 鍵")
            
            # --- 2. 自動按住 'w' 鍵 ---
            if not success or current_state != STATE_RUNNING or not running or exiting:
                continue
            print(f"[動作] 自動按住 'w' 鍵 {duration:.4f} 秒...")
            keyboard.press('w')
            
            # 使用高精度睡眠
            success = precise_sleep(duration)
            
            keyboard.release('w')
            print("[動作] 釋放 'w' 鍵")
            
        else:
            time.sleep(0.1)

# 註冊鍵盤鉤子，監聽所有實體按鍵事件
keyboard.hook(on_key_event)

# 註冊控制熱鍵（不會干涉一般的輸入）
keyboard.add_hotkey('f8', reset_to_idle)
keyboard.add_hotkey('esc', exit_script)

# 啟動背景執行緒
t_e = threading.Thread(target=e_spammer_thread, daemon=True)
t_move = threading.Thread(target=movement_thread, daemon=True)

t_e.start()
t_move.start()

try:
    # 主執行緒等待結束
    while not exiting:
        time.sleep(0.5)
except KeyboardInterrupt:
    exit_script()