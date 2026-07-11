import time
import threading
import sys
import ctypes
from collections import deque
from pynput import mouse, keyboard as pynput_keyboard

# 設定進程 DPI 感知，確保在不同縮放比例的螢幕下，座標讀取與重播完全精準
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# 定義狀態常數
STATE_IDLE = 0             # 閒置，等待按下 't' 開始錄製
STATE_RECORDING = 1        # 錄製中，記錄所有鍵盤與滑鼠操作
STATE_WAITING_5S = 2       # 倒數中，等待 5 秒後重播（此時按 'p' 可中斷）
STATE_REPLAYING = 3        # 重播中，循環播放錄製的動作（按 'p' 中斷）

# 全局控制變數
current_state = STATE_IDLE
recorded_events = []
recording_start_time = 0.0
last_move_time = 0.0
exiting = False

# 鎖，保護狀態寫入安全 (讀取狀態時不加鎖，以保證 Hook 回呼的執行效能，防止點擊遺漏)
state_lock = threading.Lock()

# Windows API 滑鼠事件常數 (用於高相容性的點擊與滾輪模擬)
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

# 模擬按鍵與滑鼠控制器
mouse_ctrl = mouse.Controller()
kb_ctrl = pynput_keyboard.Controller()

# 錄製監聽器 (全域常駐運行，避免重複註冊/註銷導致的 Hook 損壞)
mouse_listener = None
kb_listener = None

# 重播背景執行緒
replay_thread = None

# 用來過濾重播引擎產生的模擬事件的佇列 (避免自發性觸發熱鍵邏輯)
simulated_queue = deque()

# 用於追蹤在重播時目前已被按下的按鍵，以利在中斷時自動釋放，避免卡鍵
pressed_keys_during_replay = set()
pressed_mouse_buttons_during_replay = set()

def is_admin():
    """確認當前是否具備系統管理員權限"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False

print("==================================================")
print("             鍵盤與滑鼠 錄製與重複重播工具 (v2.1)")
print("==================================================")
print("使用流程：")
print("  1. 開啟程式後，系統處於「閒置狀態」")
print("  2. 在任意視窗按下 't' 鍵 -> 開始「錄製狀態」")
print("     - 將記錄所有鍵盤按鍵與滑鼠點擊、滾輪、移動的位置與時間點")
print("  3. 再次在任意視窗按下 't' 鍵 -> 結束錄製")
print("  4. 系統將進入「5 秒倒數」準備自動執行重播")
print("  5. 倒數完畢後，系統將無限「重複重播」您的操作")
print("控制熱鍵：")
print("  - [p]   ：停止重播或取消倒數，重置回「閒置狀態」")
print("  - [ESC] ：結束並關閉程式")
print("--------------------------------------------------")
if not is_admin():
    print("【⚠️ 權限警告】程式目前「未以系統管理員身分」執行！")
    print("  - 若在 Roblox 等遊戲中重播時「滑鼠無法點擊」或「鍵盤沒反應」，")
    print("    這是 Windows UIPI 權限保護機制的限制。")
    print("  - 請【關閉此視窗】，然後【右鍵點擊執行您的終端機/命令提示字元】，")
    print("    選擇【以系統管理員身分執行】後再重新啟動此程式！")
else:
    print("【已取得系統管理員權限】遊戲相容性最佳。")
print("==================================================")
print("系統狀態：已就緒。請在目標視窗按下 [t] 鍵開始錄製。")

def get_key_repr(key):
    """獲取按鍵的字串代表，統一字元與特殊按鍵格式"""
    if hasattr(key, 'char') and key.char is not None:
        return key.char.lower()
    elif hasattr(key, 'name') and key.name is not None:
        return key.name
    else:
        return str(key)

# ----------------- 錄製監聽回呼 (無鎖高速度執行) -----------------

def on_mouse_move(x, y):
    """滑鼠移動錄製回呼，精確捕捉連續的相對視角移動"""
    global current_state, recording_start_time, last_move_time
    if current_state != STATE_RECORDING:
        return
    t = time.perf_counter() - recording_start_time
    # 限流機制：滑鼠移動採樣頻率限制在最高 200Hz (每 5ms 一次)，防止消息堵塞導致滑鼠點擊被 Windows 拋棄
    if t - last_move_time < 0.005:
        return
    last_move_time = t
    recorded_events.append({
        'type': 'mouse_move',
        'time': t,
        'x': x,
        'y': y
    })

def on_mouse_click(x, y, button, pressed):
    """滑鼠點擊錄製回呼 (不使用 Lock 避免 OS 訊息超時阻斷監聽)"""
    global current_state, recording_start_time
    if current_state != STATE_RECORDING:
        return
    t = time.perf_counter() - recording_start_time
    recorded_events.append({
        'type': 'mouse_click',
        'time': t,
        'x': x,
        'y': y,
        'button': button.name,
        'pressed': pressed
    })
    # 同步在主控台印出點擊日誌
    action_type = "按下" if pressed else "放開"
    print(f"[錄製] 點擊滑鼠: {button.name} ({action_type}) -> 座標 ({x}, {y})")

def on_mouse_scroll(x, y, dx, dy):
    """滑鼠滾輪錄製回呼"""
    global current_state, recording_start_time
    if current_state != STATE_RECORDING:
        return
    t = time.perf_counter() - recording_start_time
    recorded_events.append({
        'type': 'mouse_scroll',
        'time': t,
        'x': x,
        'y': y,
        'dx': dx,
        'dy': dy
    })
    # 同步在主控台印出滾輪日誌
    direction = "向上" if dy > 0 else "向下"
    print(f"[錄製] 滑鼠滾輪: {direction} (dx: {dx}, dy: {dy}) -> 座標 ({x}, {y})")

def on_keyboard_press(key):
    """鍵盤按下錄製與熱鍵監聽回呼"""
    global current_state, recording_start_time, simulated_queue
    if exiting:
        return
    
    key_repr = get_key_repr(key)
    
    # 檢查是否為重播模擬輸出的控制鍵事件，如果是則過濾掉
    if key_repr in ('t', 'p'):
        if simulated_queue:
            expected_type, expected_name = simulated_queue[0]
            if key_repr == expected_name and expected_type == 'down':
                simulated_queue.popleft()
                return  # 直接忽略模擬事件
                
    # 僅響應按下事件，處理狀態切換
    if key_repr == 't':
        if current_state == STATE_IDLE:
            trigger_start_recording()
        elif current_state == STATE_RECORDING:
            trigger_stop_recording()
        return
    elif key_repr == 'p':
        if current_state in (STATE_WAITING_5S, STATE_REPLAYING):
            trigger_stop_replay()
        return
    elif key_repr == 'esc':
        exit_script()
        return
        
    # 錄製模式下的正常按鍵按下
    if current_state == STATE_RECORDING:
        t = time.perf_counter() - recording_start_time
        mx, my = mouse_ctrl.position
        recorded_events.append({
            'type': 'key_press',
            'time': t,
            'key': key_repr,
            'x': mx,
            'y': my
        })
        print(f"[錄製] 按下鍵盤: {key_repr}")

def on_keyboard_release(key):
    """鍵盤放開錄製回呼"""
    global current_state, recording_start_time, simulated_queue
    if exiting:
        return
        
    key_repr = get_key_repr(key)
    
    # 檢查是否為重播模擬輸出的控制鍵事件，如果是則過濾掉
    if key_repr in ('t', 'p'):
        if simulated_queue:
            expected_type, expected_name = simulated_queue[0]
            if key_repr == expected_name and expected_type == 'up':
                simulated_queue.popleft()
                return  # 直接忽略模擬事件
                
    # 錄製模式下的正常按鍵放開
    if current_state == STATE_RECORDING:
        # 過濾掉 't' (因為按下 't' 用於停止錄製，放開也不應被錄下)
        if key_repr == 't':
            return
        t = time.perf_counter() - recording_start_time
        mx, my = mouse_ctrl.position
        recorded_events.append({
            'type': 'key_release',
            'time': t,
            'key': key_repr,
            'x': mx,
            'y': my
        })
        print(f"[錄製] 放開鍵盤: {key_repr}")

# ----------------- 重播與模擬 -----------------

def release_all_keys_and_buttons():
    """防卡鍵：釋放所有目前可能正被模擬按下的鍵盤與滑鼠按鍵"""
    # 釋放鍵盤按鍵
    for key in list(pressed_keys_during_replay):
        try:
            kb_ctrl.release(key)
        except Exception:
            pass
    pressed_keys_during_replay.clear()
    
    # 釋放滑鼠按鍵 (使用 ctypes 進行安全釋放)
    for btn_name in list(pressed_mouse_buttons_during_replay):
        try:
            if btn_name == 'left':
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif btn_name == 'right':
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            elif btn_name == 'middle':
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
        except Exception:
            pass
    pressed_mouse_buttons_during_replay.clear()

def execute_event(event):
    """將單個錄製的事件轉化為系統模擬輸入 (使用 ctypes 點擊模擬以維持最高遊戲相容性)"""
    global simulated_queue
    try:
        if event['type'] == 'mouse_click':
            button_name = event['button']
            pressed = event['pressed']
            
            # 確保游標定位完成 (加入微小延遲讓作業系統與遊戲視窗更新游標狀態)
            time.sleep(0.01)
            
            # 使用高相容性的 Win32 API 進行按下/放開，可穿透大多數 3D 遊戲引擎的點擊檢測
            if button_name == 'left':
                flag = MOUSEEVENTF_LEFTDOWN if pressed else MOUSEEVENTF_LEFTUP
            elif button_name == 'right':
                flag = MOUSEEVENTF_RIGHTDOWN if pressed else MOUSEEVENTF_RIGHTUP
            elif button_name == 'middle':
                flag = MOUSEEVENTF_MIDDLEDOWN if pressed else MOUSEEVENTF_MIDDLEUP
            else:
                return
                
            # 發送低階滑鼠點擊事件 (作用於當前游標處)
            ctypes.windll.user32.mouse_event(flag, 0, 0, 0, 0)
            
            if pressed:
                pressed_mouse_buttons_during_replay.add(button_name)
            else:
                pressed_mouse_buttons_during_replay.discard(button_name)
                
            # 同步印出重播模擬日誌
            action_type = "按下" if pressed else "放開"
            print(f"[重播] 模擬點擊滑鼠: {button_name} ({action_type})")
                
        elif event['type'] == 'mouse_scroll':
            dy = event['dy']
            # 標準滾動單次捲動量值為 120
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(dy * 120), 0)
            print(f"[重播] 模擬滑鼠滾輪: {'向上' if dy > 0 else '向下'}")
            
        elif event['type'] in ('key_press', 'key_release'):
            key_repr = event['key']
            
            # 解析按鍵物件
            if key_repr in pynput_keyboard.Key.__members__:
                key = pynput_keyboard.Key[key_repr]
            elif len(key_repr) == 1:
                key = key_repr
            elif key_repr.startswith('<') and key_repr.endswith('>'):
                try:
                    vk_code = int(key_repr[1:-1])
                    key = pynput_keyboard.KeyCode.from_vk(vk_code)
                except ValueError:
                    key = key_repr
            else:
                key = key_repr
                
            # 若模擬的是控制熱鍵 't' 或 'p'，將其加入模擬佇列，防止 Hook 判定為使用者物理按下
            if key_repr in ('t', 'p'):
                expected_type = 'down' if event['type'] == 'key_press' else 'up'
                simulated_queue.append((expected_type, key_repr))
                
            if event['type'] == 'key_press':
                kb_ctrl.press(key)
                pressed_keys_during_replay.add(key)
                print(f"[重播] 模擬按下鍵盤: {key_repr}")
            else:
                kb_ctrl.release(key)
                pressed_keys_during_replay.discard(key)
                print(f"[重播] 模擬放開鍵盤: {key_repr}")
    except Exception as e:
        print(f"[警告] 模擬事件執行失敗: {e}")

# ----------------- 狀態切換與流程控制 -----------------

def trigger_start_recording():
    """啟動錄製流程"""
    global current_state, recording_start_time, recorded_events
    with state_lock:
        if current_state != STATE_IDLE:
            return
            
        current_state = STATE_RECORDING
        recorded_events.clear()
        recording_start_time = time.perf_counter()
        
        print("\n[狀態] ===== 開始錄製 =====")
        print("[說明] 請在目標視窗中進行您的操作，系統將精確記錄您的按鍵、點擊與滾動。再次按下 [t] 結束錄製...")
        print("==========================")

def trigger_stop_recording():
    """結束錄製流程"""
    global current_state, replay_thread
    with state_lock:
        if current_state != STATE_RECORDING:
            return
            
        current_state = STATE_WAITING_5S
            
        print("\n[狀態] ===== 錄製已結束 =====")
        print(f"[統計] 共錄製了 {len(recorded_events)} 個操作事件。")
        
        # 啟動背景重播與倒數執行緒
        replay_thread = threading.Thread(target=countdown_and_replay_loop, daemon=True)
        replay_thread.start()

def trigger_stop_replay():
    """中止重播，重置為閒置"""
    global current_state
    with state_lock:
        if current_state not in (STATE_WAITING_5S, STATE_REPLAYING):
            return
            
        current_state = STATE_IDLE
        # 狀態一旦變更，重播執行緒會自動跳出循環並安全釋放按鍵
        print("\n[狀態] ===== 已中斷並重置 =====")
        print("[提示] 系統已回到閒置狀態。按下 [t] 鍵可重新開始錄製。")

def countdown_and_replay_loop():
    """5秒倒數與重播主迴圈 (於背景執行緒運行)"""
    global current_state
    
    # 5 秒倒數邏輯，每 50 毫秒檢查一次狀態，以利即時響應 'p' 鍵中斷
    countdown_cancelled = False
    for i in range(5, 0, -1):
        print(f"[系統] 距離開始重複重播還有 {i} 秒... (可按 [p] 鍵取消)")
        start_wait = time.perf_counter()
        while time.perf_counter() - start_wait < 1.0:
            with state_lock:
                if current_state != STATE_WAITING_5S:
                    countdown_cancelled = True
                    break
            time.sleep(0.05)
        if countdown_cancelled:
            break
            
    if countdown_cancelled:
        return
        
    with state_lock:
        if current_state != STATE_WAITING_5S:
            return
        current_state = STATE_REPLAYING
        print("\n[狀態] ===== 開始重播 =====")
        print("[說明] 正在重複執行錄製的操作中... 按下 [p] 鍵可隨時結束並回到閒置狀態。")
        print("==========================")
        
    # 無限重複重播主迴圈
    while True:
        with state_lock:
            if current_state != STATE_REPLAYING:
                break
                
        if not recorded_events:
            print("[警告] 錄製事件列表為空，無法重播。系統回到閒置狀態。")
            with state_lock:
                current_state = STATE_IDLE
            break
            
        print(f"[重播] 開始執行操作循環 (共 {len(recorded_events)} 個事件)...")
        
        # 初始化重播起點座標，並記錄基準位置
        first_event = recorded_events[0]
        try:
            # 起始對齊：在重播一輪的開頭進行絕對定位，以防整體偏移
            mouse_ctrl.position = (first_event['x'], first_event['y'])
        except Exception:
            pass
        last_recorded_x = first_event['x']
        last_recorded_y = first_event['y']
        
        start_playback_time = time.perf_counter()
        
        for event in recorded_events:
            with state_lock:
                if current_state != STATE_REPLAYING:
                    break
            
            # 高精度等待直至事件預定的播放時間點
            target_time = start_playback_time + event['time']
            while time.perf_counter() < target_time:
                with state_lock:
                    if current_state != STATE_REPLAYING:
                        break
                # 微小 sleep 讓出 CPU 資源
                time.sleep(0.002)
                
            with state_lock:
                if current_state != STATE_REPLAYING:
                    break
            
            # 計算該事件相對上一個記錄事件的滑鼠偏移量 (dx, dy)
            dx = event['x'] - last_recorded_x
            dy = event['y'] - last_recorded_y
            
            # 1. 使用相對移動模擬視角改變 (支援 3D 遊戲引擎的相對滑鼠捕獲機制，如 Roblox)
            if dx != 0 or dy != 0:
                try:
                    mouse_ctrl.move(dx, dy)
                except Exception:
                    pass
            
            # 2. 強制設定絕對座標 (給 2D UI 點擊，修正相對移動累積造成的偏差，確保游標準確點在按鈕上)
            try:
                mouse_ctrl.position = (event['x'], event['y'])
            except Exception:
                pass
            
            last_recorded_x = event['x']
            last_recorded_y = event['y']
            
            # 執行此事件本身的其他模擬 (點擊、滾輪或按鍵)
            execute_event(event)
            
        # 每個重播循環之間微小停頓
        time.sleep(0.1)
        
    # 離開重播迴圈後，防卡鍵釋放
    release_all_keys_and_buttons()

def exit_script():
    """安全退出程式"""
    global exiting, current_state, mouse_listener, kb_listener
    with state_lock:
        exiting = True
        current_state = STATE_IDLE
        
        # 停止全域監聽器
        if mouse_listener:
            mouse_listener.stop()
        if kb_listener:
            kb_listener.stop()
            
        print("\n[狀態] 正在安全關閉程式...")
        
    release_all_keys_and_buttons()
    time.sleep(0.2)
    sys.exit(0)

# ----------------- 全域監聽器初始化與啟動 -----------------

# 建立並啟動全域常駐監聽器 (避免頻繁開啟 Hook 導致衝突與遺漏)
mouse_listener = mouse.Listener(
    on_move=on_mouse_move,
    on_click=on_mouse_click,
    on_scroll=on_mouse_scroll
)
kb_listener = pynput_keyboard.Listener(
    on_press=on_keyboard_press,
    on_release=on_keyboard_release
)

mouse_listener.start()
kb_listener.start()

try:
    # 主執行緒保持執行
    while not exiting:
        time.sleep(0.5)
except KeyboardInterrupt:
    exit_script()