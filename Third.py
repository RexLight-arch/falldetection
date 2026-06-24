import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import paho.mqtt.client as mqtt
import os
from ais_bench.infer.interface import InferSession
import subprocess  # 添加subprocess模块用于FFmpeg推流

# -------- MQTT 配置 --------
MQTT_BROKER = "47.120.65.85"
MQTT_PORT = 1883
MQTT_TOPIC = "CHome/sub/camera/alarm"

# 创建MQTT客户端并连接
client = mqtt.Client()
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("MQTT连接成功")
except Exception as e:
    print(f"MQTT连接失败: {str(e)}")

# -------- FFmpeg 推流配置 --------
RTMP_URL = "rtmp://47.120.65.85:1935/live/stream"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

# 创建FFmpeg推流进程
ffmpeg_process = subprocess.Popen([
    'ffmpeg', 
    '-y', 
    '-f', 'rawvideo', 
    '-vcodec', 'rawvideo', 
    '-pix_fmt', 'bgr24', 
    '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}', 
    '-r', str(FPS), 
    '-i', '-', 
    '-c:v', 'libx264', 
    '-pix_fmt', 'yuv420p', 
    '-preset', 'veryfast', 
    '-f', 'flv', 
    RTMP_URL
], stdin=subprocess.PIPE)

print("FFmpeg推流进程已启动")

# -------- MediaPipe 初始化 --------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -------- YOLOv12 NPU 检测器 --------
class NPUYOLODetector:
    def __init__(self, model_path):
        self.session = InferSession(device_id=0, model_path=model_path)
        self.input_shape = (640, 640)
        print("NPU模型加载成功！")
    
    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.input_shape[0]/h, self.input_shape[1]/w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(frame, (new_w, new_h))
        
        padded_img = np.full((self.input_shape[0], self.input_shape[1], 3), 114, dtype=np.uint8)
        padded_img[:new_h, :new_w] = img
        
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        padded_img = padded_img.transpose(2, 0, 1).astype(np.float32)
        padded_img = np.expand_dims(padded_img, axis=0)
        padded_img /= 255.0
        return padded_img, (h, w), (new_h, new_w), scale
    
    def postprocess(self, outputs, orig_shape, new_shape, scale):
        orig_h, orig_w = orig_shape
        new_h, new_w = new_shape
        detections = []
        
        if len(outputs) > 0 and outputs[0].shape[0] == 1 and outputs[0].shape[1] == 84:
            output = outputs[0][0]
            
            conf_mask = output[4, :] > 0.25
            scores = output[4, :][conf_mask]
            class_ids = np.argmax(output[5:85, :][:, conf_mask], axis=0)
            boxes = output[:4, :][:, conf_mask].T
            
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            x1 = (x_center - width / 2) / self.input_shape[1] * orig_w
            y1 = (y_center - height / 2) / self.input_shape[0] * orig_h
            x2 = (x_center + width / 2) / self.input_shape[1] * orig_w
            y2 = (y_center + height / 2) / self.input_shape[0] * orig_h
            
            x1 = np.clip(x1 * scale, 0, orig_w)
            y1 = np.clip(y1 * scale, 0, orig_h)
            x2 = np.clip(x2 * scale, 0, orig_w)
            y2 = np.clip(y2 * scale, 0, orig_h)
            
            for i in range(len(scores)):
                if class_ids[i] == 0 and scores[i] > 0.5:
                    # 计算高宽比用于姿势判断
                    bbox_height = y2[i] - y1[i]
                    bbox_width = x2[i] - x1[i]
                    aspect_ratio = bbox_height / (bbox_width + 1e-6)
                    
                    detections.append({
                        'box': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                        'conf': float(scores[i]),
                        'class_id': int(class_ids[i]),
                        'label': self.class_names[int(class_ids[i])],
                        'aspect_ratio': aspect_ratio
                    })
        
        return detections
    
    def predict(self, frame):
        try:
            input_data, orig_shape, new_shape, scale = self.preprocess(frame)
            outputs = self.session.infer([input_data])
            return self.postprocess(outputs, orig_shape, new_shape, scale)
        except Exception as e:
            print(f"NPU推理错误: {str(e)}")
            return []
    
    @property
    def class_names(self):
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'chair', 'couch', 'bed', 'tv', 'laptop'
        ]

# 初始化NPU YOLO检测器
yolo_model = NPUYOLODetector("/home/HwHiAiUser/Desktop/yolov12n_310b4.om")

# -------- 跌倒检测参数 --------
HEIGHT_RATIO_THRESHOLD = 0.35
TORSO_LEG_ANGLE_THRESHOLD = 100
HEAD_FLOOR_THRESHOLD = 0.7
VELOCITY_THRESHOLD = -3.0
MIN_CONFIDENCE = 0.4
COOLDOWN_PERIOD = 5
MIN_FALL_DURATION = 0.8
WINDOW_SIZE = 15  # 时间窗口大小
VOTE_THRESHOLD = 10  # 投票阈值

# 状态机定义
STATE_NORMAL = 0
STATE_FALLING = 1
STATE_FALLEN = 2
STATE_RISING = 3

# 动态基线存储
baseline = {
    'height_ratio': 0.7,
    'torso_z_angle': 90,
    'shoulder_width': 0
}

# -------- 数据窗口 --------
landmark_history = deque(maxlen=WINDOW_SIZE)
vote_history = deque(maxlen=WINDOW_SIZE)  # 用于投票机制

# -------- 计算函数 --------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def is_landmark_visible(landmark):
    return landmark.visibility > MIN_CONFIDENCE

def extract_features(landmarks, shape):
    h, w = shape[:2]
    
    # 关键点可见性检查
    visible_points = {
        'shoulder_l': is_landmark_visible(landmarks[11]),
        'shoulder_r': is_landmark_visible(landmarks[12]),
        'hip_l': is_landmark_visible(landmarks[23]),
        'hip_r': is_landmark_visible(landmarks[24]),
        'knee_l': is_landmark_visible(landmarks[25]),
        'knee_r': is_landmark_visible(landmarks[26]),
        'ankle_l': is_landmark_visible(landmarks[27]),
        'ankle_r': is_landmark_visible(landmarks[28]),
        'head': is_landmark_visible(landmarks[0]),
        'ear_l': is_landmark_visible(landmarks[7]),
        'ear_r': is_landmark_visible(landmarks[8]),
    }
    
    try:
        # 计算关键点位置（考虑可见性）
        shoulder_l = [landmarks[11].x * w, landmarks[11].y * h] if visible_points['shoulder_l'] else None
        shoulder_r = [landmarks[12].x * w, landmarks[12].y * h] if visible_points['shoulder_r'] else None
        
        # 肩部中点计算
        if shoulder_l and shoulder_r:
            shoulder = [(shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2]
        elif shoulder_l:
            shoulder = shoulder_l
        elif shoulder_r:
            shoulder = shoulder_r
        else:
            return None

        hip_l = [landmarks[23].x * w, landmarks[23].y * h] if visible_points['hip_l'] else None
        hip_r = [landmarks[24].x * w, landmarks[24].y * h] if visible_points['hip_r'] else None
        
        # 髋部中点计算
        if hip_l and hip_r:
            hip = [(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2]
        elif hip_l:
            hip = hip_l
        elif hip_r:
            hip = hip_r
        else:
            return None

        # 膝盖和脚踝
        knee = []
        if visible_points['knee_l']:
            knee.append([landmarks[25].x * w, landmarks[25].y * h])
        if visible_points['knee_r']:
            knee.append([landmarks[26].x * w, landmarks[26].y * h])
        knee = np.mean(knee, axis=0) if knee else None

        ankle = []
        if visible_points['ankle_l']:
            ankle.append([landmarks[27].x * w, landmarks[27].y * h])
        if visible_points['ankle_r']:
            ankle.append([landmarks[28].x * w, landmarks[28].y * h])
        ankle = np.mean(ankle, axis=0) if ankle else None

        # 头部位置
        if visible_points['head']:
            head_ref = [landmarks[0].x * w, landmarks[0].y * h]
        elif visible_points['ear_l'] and visible_points['ear_r']:
            head_ref = [(landmarks[7].x + landmarks[8].x) / 2 * w, 
                        (landmarks[7].y + landmarks[8].y) / 2 * h]
        elif visible_points['ear_l']:
            head_ref = [landmarks[7].x * w, landmarks[7].y * h]
        elif visible_points['ear_r']:
            head_ref = [landmarks[8].x * w, landmarks[8].y * h]
        else:
            return None

        # 计算特征
        height_ratio = abs(shoulder[1] - hip[1]) / h
        
        # 躯干-腿部角度
        torso_leg_angle = calculate_angle(shoulder, hip, knee) if knee is not None else 90
        
        head_floor_dist = head_ref[1] / h
        
        # 肩宽计算
        if shoulder_l and shoulder_r:
            shoulder_width = np.linalg.norm(np.array(shoulder_r) - np.array(shoulder_l))
        else:
            shoulder_width = baseline['shoulder_width'] if baseline['shoulder_width'] > 0 else 100
        
        # 踝宽计算
        ankle_width = None
        if visible_points['ankle_l'] and visible_points['ankle_r']:
            ankle_width = np.linalg.norm(
                np.array([landmarks[28].x * w, landmarks[28].y * h]) - 
                np.array([landmarks[27].x * w, landmarks[27].y * h])
            )
        
        # 躯干Z轴角度（髋-肩-脚踝）
        torso_z_angle = calculate_angle(shoulder, hip, ankle) if ankle is not None else 90
        
        # 躯干地面角度（髋-肩向量与垂直方向的夹角）
        torso_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
        vertical_vector = np.array([0, -1])  # 垂直向下
        cos_angle = np.dot(torso_vector, vertical_vector) / (np.linalg.norm(torso_vector) + 1e-6)
        torso_vertical_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # 更新基线（仅当人站立时更新）
        if height_ratio > 0.6 and torso_vertical_angle < 20:
            baseline['height_ratio'] = baseline['height_ratio'] * 0.9 + height_ratio * 0.1
            baseline['torso_z_angle'] = baseline['torso_z_angle'] * 0.9 + torso_z_angle * 0.1
            if shoulder_l and shoulder_r:
                baseline['shoulder_width'] = baseline['shoulder_width'] * 0.9 + shoulder_width * 0.1

        return {
            'shoulder': shoulder, 'hip': hip, 'knee': knee, 'ankle': ankle,
            'head': head_ref, 'height_ratio': height_ratio,
            'torso_leg_angle': torso_leg_angle,
            'head_floor_dist': head_floor_dist,
            'shoulder_width': shoulder_width,
            'ankle_width': ankle_width,
            'torso_z_angle': torso_z_angle,
            'torso_vertical_angle': torso_vertical_angle,
            'visible': visible_points
        }
    except Exception as e:
        print(f"特征提取错误: {str(e)}")
        return None

def compute_vertical_velocity(curr, prev):
    return curr['head'][1] - prev['head'][1] if curr and prev else 0

def detect_fall(current, previous, state):
    if not current: 
        return False
    
    flags = 0
    
    # 动态调整阈值（基于基线）
    dynamic_height_threshold = HEIGHT_RATIO_THRESHOLD * (baseline['height_ratio'] / 0.7)
    dynamic_torso_angle_threshold = TORSO_LEG_ANGLE_THRESHOLD * (baseline['torso_z_angle'] / 90)
    
    # 1. 高度比例特征
    if current['height_ratio'] < dynamic_height_threshold:
        flags += 1
    
    # 2. 躯干-腿部角度特征
    if current['torso_leg_angle'] > dynamic_torso_angle_threshold:
        flags += 1
    
    # 3. 头部离地距离
    if current['head_floor_dist'] > HEAD_FLOOR_THRESHOLD:  # 值越大越接近底部
        flags += 1
    
    # 4. 垂直速度（仅当有前一帧时）
    if previous and compute_vertical_velocity(current, previous) < VELOCITY_THRESHOLD:
        flags += 1
    
    # 5. 躯干Z轴角度（倒地时角度变大）
    if current['torso_z_angle'] > 140:
        flags += 1
    
    # 6. 踝宽大于肩宽（倒地时腿部分开）
    if current.get('ankle_width') is not None:
        if current['ankle_width'] > current['shoulder_width'] * 1.5:
            flags += 1
    
    # 7. 躯干垂直角度（倒地时躯干倾斜）
    if current['torso_vertical_angle'] > 60:
        flags += 1
    
    # 状态相关阈值调整
    if state == STATE_NORMAL:
        threshold = 4  # 正常状态需要更多证据
    elif state == STATE_FALLING:
        threshold = 3  # 跌倒中状态降低要求
    else:
        threshold = 5  # 其他状态提高要求
    
    return flags >= threshold

def detect_posture_yolo(detections):
    """通过YOLO检测框的高宽比判断姿势"""
    for det in detections:
        if det['class_id'] == 0:  # 只处理人物检测
            aspect_ratio = det['aspect_ratio']
            
            # 使用高宽比判断姿势（优化阈值）
            if aspect_ratio < 0.7: 
                return "lying"
            elif aspect_ratio > 1.0:  # 降低站立阈值
                return "standing"
            elif aspect_ratio > 0.8: 
                return "sitting"
    
    return "unknown"

def is_rising(current_features):
    """判断是否正在起身"""
    if not current_features:
        return False
    
    # 起身特征判断条件
    rising_conditions = [
        current_features['height_ratio'] > 0.5,  # 高度比例增加
        current_features['torso_vertical_angle'] < 45,  # 身体更直立
        current_features.get('ankle_width', 0) < current_features['shoulder_width'] * 1.2  # 脚踝宽度正常
    ]
    
    return sum(rising_conditions) >= 2  # 满足2个条件即判定为起身

# -------- 主循环 --------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# 设置设备权限
os.system("sudo chmod 777 /dev/davinci*")

frame_count = 0
prev_features = None
last_time = time.time()

# 状态机变量
current_state = STATE_NORMAL
last_state_change = time.time()
fall_alert_sent = False
rise_counter = 0
last_mqtt_attempt_time = 0  # 记录上次MQTT尝试发送的时间

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    current_time = time.time()
    fps = 1.0 / (current_time - last_time) if current_time - last_time > 0 else 0
    last_time = current_time
    
    # 每3帧进行一次目标检测
    if frame_count % 3 == 0:
        try:
            detections = yolo_model.predict(frame)
            
            # 绘制检测结果
            for det in detections:
                x1, y1, x2, y2 = det['box']
                label = det['label']
                conf = det['conf']
                
                # 只绘制高置信度的人物框
                if det['class_id'] == 0 and conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"检测处理错误: {str(e)}")
    
    # 姿态检测
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    
    current_features = None
    mediapipe_fall = False
    
    if result.pose_landmarks:
        # 绘制关键点
        mp_drawing.draw_landmarks(
            frame, 
            result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        # 提取特征
        current_features = extract_features(result.pose_landmarks.landmark, frame.shape)
        landmark_history.append(current_features)
        
        if current_features and prev_features:
            # 检测跌倒
            mediapipe_fall = detect_fall(current_features, prev_features, current_state)
            vote_history.append(1 if mediapipe_fall else 0)
        
        # 更新前一帧特征
        prev_features = current_features
    else:
        vote_history.append(0)
    
    # 使用YOLO检测姿势
    yolo_posture = detect_posture_yolo(detections) if 'detections' in locals() else "unknown"
    
    # 状态机处理
    prev_state = current_state
    
    # 1. 正常状态检测
    if current_state == STATE_NORMAL:
        # 投票检测到跌倒（时间窗口内满足阈值）
        if sum(vote_history) >= VOTE_THRESHOLD:
            current_state = STATE_FALLING
            last_state_change = current_time
        # YOLO检测到躺卧
        elif yolo_posture == "lying":
            current_state = STATE_FALLING
            last_state_change = current_time
    
    # 2. 跌倒中状态
    elif current_state == STATE_FALLING:
        # 持续跌倒超过1秒
        if current_time - last_state_change > 1.0:
            current_state = STATE_FALLEN
            last_state_change = current_time
        # 中途恢复 - 添加MediaPipe起身检测
        elif (yolo_posture == "standing" or 
             (current_features and is_rising(current_features))):
            current_state = STATE_NORMAL
            last_state_change = current_time
    
    # 3. 已跌倒状态
    elif current_state == STATE_FALLEN:
        # 检测到起身 - 添加MediaPipe起身检测
        if (yolo_posture == "standing" or 
           (current_features and is_rising(current_features))):
            current_state = STATE_RISING
            last_state_change = current_time
            rise_counter = 0
        
        # 长时间躺卧（超过5秒）且未发送警报
        elif current_time - last_state_change > 5.0 and not fall_alert_sent:
            # 在本地显示警报状态
            cv2.putText(frame, "FALL DETECTED!", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(frame, (20, 20), (620, 100), (0, 0, 255), 3)
            
            # 每1秒尝试发送一次MQTT消息
            if current_time - last_mqtt_attempt_time >= 1.0:
                try:
                    print("[ALERT] Fall detected! Sending MQTT message...")
                    client.publish(MQTT_TOPIC, '{"fall":"fall_down"}')
                    fall_alert_sent = True  # 标记为已发送
                    print("MQTT消息发送成功")
                except Exception as e:
                    print(f"发送MQTT消息失败: {str(e)}")
                finally:
                    last_mqtt_attempt_time = current_time  # 更新上次尝试时间
    
    # 4. 起身中状态
    elif current_state == STATE_RISING:
        # 添加起身维持检测
        is_maintaining_rise = (yolo_posture == "standing" or 
                              (current_features and is_rising(current_features)))
        
        if is_maintaining_rise:
            rise_counter += 1
        else:
            rise_counter = max(0, rise_counter - 2)  # 未维持起身状态时快速回退
        
        # 持续站立超过1秒
        if rise_counter > FPS:  # 约1秒
            current_state = STATE_NORMAL
            fall_alert_sent = False  # 重置警报状态
            last_state_change = current_time
        # 起身失败，返回已跌倒状态
        elif rise_counter <= 0:
            current_state = STATE_FALLEN
            last_state_change = current_time
    
    # 添加状态超时机制：长时间正常状态保持
    if current_state == STATE_NORMAL and current_time - last_state_change > 10.0:
        # 重置基线数据，适应当前姿势
        if current_features:
            baseline['height_ratio'] = baseline['height_ratio'] * 0.7 + current_features['height_ratio'] * 0.3
            baseline['torso_z_angle'] = baseline['torso_z_angle'] * 0.7 + current_features['torso_z_angle'] * 0.3
            baseline['shoulder_width'] = baseline['shoulder_width'] * 0.7 + current_features['shoulder_width'] * 0.3
        last_state_change = current_time  # 重置计时器

    # 显示状态信息
    state_text = {
        STATE_NORMAL: "NORMAL",
        STATE_FALLING: "FALLING",
        STATE_FALLEN: "FALLEN",
        STATE_RISING: "RISING"
    }.get(current_state, "UNKNOWN")
    
    cv2.putText(frame, f"State: {state_text}", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 显示性能信息
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示状态信息
    status_color = (0, 0, 255) if current_state in [STATE_FALLING, STATE_FALLEN] else (0, 255, 0)
    status_text = "ALERT!" if current_state in [STATE_FALLING, STATE_FALLEN] else "Monitoring"
    cv2.putText(frame, f"Status: {status_text}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # 显示YOLO姿势检测结果
    cv2.putText(frame, f"Posture: {yolo_posture}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 推流处理
    try:
        # 将处理后的帧写入FFmpeg进程
        ffmpeg_process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg管道已断开，尝试重新连接...")
        # 尝试重新启动FFmpeg进程
        try:
            ffmpeg_process = subprocess.Popen([
                'ffmpeg', 
                '-y', 
                '-f', 'rawvideo', 
                '-vcodec', 'rawvideo', 
                '-pix_fmt', 'bgr24', 
                '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}', 
                '-r', str(FPS), 
                '-i', '-', 
                '-c:v', 'libx264', 
                '-pix_fmt', 'yuv420p', 
                '-preset', 'veryfast', 
                '-f', 'flv', 
                RTMP_URL
            ], stdin=subprocess.PIPE)
            print("FFmpeg进程已重启")
            ffmpeg_process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"重启FFmpeg失败: {str(e)}")
    except Exception as e:
        print(f"推流错误: {str(e)}")
    
    # 显示本地画面
    cv2.imshow("Fall Detection", frame)
    
    # 退出检测
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC或Q键退出
        break
        
    frame_count += 1

# 清理资源
cap.release()
cv2.destroyAllWindows()

# 关闭FFmpeg进程
try:
    if ffmpeg_process.poll() is None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print("FFmpeg进程已关闭")
except Exception as e:
    print(f"关闭FFmpeg进程时出错: {str(e)}")

print("程序正常退出")