import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import time
from PIL import Image, ImageDraw, ImageFont

# 辅助函数，用于绘制中文文本
def put_chinese_text(img, text, position, font_path="C:/Windows/Fonts/simhei.ttf", font_size=30, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文文本
    
    Args:
        img: OpenCV图像
        text: 要绘制的文本
        position: 位置坐标 (x, y)
        font_path: 字体文件路径
        font_size: 字体大小
        color: 字体颜色 (B, G, R)
    
    Returns:
        绘制完成的图像
    """
    if isinstance(img, np.ndarray):
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        return img
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img_pil)
    
    # 创建字体对象
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 如果找不到指定字体，尝试使用默认字体
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB顺序而OpenCV使用BGR顺序
    
    # 转换回来
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

class GestureRecognizer:
    def __init__(self, model_path=None, img_size=(128, 128), num_classes=9, confidence_threshold=0.8):
        """
        初始化手势识别器
        
        Args:
            model_path: 模型路径，如果为None，将尝试加载默认模型
            img_size: 图像大小
            num_classes: 类别数量
            confidence_threshold: 置信度阈值，低于该值的预测将被忽略
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # 初始化MediaPipe手部检测器
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 尝试加载模型
        if model_path is None:
            # 尝试加载默认模型路径
            default_paths = [
                '../models/gesture_model.h5',
                '../models/gesture_model_final.h5'
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            self.model = load_model(model_path)
        else:
            print("警告: 没有找到有效的模型文件，请先训练模型")
            self.model = None
        
        # 手势标签
        self.gesture_labels = {
            0: "手势1: 全屏截图",
            1: "手势2: 音量增加",
            2: "手势3: 音量减少",
            3: "手势4: 静音/取消静音",
            4: "手势5: 打开浏览器",
            5: "手势6: 打开任务管理器",
            6: "手势7: 锁屏",
            7: "手势8: 打开计算器",
            8: "手势9: 打开系统菜单"
        }
        
        # 初始化手势跟踪
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0  # 手势冷却时间（秒）
        self.stable_gesture_count = 0
        self.required_stable_frames = 5  # 需要连续识别到相同手势的帧数
        self.current_stable_gesture = None
    
    def preprocess_frame(self, frame):
        """
        预处理帧，提取手部特征
        
        Args:
            frame: 输入帧
            
        Returns:
            processed_image: 处理后的图像
            hand_landmarks: 手部关键点
            hand_bbox: 手部边界框 (x, y, w, h)
        """
        if frame is None:
            return None, None, None
        
        # 创建帧的副本以便绘制
        display_frame = frame.copy()
        
        # 转换为RGB格式（MediaPipe需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(frame_rgb)
        
        # 初始化返回值
        processed_image = None
        hand_landmarks = None
        hand_bbox = None
        
        # 如果检测到手部
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # 使用第一个手部
            
            # 创建手部图像
            h, w, c = frame.shape
            hand_image = np.zeros((h, w, c), dtype=np.uint8)
            
            # 绘制手部关键点
            self.mp_drawing.draw_landmarks(
                hand_image, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
            # 绘制手部关键点到原始帧
            self.mp_drawing.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS)
            
            # 获取手部边界框
            gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            if np.sum(gray) > 0:  # 如果有手部
                # 获取非零像素点
                coords = cv2.findNonZero(gray)
                x, y, w, h = cv2.boundingRect(coords)
                
                # 添加边界框
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                # 保存手部边界框
                hand_bbox = (x, y, w, h)
                
                # 获取手部图像
                hand_image = hand_image[y:y+h, x:x+w]
                
                # 缩放图像
                processed_image = cv2.resize(hand_image, self.img_size)
                
                # 归一化图像
                processed_image = processed_image / 255.0
        
        return processed_image, hand_landmarks, hand_bbox, display_frame
    
    def predict_gesture(self, frame):
        """
        预测手势
        
        Args:
            frame: 输入帧
            
        Returns:
            gesture_id: 手势ID
            confidence: 置信度
            display_frame: 绘制后的帧
        """
        if self.model is None:
            return None, 0, frame
        
        # 预处理帧
        processed_image, hand_landmarks, hand_bbox, display_frame = self.preprocess_frame(frame)
        
        # 如果没有检测到手部
        if processed_image is None:
            return None, 0, display_frame
        
        # 预测手势
        prediction = self.model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
        gesture_id = np.argmax(prediction)
        confidence = prediction[gesture_id]
        
        # 如果置信度低于阈值
        if confidence < self.confidence_threshold:
            gesture_id = None
        
        # 绘制手部边界框
        if hand_bbox is not None:
            x, y, w, h = hand_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制手势文本
            if gesture_id is not None:
                gesture_text = self.gesture_labels.get(gesture_id, f"未知手势: {gesture_id}")
                conf_text = f"置信度: {confidence:.2f}"
                
                # 根据置信度设置颜色
                if confidence > 0.9:
                    color = (0, 255, 0)  # 绿色（高置信度）
                elif confidence > 0.7:
                    color = (0, 255, 255)  # 黄色（中置信度）
                else:
                    color = (0, 0, 255)  # 红色（低置信度）
                
                display_frame = put_chinese_text(display_frame, gesture_text, (x, y - 30), font_size=30, color=color)
                display_frame = put_chinese_text(display_frame, conf_text, (x, y + h + 30), font_size=25, color=color)
        
        return gesture_id, confidence, display_frame
    
    def process_frame_with_tracking(self, frame):
        """
        处理帧并跟踪手势
        
        Args:
            frame: 输入帧
            
        Returns:
            stable_gesture: 稳定的手势ID（如果有）
            display_frame: 绘制后的帧
        """
        # 预测手势
        gesture_id, confidence, display_frame = self.predict_gesture(frame)
        
        # 绘制当前手势文本
        if self.current_stable_gesture is not None:
            display_frame = put_chinese_text(display_frame, 
                      f"当前手势: {self.gesture_labels.get(self.current_stable_gesture, '')}", 
                      (10, 30), font_size=30, color=(0, 255, 0))
        
        # 如果没有检测到手部或置信度低于0.8
        if gesture_id is None or confidence < 0.8:
            # 如果置信度低于0.8，显示提示文本
            if gesture_id is not None:
                display_frame = put_chinese_text(display_frame, 
                          f"置信度过低: {confidence:.2f} < 0.8", 
                          (10, 150), font_size=25, color=(0, 0, 255))
            self.stable_gesture_count = 0
            return None, display_frame
        
        # 如果当前手势与之前的手势相同
        if self.last_gesture == gesture_id:
            self.stable_gesture_count += 1
        else:
            # 如果手势改变
            self.stable_gesture_count = 1
            self.last_gesture = gesture_id
        
        # 绘制稳定帧数文本
        display_frame = put_chinese_text(display_frame, 
                  f"稳定帧数: {self.stable_gesture_count}/{self.required_stable_frames}", 
                  (10, 70), font_size=25, color=(0, 255, 255))
        
        # 绘制当前置信度文本
        display_frame = put_chinese_text(display_frame, 
                  f"当前置信度: {confidence:.2f}", 
                  (10, 110), font_size=25, color=(0, 255, 255))
        
        # 检查是否达到稳定帧数
        stable_gesture = None
        current_time = time.time()
        
        if self.stable_gesture_count >= self.required_stable_frames:
            # 检查冷却时间
            if (current_time - self.last_gesture_time) >= self.gesture_cooldown:
                stable_gesture = gesture_id
                self.current_stable_gesture = gesture_id
                self.last_gesture_time = current_time
                self.stable_gesture_count = 0  # 重置稳定帧数
            else:
                # 绘制冷却时间文本
                remaining = self.gesture_cooldown - (current_time - self.last_gesture_time)
                display_frame = put_chinese_text(display_frame, 
                          f"冷却时间: {remaining:.1f}s", 
                          (10, 150), font_size=25, color=(0, 0, 255))
        
        return stable_gesture, display_frame
    
    def run_detection(self, camera_index=0, window_name="手势识别", quit_key='q'):
        """
        运行手势识别
        
        Args:
            camera_index: 相机索引
            window_name: 窗口名称
            quit_key: 退出键
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("无法打开相机")
            return
        
        print("按 'q' 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break
            
            # 翻转帧
            frame = cv2.flip(frame, 1)
            
            # 处理帧并跟踪手势
            gesture_id, display_frame = self.process_frame_with_tracking(frame)
            
            # 显示帧
            cv2.imshow(window_name, display_frame)
            
            # 如果检测到稳定的手势
            if gesture_id is not None:
                print(f"检测到稳定的手势: {self.gesture_labels.get(gesture_id, '')}")
                # 在这里添加你想要的动作
            
            # 检查退出键
            key = cv2.waitKey(1)
            if key == ord(quit_key):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建手势识别器
    recognizer = GestureRecognizer()
    
    # 运行手势识别
    recognizer.run_detection()
