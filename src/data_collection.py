import cv2
import os
import numpy as np
import time

class DataCollector:
    def __init__(self, data_dir='../data', num_classes=9, samples_per_class=100):
        """
        初始化数据收集器
        
        Args:
            data_dir: 数据保存目录
            num_classes: 手势类别数量（1-9）
            samples_per_class: 每个类别的样本数量
        """
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # 为每个类别创建目录
        for i in range(1, num_classes + 1):
            class_dir = os.path.join(data_dir, str(i))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    def collect_data(self):
        """
        从摄像头收集手势数据
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        for class_id in range(1, self.num_classes + 1):
            print(f"\n准备收集手势 {class_id} 的数据")
            print("请做好准备，3秒后开始...")
            time.sleep(3)
            
            count = 0
            while count < self.samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取帧")
                    break
                
                # 显示当前帧和计数
                cv2.putText(frame, f"手势 {class_id}: {count}/{self.samples_per_class}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('数据收集', frame)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):  # 按's'保存当前帧
                    # 保存图像
                    filename = os.path.join(self.data_dir, str(class_id), f"{count}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"保存: {filename}")
                    count += 1
                    time.sleep(0.5)  # 短暂延迟，避免重复保存
            
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("数据收集完成")
    
    def collect_data_auto(self, frames_per_second=5, duration_per_class=20):
        """
        自动从摄像头收集手势数据
        
        Args:
            frames_per_second: 每秒捕获的帧数
            duration_per_class: 每个类别的收集时间（秒）
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        for class_id in range(1, self.num_classes + 1):
            print(f"\n准备收集手势 {class_id} 的数据")
            print(f"请在摄像头前做手势 {class_id}，{duration_per_class}秒后自动进入下一个手势")
            print("3秒后开始...")
            
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            print("开始收集!")
            
            count = 0
            max_samples = min(self.samples_per_class, frames_per_second * duration_per_class)
            start_time = time.time()
            
            while count < max_samples and (time.time() - start_time) < duration_per_class:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取帧")
                    break
                
                # 显示当前帧和计数
                remaining = duration_per_class - int(time.time() - start_time)
                cv2.putText(frame, f"手势 {class_id}: {count}/{max_samples} 剩余时间: {remaining}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('数据收集', frame)
                
                # 每隔一定时间保存一帧
                if count < max_samples and (time.time() - start_time) > (count / frames_per_second):
                    filename = os.path.join(self.data_dir, str(class_id), f"{count}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"保存: {filename}")
                    count += 1
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("数据收集完成")

if __name__ == "__main__":
    collector = DataCollector()
    print("选择数据收集模式:")
    print("1. 手动模式 (按's'保存帧)")
    print("2. 自动模式 (自动保存帧)")
    
    mode = input("请输入选择 (1 或 2): ")
    
    if mode == '1':
        collector.collect_data()
    elif mode == '2':
        collector.collect_data_auto()
    else:
        print("无效选择")
