import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import mediapipe as mp

class GestureModelTrainer:
    def __init__(self, data_dir='../data', model_dir='../models', img_size=(128, 128), num_classes=9):
        """
        初始化手势识别模型训练器
        
        Args:
            data_dir: 数据目录
            model_dir: 模型保存目录
            img_size: 图像大小
            num_classes: 手势类别数量
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.img_size = img_size
        self.num_classes = num_classes
        
        # 确保模型目录存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 初始化MediaPipe手部检测器
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
    def preprocess_image(self, image_path, use_mediapipe=True):
        """
        预处理图像，可选用MediaPipe提取手部特征
        
        Args:
            image_path: 图像路径
            use_mediapipe: 是否使用MediaPipe提取手部特征
            
        Returns:
            处理后的图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        if use_mediapipe:
            # 使用MediaPipe提取手部特征
            with self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                # 转换为RGB格式（MediaPipe需要）
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                # 创建一个空白图像来绘制手部关键点
                h, w, c = image.shape
                hand_image = np.zeros((h, w, c), dtype=np.uint8)
                
                # 如果检测到手，绘制手部关键点
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            hand_image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
                    
                    # 裁剪手部区域
                    gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
                    if np.sum(gray) > 0:  # 确保有手部内容
                        # 找到非零像素的位置
                        coords = cv2.findNonZero(gray)
                        x, y, w, h = cv2.boundingRect(coords)
                        # 添加一些边距
                        padding = 20
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(image.shape[1] - x, w + 2 * padding)
                        h = min(image.shape[0] - y, h + 2 * padding)
                        hand_image = hand_image[y:y+h, x:x+w]
                    
                    # 调整大小
                    processed_image = cv2.resize(hand_image, self.img_size)
                else:
                    # 如果没有检测到手，使用原始图像
                    processed_image = cv2.resize(image, self.img_size)
        else:
            # 不使用MediaPipe，直接调整大小
            processed_image = cv2.resize(image, self.img_size)
        
        # 归一化
        processed_image = processed_image / 255.0
        
        return processed_image
    
    def load_data(self, use_mediapipe=True, test_split=0.2):
        """
        加载数据集
        
        Args:
            use_mediapipe: 是否使用MediaPipe提取手部特征
            test_split: 测试集比例
            
        Returns:
            (x_train, y_train), (x_test, y_test): 训练集和测试集
        """
        images = []
        labels = []
        
        for class_id in range(1, self.num_classes + 1):
            class_dir = os.path.join(self.data_dir, str(class_id))
            if not os.path.exists(class_dir):
                print(f"警告: 类别 {class_id} 的目录不存在")
                continue
                
            print(f"加载类别 {class_id} 的数据...")
            files = os.listdir(class_dir)
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(class_dir, file)
                    processed_image = self.preprocess_image(image_path, use_mediapipe)
                    
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(class_id - 1)  # 类别从0开始
        
        if not images:
            raise ValueError("没有找到有效的图像数据")
            
        # 转换为numpy数组
        X = np.array(images)
        y = np.array(labels)
        
        # 分割训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
        
        # 转换为独热编码
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        print(f"数据加载完成: 训练集 {x_train.shape[0]} 样本, 测试集 {x_test.shape[0]} 样本")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """
        构建卷积神经网络模型
        
        Returns:
            构建的模型
        """
        model = Sequential([
            # 第一个卷积层块
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_size[0], self.img_size[1], 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # 第二个卷积层块
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # 第三个卷积层块
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # 全连接层
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型构建完成:")
        model.summary()
        
        return model
    
    def train(self, epochs=50, batch_size=32, use_mediapipe=True, use_augmentation=True):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批量大小
            use_mediapipe: 是否使用MediaPipe提取手部特征
            use_augmentation: 是否使用数据增强
            
        Returns:
            训练历史
        """
        # 加载数据
        (x_train, y_train), (x_test, y_test) = self.load_data(use_mediapipe=use_mediapipe)
        
        # 构建模型
        model = self.build_model()
        
        # 设置回调
        model_path = os.path.join(self.model_dir, 'gesture_model.h5')
        checkpoint = ModelCheckpoint(
            model_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping]
        
        # 数据增强
        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            datagen.fit(x_train)
            
            # 训练模型使用数据增强
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=callbacks
            )
        else:
            # 不使用数据增强训练模型
            history = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=callbacks
            )
        
        # 评估模型
        print("\n评估模型在测试集上的表现:")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # 保存模型
        model.save(os.path.join(self.model_dir, 'gesture_model_final.h5'))
        print(f"模型已保存到 {self.model_dir}")
        
        # 绘制训练过程图
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """
        绘制训练过程图
        
        Args:
            history: 训练历史
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制准确率
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 绘制损失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.show()

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    
    print("手势识别模型训练程序")
    print("\n选项:")
    print("1. 使用MediaPipe提取手部特征进行训练 (推荐)")
    print("2. 不使用MediaPipe直接训练")
    
    choice = input("\n请选择 (1 或 2): ")
    
    use_mediapipe = True if choice == '1' else False
    
    print("\n开始训练...")
    trainer.train(use_mediapipe=use_mediapipe, epochs=50)
