import os
import sys
import cv2
import numpy as np
import time

# 确认可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_camera():
    """
    测试摄像头是否可用
    """
    print("\n测试摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[失败] 无法打开摄像头")
        return False
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[失败] 无法从摄像头读取帧")
        cap.release()
        return False
    
    print("[成功] 摄像头工作正常")
    print(f"      分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    cap.release()
    return True

def test_dependencies():
    """
    测试依赖项是否已安装
    """
    print("\n测试依赖项...")
    dependencies = {
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "TensorFlow": "tensorflow",
        "Keras": "keras",
        "Matplotlib": "matplotlib",
        "PyAutoGUI": "pyautogui",
        "MediaPipe": "mediapipe",
        "PyWin32": "win32api"
    }
    
    all_passed = True
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"[成功] {name} 已安装")
        except ImportError:
            print(f"[失败] {name} 未安装")
            all_passed = False
    
    return all_passed

def test_directories():
    """
    测试必要的目录是否存在
    """
    print("\n测试目录结构...")
    directories = [
        "data",
        "models",
        "src"
    ]
    
    all_passed = True
    for directory in directories:
        dir_path = os.path.join(parent_dir, directory)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"[成功] 目录 {directory} 存在")
        else:
            print(f"[失败] 目录 {directory} 不存在")
            all_passed = False
    
    return all_passed

def test_source_files():
    """
    测试源代码文件是否存在
    """
    print("\n测试源代码文件...")
    source_files = [
        "data_collection.py",
        "train_model.py",
        "gesture_recognition.py",
        "system_commands.py",
        "gui.py",
        "main.py"
    ]
    
    all_passed = True
    for file in source_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"[成功] 文件 {file} 存在")
        else:
            print(f"[失败] 文件 {file} 不存在")
            all_passed = False
    
    return all_passed

def test_mediapipe_hands():
    """
    测试MediaPipe手部检测是否正常工作
    """
    print("\n测试MediaPipe手部检测...")
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        
        # 测试单帧图像
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        results = hands.process(test_image)
        
        print("[成功] MediaPipe手部检测正常工作")
        return True
    except Exception as e:
        print(f"[失败] MediaPipe手部检测测试失败: {str(e)}")
        return False

def run_all_tests():
    """
    运行所有测试
    """
    print("\n===== 运行所有测试 =====")
    
    # 运行测试
    deps_passed = test_dependencies()
    dirs_passed = test_directories()
    files_passed = test_source_files()
    camera_passed = test_camera()
    mediapipe_passed = test_mediapipe_hands()
    
    # 显示测试结果
    print("\n===== 测试结果汇总 =====")
    tests = [
        ("依赖项检查", deps_passed),
        ("目录结构检查", dirs_passed),
        ("源代码文件检查", files_passed),
        ("摄像头检查", camera_passed),
        ("MediaPipe手部检测检查", mediapipe_passed)
    ]
    
    for test_name, passed in tests:
        status = "成功" if passed else "失败"
        print(f"{test_name}: [{status}]")
    
    # 显示总体结果
    all_passed = all([deps_passed, dirs_passed, files_passed, camera_passed, mediapipe_passed])
    if all_passed:
        print("\n所有测试均已通过!您可以正常运行程序。")
    else:
        print("\n部分测试未通过。请检查错误信息并解决问题后再次运行测试。")

if __name__ == "__main__":
    run_all_tests()
    input("\n按回车键退出...")
