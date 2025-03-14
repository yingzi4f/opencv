import os
import sys
import platform
import importlib.util

def check_python_version():
    """检查Python版本"""
    print("\n检查Python版本...")
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")
    
    # 检查是否为Python 3.8或更高版本
    major, minor, _ = map(int, python_version.split('.'))
    if major >= 3 and minor >= 8:
        print("✓ Python版本符合要求")
        return True
    else:
        print("✗ Python版本不符合要求，需要Python 3.8或更高版本")
        return False

def check_dependencies():
    """检查依赖项是否已安装"""
    print("\n检查依赖项...")
    dependencies = {
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "TensorFlow": "tensorflow",
        "Keras": "keras",
        "Matplotlib": "matplotlib",
        "PyAutoGUI": "pyautogui",
        "MediaPipe": "mediapipe",
        "PyWin32": "win32api",
        "PIL": "PIL"
    }
    
    all_installed = True
    for name, module_name in dependencies.items():
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "__version__"):
                    version = module.__version__
                else:
                    version = "未知版本"
                print(f"✓ {name} 已安装 (版本: {version})")
            else:
                print(f"✗ {name} 未安装")
                all_installed = False
        except ImportError:
            print(f"✗ {name} 未安装或导入出错")
            all_installed = False
    
    return all_installed

def check_camera():
    """检查摄像头是否可用"""
    print("\n检查摄像头...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ 摄像头工作正常 (分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
                cap.release()
                return True
            else:
                print("✗ 无法从摄像头读取图像")
                cap.release()
                return False
        else:
            print("✗ 无法打开摄像头")
            return False
    except Exception as e:
        print(f"✗ 检查摄像头时出错: {str(e)}")
        return False

def check_directories():
    """检查必要的目录结构"""
    print("\n检查目录结构...")
    required_dirs = ["src", "data", "models"]
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ 目录 {directory} 存在")
        else:
            print(f"✗ 目录 {directory} 不存在")
            all_exist = False
    
    return all_exist

def check_source_files():
    """检查源代码文件"""
    print("\n检查源代码文件...")
    required_files = [
        os.path.join("src", "main.py"),
        os.path.join("src", "data_collection.py"),
        os.path.join("src", "train_model.py"),
        os.path.join("src", "gesture_recognition.py"),
        os.path.join("src", "system_commands.py"),
        os.path.join("src", "gui.py")
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✓ 文件 {file_path} 存在")
        else:
            print(f"✗ 文件 {file_path} 不存在")
            all_exist = False
    
    return all_exist

def main():
    print("===== 手势识别系统环境检查 =====")
    
    # 检查Python版本
    python_ok = check_python_version()
    
    # 检查依赖项
    deps_ok = check_dependencies()
    
    # 检查摄像头
    camera_ok = check_camera()
    
    # 检查目录结构
    dirs_ok = check_directories()
    
    # 检查源代码文件
    files_ok = check_source_files()
    
    # 汇总结果
    print("\n===== 检查结果汇总 =====")
    checks = [
        ("Python版本", python_ok),
        ("依赖项", deps_ok),
        ("摄像头", camera_ok),
        ("目录结构", dirs_ok),
        ("源代码文件", files_ok)
    ]
    
    for name, result in checks:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    # 总体结果
    all_passed = all([python_ok, deps_ok, camera_ok, dirs_ok, files_ok])
    if all_passed:
        print("\n✓ 所有检查都已通过！系统环境配置正确。")
    else:
        print("\n✗ 部分检查未通过。请解决上述问题后再运行系统。")
    
    return all_passed

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
