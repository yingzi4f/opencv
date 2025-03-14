import os
import sys
import tkinter as tk
from tkinter import messagebox

# 确保可以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.gesture_recognition import GestureRecognizer
    from src.system_commands import SystemCommands
    from src.gui import GestureRecognitionGUI
    from src.data_collection import DataCollector
    from src.train_model import GestureModelTrainer
except ImportError as e:
    messagebox.showerror("导入错误", f"无法导入必要的模块: {str(e)}\n请确保已安装所有依赖项")
    sys.exit(1)

def main():
    """
    主程序入口
    """
    try:
        # 创建根窗口
        root = tk.Tk()
        
        # 创建各个组件
        data_collector = DataCollector()
        model_trainer = GestureModelTrainer()
        recognizer = GestureRecognizer()
        system_commands = SystemCommands()
        
        # 创建GUI
        app = GestureRecognitionGUI(
            root, 
            recognizer, 
            system_commands, 
            data_collector, 
            model_trainer
        )
        
        # 运行主循环
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("错误", f"程序运行时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
