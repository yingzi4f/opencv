import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
import os
import numpy as np
from PIL import Image, ImageTk

class GestureRecognitionGUI:
    def __init__(self, root, recognizer, system_commands, data_collector=None, model_trainer=None):
        """
        手势识别系统
        
        Args:
            root: tkinter根窗口
            recognizer: 手势识别器实例
            system_commands: 系统命令实例
            data_collector: 数据收集器实例
            model_trainer: 模型训练器实例
        """
        self.root = root
        self.recognizer = recognizer
        self.system_commands = system_commands
        self.data_collector = data_collector
        self.model_trainer = model_trainer
        
        # 设置窗口标题和大小
        self.root.title("手势识别系统")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建顶部标题
        title_label = ttk.Label(self.main_frame, text="基于深度学习的手势识别系统", font=("微软雅黑", 16, "bold"))
        title_label.pack(pady=10)
        
        # 创建内容框架
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧框架（视频显示）
        self.video_frame = ttk.LabelFrame(self.content_frame, text="摄像头预览")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建视频显示标签
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建右侧框架（控制和状态）
        self.control_frame = ttk.Frame(self.content_frame, width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        
        # 创建状态框架
        self.status_frame = ttk.LabelFrame(self.control_frame, text="状态信息")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 当前手势显示
        self.gesture_var = tk.StringVar(value="未检测到手势")
        gesture_label = ttk.Label(self.status_frame, text="当前手势:")
        gesture_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        gesture_value = ttk.Label(self.status_frame, textvariable=self.gesture_var, font=("微软雅黑", 10, "bold"))
        gesture_value.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # 最近执行的命令
        self.command_var = tk.StringVar(value="无")
        command_label = ttk.Label(self.status_frame, text="最近命令:")
        command_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        command_value = ttk.Label(self.status_frame, textvariable=self.command_var, font=("微软雅黑", 10, "bold"))
        command_value.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # 系统状态
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(self.status_frame, text="系统状态:")
        status_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        status_value = ttk.Label(self.status_frame, textvariable=self.status_var, font=("微软雅黑", 10, "bold"))
        status_value.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # 创建控制框架
        self.controls_frame = ttk.LabelFrame(self.control_frame, text="控制面板")
        self.controls_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 启动/停止按钮
        self.start_button = ttk.Button(self.controls_frame, text="启动识别", command=self.toggle_recognition)
        self.start_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加分隔线
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        # 数据收集按钮
        self.collect_button = ttk.Button(self.controls_frame, text="数据收集", command=self.start_data_collection)
        self.collect_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 模型训练按钮
        self.train_button = ttk.Button(self.controls_frame, text="训练模型", command=self.start_model_training)
        self.train_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 加载模型按钮
        self.load_model_button = ttk.Button(self.controls_frame, text="加载模型", command=self.load_model)
        self.load_model_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加分隔线
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        # 手势映射表
        self.mapping_frame = ttk.LabelFrame(self.control_frame, text="手势映射表")
        self.mapping_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 添加手势映射信息
        mappings = [
            ("手势1", "全屏截图"),
            ("手势2", "音量增加"),
            ("手势3", "音量减小"),
            ("手势4", "静音/取消静音"),
            ("手势5", "打开浏览器"),
            ("手势6", "打开任务管理器"),
            ("手势7", "锁屏"),
            ("手势8", "打开计算器"),
            ("手势9", "打开系统菜单")
        ]
        
        # 创建滚动区域
        mapping_canvas = tk.Canvas(self.mapping_frame)
        mapping_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.mapping_frame, orient=tk.VERTICAL, command=mapping_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 配置画布
        mapping_canvas.configure(yscrollcommand=scrollbar.set)
        mapping_canvas.bind('<Configure>', lambda e: mapping_canvas.configure(scrollregion=mapping_canvas.bbox(tk.ALL)))
        
        # 创建第二个框架用于滚动
        inner_frame = ttk.Frame(mapping_canvas)
        mapping_canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        # 添加映射条目
        for i, (gesture, command) in enumerate(mappings):
            gesture_label = ttk.Label(inner_frame, text=gesture, width=10)
            gesture_label.grid(row=i, column=0, sticky="w", padx=5, pady=3)
            
            arrow_label = ttk.Label(inner_frame, text="→")
            arrow_label.grid(row=i, column=1, padx=2, pady=3)
            
            command_label = ttk.Label(inner_frame, text=command)
            command_label.grid(row=i, column=2, sticky="w", padx=5, pady=3)
        
        # 添加底部状态栏
        self.footer_frame = ttk.Frame(self.main_frame)
        self.footer_frame.pack(fill=tk.X, pady=5)
        
        self.footer_label = ttk.Label(self.footer_frame, text="就绪中• 按ESC键退出")
        self.footer_label.pack(side=tk.LEFT, padx=10)
        
        # 初始化变量
        self.cap = None
        self.is_running = False
        self.video_thread = None
        self.last_command_time = 0
        self.command_cooldown = 2.0  # 命令冷却时间
        
        # 绑定键盘事件
        self.root.bind('<Escape>', lambda e: self.quit_application())
        
        # 在关闭时清理资源
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
    
    def toggle_recognition(self):
        """
        切换识别状态（启动/停止）
        """
        if self.is_running:
            self.stop_recognition()
        else:
            self.start_recognition()
    
    def start_recognition(self):
        """
        启动手势识别
        """
        if self.is_running:
            return
        
        # 检查模型是否加载成功
        if self.recognizer.model is None:
            messagebox.showerror("错误", "未加载模型，请先训练或加载模型")
            return
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        
        # 更新状态
        self.is_running = True
        self.status_var.set("运行中")
        self.start_button.config(text="停止识别")
        self.footer_label.config(text="识别中• 按ESC键退出")
        
        # 启动视频处理线程
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_recognition(self):
        """
        停止手势识别
        """
        self.is_running = False
        self.status_var.set("就绪")
        self.start_button.config(text="启动识别")
        self.footer_label.config(text="就绪中• 按ESC键退出")
        
        # 释放摄像头资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_video(self):
        """
        处理视频帧并识别手势
        """
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 水平翻转帧，使其像镜子一样
            frame = cv2.flip(frame, 1)
            
            # 处理帧并识别手势
            gesture_id, display_frame = self.recognizer.process_frame_with_tracking(frame)
            
            # 如果检测到稳定手势，执行相应命令
            if gesture_id is not None:
                current_time = time.time()
                # 检查命令冷却时间
                if (current_time - self.last_command_time) >= self.command_cooldown:
                    # 更新手势显示
                    gesture_text = self.recognizer.gesture_labels.get(gesture_id, f"未知手势: {gesture_id}")
                    self.gesture_var.set(gesture_text)
                    
                    # 执行系统命令
                    command_name, success = self.system_commands.execute_command(gesture_id)
                    self.command_var.set(command_name + (" (成功)" if success else " (失败)"))
                    
                    # 更新命令时间
                    self.last_command_time = current_time
            
            # 将帧转换为Tkinter可显示的格式
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # 更新视频显示
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
            # 短暂延迟
            time.sleep(0.03)
    
    def start_data_collection(self):
        """
        启动数据收集模块
        """
        if self.data_collector is None:
            messagebox.showinfo("提示", "数据收集模块未初始化")
            return
            
        # 停止当前的识别过程
        self.stop_recognition()
        
        # 询问收集模式
        result = messagebox.askyesno("数据收集", "选择数据收集模式:\n\n是: 自动收集模式\n否: 手动收集模式")
        
        # 创建新线程运行数据收集
        threading.Thread(target=self._run_data_collection, args=(result,), daemon=True).start()
    
    def _run_data_collection(self, auto_mode):
        """
        在新线程中运行数据收集
        
        Args:
            auto_mode: 是否使用自动收集模式
        """
        self.status_var.set("数据收集中")
        self.footer_label.config(text="数据收集中• 请按提示操作")
        
        try:
            if auto_mode:
                self.data_collector.collect_data_auto()
            else:
                self.data_collector.collect_data()
        except Exception as e:
            messagebox.showerror("错误", f"数据收集出错: {str(e)}")
        
        self.status_var.set("就绪")
        self.footer_label.config(text="就绪中• 按ESC键退出")
        messagebox.showinfo("完成", "数据收集完成")
    
    def start_model_training(self):
        """
        启动模型训练
        """
        if self.model_trainer is None:
            messagebox.showinfo("提示", "模型训练模块未初始化")
            return
            
        # 停止当前的识别过程
        self.stop_recognition()
        
        # 询问训练选项
        result = messagebox.askyesno("模型训练", "是否使用MediaPipe提取手部特征进行训练？\n\n(推荐选择'是')")
        
        # 创建新线程运行模型训练
        threading.Thread(target=self._run_model_training, args=(result,), daemon=True).start()
    
    def _run_model_training(self, use_mediapipe):
        """
        在新线程中运行模型训练
        
        Args:
            use_mediapipe: 是否使用MediaPipe提取手部特征
        """
        self.status_var.set("模型训练中")
        self.footer_label.config(text="模型训练中• 请等待")
        
        try:
            # 设置训练参数
            epochs = 30  # 可以根据需要调整
            
            # 运行训练
            self.model_trainer.train(epochs=epochs, use_mediapipe=use_mediapipe)
            
            # 训练完成后重新加载模型
            model_path = os.path.join('../models', 'gesture_model.h5')
            if os.path.exists(model_path):
                self.recognizer.model = self.model_trainer.build_model()
                self.recognizer.model.load_weights(model_path)
                messagebox.showinfo("成功", f"模型训练完成并已加载")
            else:
                messagebox.showwarning("警告", "模型训练完成，但无法找到模型文件")
        except Exception as e:
            messagebox.showerror("错误", f"模型训练出错: {str(e)}")
        
        self.status_var.set("就绪")
        self.footer_label.config(text="就绪中• 按ESC键退出")
    
    def load_model(self):
        """
        加载现有模型
        """
        # 停止当前的识别过程
        self.stop_recognition()
        
        # 打开文件选择对话框
        model_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")],
            initialdir="../models"
        )
        
        if model_path:
            try:
                # 重新初始化识别器并加载模型
                self.recognizer.model = None  # 清除当前模型
                from tensorflow.keras.models import load_model
                self.recognizer.model = load_model(model_path)
                messagebox.showinfo("成功", f"模型加载成功: {os.path.basename(model_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"模型加载出错: {str(e)}")
    
    def quit_application(self):
        """
        退出应用
        """
        # 停止视频处理
        self.is_running = False
        
        # 释放资源
        if self.cap is not None:
            self.cap.release()
        
        # 退出
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    # 这里只是为了测试GUI单独运行，实际使用时应该从主程序创建实例
    from gesture_recognition import GestureRecognizer
    from system_commands import SystemCommands
    
    root = tk.Tk()
    recognizer = GestureRecognizer()
    system_commands = SystemCommands()
    
    app = GestureRecognitionGUI(root, recognizer, system_commands)
    root.mainloop()
