import pyautogui
import subprocess
import os
import time
import ctypes
from datetime import datetime
import win32api
import win32con

class SystemCommands:
    def __init__(self):
        """
        初始化系统命令执行器
        """
        # 设置截图保存目录
        self.screenshot_dir = os.path.join(os.path.expanduser('~'), 'Pictures', 'GestureScreenshots')
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        
        # 初始化静音状态
        self.is_muted = False
    
    def execute_command(self, gesture_id):
        """
        根据手势ID执行相应的系统命令
        
        Args:
            gesture_id: 手势ID (0-8对应手势1-9)
            
        Returns:
            command_name: 执行的命令名称
            success: 是否执行成功
        """
        command_functions = {
            0: self.take_screenshot,       # 手势1: 全屏截图
            1: self.volume_up,            # 手势2: 音量增加
            2: self.volume_down,          # 手势3: 音量减小
            3: self.toggle_mute,          # 手势4: 静音/取消静音
            4: self.open_browser,         # 手势5: 打开浏览器
            5: self.open_task_manager,    # 手势6: 打开任务管理器
            6: self.lock_screen,          # 手势7: 锁屏
            7: self.open_calculator,      # 手势8: 打开计算器
            8: self.open_system_menu      # 手势9: 打开系统菜单
        }
        
        command_names = {
            0: "全屏截图",
            1: "音量增加",
            2: "音量减小",
            3: "静音/取消静音",
            4: "打开浏览器",
            5: "打开任务管理器",
            6: "锁屏",
            7: "打开计算器",
            8: "打开系统菜单"
        }
        
        if gesture_id not in command_functions:
            return f"无效的手势ID: {gesture_id}", False
        
        try:
            # 执行相应的命令函数
            command_functions[gesture_id]()
            return command_names[gesture_id], True
        except Exception as e:
            print(f"执行命令时出错: {str(e)}")
            return command_names[gesture_id], False
    
    def take_screenshot(self):
        """
        全屏截图
        """
        # 生成截图文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        
        # 截取全屏
        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)
        print(f"截图已保存到: {screenshot_path}")
    
    def volume_up(self):
        """
        增加系统音量
        """
        for _ in range(2):  # 按两次增加音量键以确保效果
            pyautogui.press('volumeup')
        print("音量已增加")
    
    def volume_down(self):
        """
        减小系统音量
        """
        for _ in range(2):  # 按两次减小音量键以确保效果
            pyautogui.press('volumedown')
        print("音量已减小")
    
    def toggle_mute(self):
        """
        切换静音/取消静音
        """
        pyautogui.press('volumemute')
        self.is_muted = not self.is_muted
        status = "静音" if self.is_muted else "取消静音"
        print(f"已{status}")
    
    def open_browser(self):
        """
        打开默认浏览器
        """
        try:
            # 尝试打开默认浏览器
            os.startfile('http://www.bing.com')
            print("已打开浏览器")
        except Exception as e:
            print(f"打开浏览器时出错: {str(e)}")
            # 备用方法，尝试直接运行Edge
            try:
                subprocess.Popen(["C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"])
                print("已打开Edge 浏览器")
            except:
                # 如果还是失败，尝试运行IE
                os.system('start iexplore')
                print("已打开IE 浏览器")
    
    def open_task_manager(self):
        """
        打开任务管理器
        """
        try:
            subprocess.Popen(["taskmgr.exe"])
            print("已打开任务管理器")
        except Exception as e:
            print(f"打开任务管理器时出错: {str(e)}")
    
    def lock_screen(self):
        """
        锁屏
        """
        try:
            ctypes.windll.user32.LockWorkStation()
            print("已锁屏")
        except Exception as e:
            print(f"锁屏时出错: {str(e)}")
    
    def open_calculator(self):
        """
        打开计算器
        """
        try:
            subprocess.Popen(["calc.exe"])
            print("已打开计算器")
        except Exception as e:
            print(f"打开计算器时出错: {str(e)}")
    
    def open_system_menu(self):
        """
        打开系统菜单 (Windows菜单)
        """
        try:
            # 模拟按Windows键
            pyautogui.press('win')
            print("已打开系统菜单")
        except Exception as e:
            print(f"打开系统菜单时出错: {str(e)}")

if __name__ == "__main__":
    # 测试系统命令
    commands = SystemCommands()
    
    print("系统命令测试")
    print("1-9: 执行相应的手势命令")
    print("q: 退出")
    
    while True:
        key = input("\n请输入命令 (1-9, q 退出): ")
        
        if key.lower() == 'q':
            break
        
        try:
            gesture_id = int(key) - 1  # 转换为0-8的索引
            if 0 <= gesture_id <= 8:
                command_name, success = commands.execute_command(gesture_id)
                print(f"执行命令: {command_name} - {'成功' if success else '失败'}")
            else:
                print("无效的命令，请输入1-9")
        except ValueError:
            print("请输入数字 1-9 或 q 退出")
