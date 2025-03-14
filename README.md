# 基于深度学习的手势识别系统

## 项目概述

这个项目实现了一个基于深度学习的手势识别系统，可以识别1-9的手势并执行相应的电脑操作：

- 1：全屏截图
- 2：音量增加
- 3：音量减少
- 4：静音/取消静音
- 5：打开浏览器
- 6：打开任务管理器
- 7：锁屏
- 8：打开计算器
- 9：打开系统菜单

## 系统架构

- 手势识别模型训练
- 实时摄像头检测
- 系统指令映射
- GUI界面

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型（如果需要）：
```bash
python src/train_model.py
```

2. 启动手势识别系统：
```bash
python src/main.py
```

## 项目结构

- `data/`: 存放训练数据和测试数据
- `models/`: 存放训练好的模型
- `src/`: 源代码
  - `data_collection.py`: 收集手势数据
  - `train_model.py`: 训练手势识别模型
  - `gesture_recognition.py`: 手势识别核心功能
  - `system_commands.py`: 系统命令执行功能
  - `gui.py`: 图形用户界面
  - `main.py`: 主程序入口
