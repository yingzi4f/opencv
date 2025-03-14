@echo off
echo 正在启动手势识别系统...

REM 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 检查Python版本
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo 检测到Python版本: %PYTHON_VERSION%

REM 创建必要的目录
if not exist data (
    echo 创建数据目录...
    mkdir data
    mkdir data\raw
    mkdir data\processed
)

if not exist models (
    echo 创建模型目录...
    mkdir models
)

REM 检查是否需要安装依赖
if not exist venv (
    echo 首次运行，正在创建虚拟环境并安装依赖...
    python -m venv venv
    call venv\Scripts\activate.bat
    
    echo 正在安装依赖，这可能需要几分钟时间...
    pip install -r requirements.txt
    
    if %errorlevel% neq 0 (
        echo 安装依赖时出错。请尝试手动运行以下命令：
        echo pip install -r requirements.txt
        pause
        exit /b 1
    )
    
    echo 依赖安装完成！
) else (
    call venv\Scripts\activate.bat
)

REM 检查摄像头
echo 正在检查摄像头...
python -c "import cv2; cap = cv2.VideoCapture(0); print('摄像头状态:', '正常' if cap.isOpened() else '无法访问'); cap.release()"

REM 启动应用程序
echo 正在启动主程序...
python src\main.py

REM 如果程序异常退出，保持窗口打开
if %errorlevel% neq 0 (
    echo 程序异常退出，错误代码: %errorlevel%
    pause
)

deactivate
