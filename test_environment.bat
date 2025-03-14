@echo off
echo 正在检查手势识别系统环境...

REM 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 如果虚拟环境存在，则激活它
if exist venv (
    call venv\Scripts\activate.bat
    echo 已激活虚拟环境
) else (
    echo 警告: 虚拟环境不存在，将使用系统 Python 运行测试
)

REM 运行环境检查脚本
python test_environment.py

REM 如果激活了虚拟环境，则退出虚拟环境
if exist venv (
    deactivate
)
