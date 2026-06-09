@echo off
chcp 65001 >nul 2>&1
title 天气图像分类系统 — Web 界面
color 0B

echo ============================================================
echo    天气图像分类系统 — Web 界面启动
echo ============================================================
echo.

:: 检测 Python
set PYTHON_CMD=
where python3 >nul 2>&1
if %errorlevel%==0 (set PYTHON_CMD=python3& goto :check_streamlit)
where python >nul 2>&1
if %errorlevel%==0 (set PYTHON_CMD=python& goto :check_streamlit)

echo [X] 未找到 Python，请先运行 launch.bat 安装环境
pause
exit /b 1

:check_streamlit
%PYTHON_CMD% -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Streamlit 未安装，正在自动安装...
    %PYTHON_CMD% -m pip install streamlit -q 2>nul
)

echo [√] 环境就绪，启动 Web 应用...
echo.
echo 浏览器将自动打开: http://localhost:8501
echo 如未打开，请手动在浏览器中访问上述地址
echo 按 Ctrl+C 可停止服务
echo ============================================================
echo.

%PYTHON_CMD% -m streamlit run app.py --server.port 8501 --server.headless true

if %errorlevel% neq 0 (
    echo.
    echo [X] 启动失败，请检查：
    echo   1. 端口 8501 是否被占用
    echo   2. 是否已安装 streamlit: %PYTHON_CMD% -m pip install streamlit
    echo   3. 是否已训练模型: %PYTHON_CMD% train.py
    pause
)
