@echo off
chcp 65001 >nul 2>&1
title 天气图像分类系统 — 启动脚本
color 0A

echo ============================================================
echo    天气图像分类系统 — 一键启动脚本
echo    Weather Image Classification System
echo ============================================================
echo.

:: ============================================================
::  流程一：检测 Python 环境
:: ============================================================
echo [流程一] 检测 Python 环境...
echo.

set PYTHON_CMD=

where python3 >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python3
    goto :check_version
)

where python >nul 2>&1
if %errorlevel%==0 (
    python -c "import sys; exit(0 if sys.version_info[0]>=3 else 1)" >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON_CMD=python
        goto :check_version
    )
)

echo [!] 未检测到 Python3 环境。
echo     正在弹窗询问是否安装...
echo.

powershell -Command "& {
    Add-Type -AssemblyName System.Windows.Forms
    $result = [System.Windows.Forms.MessageBox]::Show(
        '未检测到 Python3 环境，是否自动安装？`n`n点击「是」将自动安装 Python 3.11`n点击「否」将退出程序',
        'Python3 未找到',
        'YesNo',
        'Question'
    )
    exit [int]($result -eq 'Yes')
}" >nul 2>&1

if %errorlevel%==1 (
    echo [√] 用户选择安装 Python3，正在下载...
    echo.
    set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    set PYTHON_INSTALLER=%TEMP%\python-installer.exe

    echo     下载中: %PYTHON_URL%
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'" 2>nul

    if not exist "%PYTHON_INSTALLER%" (
        echo [X] 下载失败，请手动安装 Python 3.11+
        pause
        exit /b 1
    )

    echo     安装中（静默模式，添加到 PATH）...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    timeout /t 5 /nobreak >nul
    set PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts

    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON_CMD=python
        echo [√] Python3 安装成功！
    ) else (
        echo [X] 安装后仍无法找到 Python，请重启终端后重试
        pause
        exit /b 1
    )
    del "%PYTHON_INSTALLER%" >nul 2>&1
) else (
    echo [X] 用户取消安装，程序退出。
    pause
    exit /b 1
)

:check_version
echo [√] 找到 Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.


:: ============================================================
::  流程二：检测并安装依赖
:: ============================================================
echo [流程二] 检测并安装 Python 依赖包...
echo.

if not exist "requirements.txt" (
    echo     [!] 未找到 requirements.txt，跳过依赖安装。
    goto :skip_deps
)

echo     正在检查依赖...
%PYTHON_CMD% -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo     [!] 缺少 torch，正在安装（CPU版）...
    %PYTHON_CMD% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>nul
)

set DEPS=torch torchvision numpy matplotlib seaborn scikit-learn pillow streamlit
for %%p in (%DEPS%) do (
    %PYTHON_CMD% -c "import %%p" >nul 2>&1
    if %errorlevel% neq 0 (
        echo     [!] 缺少 %%p，正在安装...
        %PYTHON_CMD% -m pip install %%p -q 2>nul
    )
)

echo [√] 依赖检查完成。
echo.

:skip_deps


:: ============================================================
::  流程三：选择训练配置
:: ============================================================
echo ============================================================
echo [流程三] 选择训练配置
echo ============================================================
echo.
echo   请选择模型训练的预设配置（输入数字后回车）：
echo.
echo   ┌─────────────────────────────────────────────────────────┐
echo   │  1. default   — 默认配置（快速验证，Adam + ReLU）      │
echo   │  2. best      — 最优配置（推荐，AdamW + LeakyReLU）★  │
echo   │  3. 自定义    — 手动输入所有参数                       │
echo   │  4. 跳过训练  — 直接进入下一步（使用已有模型）         │
echo   └─────────────────────────────────────────────────────────┘
echo.

set /p TRAIN_CHOICE=  请输入选择 [1/2/3/4]（默认2）:

if "%TRAIN_CHOICE%"=="" set TRAIN_CHOICE=2

if "%TRAIN_CHOICE%"=="1" (
    set TRAIN_CONFIG=default
    echo     → 选择了: default（默认配置）
) else if "%TRAIN_CHOICE%"=="2" (
    set TRAIN_CONFIG=best
    echo     → 选择了: best（最优配置）★
) else if "%TRAIN_CHOICE%"=="3" (
    goto :custom_train
) else if "%TRAIN_CHOICE%"=="4" (
    echo     → 跳过训练，直接使用已有模型
    goto :after_train
) else (
    echo     [!] 无效输入，使用默认配置 best
    set TRAIN_CONFIG=best
)
echo.
goto :run_train

:: ---------- 自定义训练参数 ----------
:custom_train
echo.
echo   ── 自定义训练参数 ──
echo.

:: 优化器
echo   优化器选择：
echo     1. adam     — 通用稳健（默认）
echo     2. adamw    — Adam升级版，更强正则化
echo     3. sgd      — 传统方法，慢但后期好
echo.
set /p OPT_CHOICE=  优化器 [1/2/3]（默认1）:
if "%OPT_CHOICE%"=="" set OPT_CHOICE=1
if "%OPT_CHOICE%"=="1" (set TRAIN_OPT=adam)
else if "%OPT_CHOICE%"=="2" (set TRAIN_OPT=adamw)
else if "%OPT_CHOICE%"=="3" (set TRAIN_OPT=sgd)
else (set TRAIN_OPT=adam)

:: 学习率
echo.
echo   学习率（控制每步调整幅度）：
echo     常用值: 0.01（大） / 0.001（中） / 0.0008（小，更稳）
echo.
set /p TRAIN_LR=  学习率（默认0.0008）:
if "%TRAIN_LR%"=="" set TRAIN_LR=0.0008

:: 激活函数
echo.
echo   激活函数选择：
echo     1. relu        — 经典简单
echo     2. leaky_relu  — 不会神经元死亡（推荐）
echo     3. gelu        — 更平滑，计算稍慢
echo.
set /p ACT_CHOICE=  激活函数 [1/2/3]（默认2）:
if "%ACT_CHOICE%"=="" set ACT_CHOICE=2
if "%ACT_CHOICE%"=="1" (set TRAIN_ACT=relu)
else if "%ACT_CHOICE%"=="2" (set TRAIN_ACT=leaky_relu)
else if "%ACT_CHOICE%"=="3" (set TRAIN_ACT=gelu)
else (set TRAIN_ACT=leaky_relu)

:: 调度器
echo.
echo   学习率调度器：
echo     1. plateau  — 验证集不进步就减半
echo     2. cosine   — 平滑衰减（推荐）
echo.
set /p SCHED_CHOICE=  调度器 [1/2]（默认2）:
if "%SCHED_CHOICE%"=="" set SCHED_CHOICE=2
if "%SCHED_CHOICE%"=="1" (set TRAIN_SCHED=plateau)
else if "%SCHED_CHOICE%"=="2" (set TRAIN_SCHED=cosine)
else (set TRAIN_SCHED=cosine)

:: 训练轮数
echo.
echo   训练轮数（epochs）：
echo     建议值: 20（快速） / 30（标准） / 40（充分） / 50（深度）
echo.
set /p TRAIN_EPOCHS=  训练轮数（默认30）:
if "%TRAIN_EPOCHS%"=="" set TRAIN_EPOCHS=30

:: 批大小
echo.
echo   批大小（batch_size）：
echo     建议值: 16（小批量） / 32（标准） / 64（大批量）
echo.
set /p TRAIN_BS=  批大小（默认32）:
if "%TRAIN_BS%"=="" set TRAIN_BS=32

:: 权重衰减
echo.
echo   权重衰减（weight_decay）：
echo     建议值: 1e-4（轻） / 1e-3（中） / 5e-3（强）
echo.
set /p TRAIN_WD=  权重衰减（默认1e-3）:
if "%TRAIN_WD%"=="" set TRAIN_WD=1e-3

echo.
echo   ── 自定义参数汇总 ──
echo   优化器: %TRAIN_OPT%
echo   学习率: %TRAIN_LR%
echo   激活函数: %TRAIN_ACT%
echo   调度器: %TRAIN_SCHED%
echo   训练轮数: %TRAIN_EPOCHS%
echo   批大小: %TRAIN_BS%
echo   权重衰减: %TRAIN_WD%
echo.

set /p CONFIRM=  确认开始训练？[Y/n]（默认Y）:
if /i "%CONFIRM%"=="n" (
    echo     已取消，返回主菜单...
    goto :menu_train
)

echo.
echo     开始训练（自定义配置）...
%PYTHON_CMD% train.py --config best --optimizer %TRAIN_OPT% --lr %TRAIN_LR% --activation %TRAIN_ACT% --scheduler %TRAIN_SCHED% --epochs %TRAIN_EPOCHS% --batch-size %TRAIN_BS%
goto :train_done

:: ---------- 执行预设训练 ----------
:run_train
echo.
echo   开始训练（配置: %TRAIN_CONFIG%）...
echo.
%PYTHON_CMD% train.py --config %TRAIN_CONFIG%
goto :train_done

:train_done
echo.
if %errorlevel%==0 (
    echo [√] 模型训练完成！
) else (
    echo [!] 训练过程出现错误，请检查上方输出。
)
echo.

:after_train


:: ============================================================
::  流程四：选择实验模式
:: ============================================================
echo ============================================================
echo [流程四] 选择实验模式
echo ============================================================
echo.
echo   请选择是否运行对比实验（输入数字后回车）：
echo.
echo   ┌─────────────────────────────────────────────────────────┐
echo   │  1. 快速对比（优化器，15轮）                           │
echo   │  2. 快速对比（激活函数，15轮）                         │
echo   │  3. 完整对比（优化器 + 激活函数，30轮）                │
echo   │  4. 自定义实验参数                                     │
echo   │  5. 跳过实验  — 直接启动 Web 应用                      │
echo   └─────────────────────────────────────────────────────────┘
echo.

set /p EXP_CHOICE=  请输入选择 [1/2/3/4/5]（默认5）:

if "%EXP_CHOICE%"=="" set EXP_CHOICE=5

if "%EXP_CHOICE%"=="1" (
    echo     → 运行快速优化器对比...
    echo.
    %PYTHON_CMD% experiment.py --fast --only-optimizer
) else if "%EXP_CHOICE%"=="2" (
    echo     → 运行快速激活函数对比...
    echo.
    %PYTHON_CMD% experiment.py --fast --only-activation
) else if "%EXP_CHOICE%"=="3" (
    echo     → 运行完整对比实验（可能需要较长时间）...
    echo.
    %PYTHON_CMD% experiment.py
) else if "%EXP_CHOICE%"=="4" (
    goto :custom_exp
) else if "%EXP_CHOICE%"=="5" (
    echo     → 跳过实验，直接启动 Web 应用
) else (
    echo     [!] 无效输入，跳过实验
)

goto :after_exp

:: ---------- 自定义实验参数 ----------
:custom_exp
echo.
echo   ── 自定义实验参数 ──
echo.

echo   实验模式：
echo     1. 仅对比优化器（adam / adamw / sgd）
echo     2. 仅对比激活函数（relu / leaky_relu / gelu）
echo     3. 全部对比（优化器 + 激活函数）
echo.
set /p EXP_MODE=  实验模式 [1/2/3]（默认3）:
if "%EXP_MODE%"=="" set EXP_MODE=3

echo.
echo   实验训练轮数：
echo     快速: 15轮 / 标准: 30轮 / 详细: 50轮
echo.
set /p EXP_EPOCHS=  实验轮数（默认15）:
if "%EXP_EPOCHS%"=="" set EXP_EPOCHS=15

:: 计算是否需要 --fast
set EXP_ARGS=
if %EXP_EPOCHS% leq 20 set EXP_ARGS=--fast

echo.
echo   ── 实验参数汇总 ──
if "%EXP_MODE%"=="1" (echo   模式: 仅对比优化器)
if "%EXP_MODE%"=="2" (echo   模式: 仅对比激活函数)
if "%EXP_MODE%"=="3" (echo   模式: 全部对比）
echo   训练轮数: %EXP_EPOCHS%
echo.

set /p EXP_CONFIRM=  确认开始实验？[Y/n]（默认Y）:
if /i "%EXP_CONFIRM%"=="n" (
    echo     已取消，跳过实验。
    goto :after_exp
)

echo.
echo     开始实验...
if "%EXP_MODE%"=="1" (
    %PYTHON_CMD% experiment.py %EXP_ARGS% --only-optimizer
) else if "%EXP_MODE%"=="2" (
    %PYTHON_CMD% experiment.py %EXP_ARGS% --only-activation
) else (
    %PYTHON_CMD% experiment.py %EXP_ARGS%
)

echo.
echo [√] 实验完成！
echo.

:after_exp


:: ============================================================
::  流程五：启动 Web 应用
:: ============================================================
echo ============================================================
echo [流程五] 启动 Web 应用
echo ============================================================
echo.

if not exist "model\weather_cnn.pth" (
    echo [!] 未找到已训练的模型文件 model\weather_cnn.pth
    echo     请先完成训练步骤。
    pause
    goto :end
)

%PYTHON_CMD% -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Streamlit 未安装，跳过 Web 应用启动。
    echo     请运行: %PYTHON_CMD% -m pip install streamlit
    goto :end
)

echo     正在启动 Streamlit 服务...
echo     浏览器将自动打开: http://localhost:8501
echo     如未打开请手动访问上述地址
echo     按 Ctrl+C 可停止服务
echo ============================================================
echo.

%PYTHON_CMD% -m streamlit run app.py --server.port 8501 --server.headless true
if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo [!] Streamlit 启动失败！
    echo.
    echo 请尝试运行 webui.bat 单独启动网页
    echo ============================================================
    echo.
    pause
)


:end
echo.
echo ============================================================
echo    感谢使用天气图像分类系统！
echo ============================================================
pause
