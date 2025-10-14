@echo off
setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0..
pushd "%PROJECT_ROOT%"

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

set "MODEL=%MODEL%"
if "%MODEL%"=="" set "MODEL=yolov8n.pt"

set "DATA=%DATA%"
if "%DATA%"=="" set "DATA=data/data.yaml"

set "EPOCHS=%EPOCHS%"
if "%EPOCHS%"=="" set "EPOCHS=50"

set "IMGSZ=%IMGSZ%"
if "%IMGSZ%"=="" set "IMGSZ=640"

set "BATCH=%BATCH%"
if "%BATCH%"=="" set "BATCH=16"

set "RUN_NAME=%RUN_NAME%"
if "%RUN_NAME%"=="" set "RUN_NAME=baseline_run"

set "SEED=%SEED%"
if "%SEED%"=="" set "SEED=42"

setlocal disableDelayedExpansion
:parse_args
if "%~1"=="" goto after_parse
if "%~1"=="--model" (
    set "MODEL=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="--data" (
    set "DATA=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="--epochs" (
    set "EPOCHS=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="--imgsz" (
    set "IMGSZ=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="--batch" (
    set "BATCH=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="--name" (
    set "RUN_NAME=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="--seed" (
    set "SEED=%~2"
    shift & shift
    goto parse_args
) else if "%~1"=="-h" (
    goto usage
) else if "%~1"=="--help" (
    goto usage
) else (
    echo Unexpected argument: %~1
    goto usage
)

:after_parse
setlocal enabledelayedexpansion
set "YOLO_EVAL_SEED=%SEED%"

yolo detect train ^
    model="%MODEL%" ^
    data="%DATA%" ^
    epochs="%EPOCHS%" ^
    imgsz="%IMGSZ%" ^
    batch="%BATCH%" ^
    seed="%SEED%" ^
    name="%RUN_NAME%"

popd
exit /b 0

:usage
echo Usage: train_windows.bat [--model WEIGHTS] [--data DATA] [--epochs E] [--imgsz S] [--batch B] [--name RUN_NAME] [--seed SEED]
popd
exit /b 1
