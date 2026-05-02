@echo off
setlocal

cd /d %~dp0\..

if not exist build\main.exe (
    echo Executable not found. Building first...
    call scripts\build.bat
    if %errorlevel% neq 0 exit /b 1
)

echo Running CUDA program...
build\main.exe %*

endlocal