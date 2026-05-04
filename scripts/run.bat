@echo off
setlocal

cd /d %~dp0\..

set MODE=%1
if "%MODE%"=="" set MODE=release

if "%MODE%"=="debug" (
    set EXE=build\main_debug.exe
) else (
    set EXE=build\main.exe
)

if not exist %EXE% (
    echo Executable not found. Building first...
    call scripts\build.bat %MODE%
    if %errorlevel% neq 0 exit /b 1
)

echo Running CUDA program [%MODE%]...
%EXE% %*

endlocal