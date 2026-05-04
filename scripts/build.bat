@echo off
setlocal

REM ==========================================
REM Load Visual Studio environment
REM ==========================================
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 (
    echo Failed to load Visual Studio environment
    exit /b 1
)

REM ==========================================
REM Go to project root
REM ==========================================
cd /d %~dp0\..

REM ==========================================
REM Create build directory
REM ==========================================
if not exist build mkdir build

REM ==========================================
REM Default build mode = release
REM ==========================================
set MODE=%1

if "%MODE%"=="" set MODE=release

echo Build mode: %MODE%

REM ==========================================
REM Compiler flags
REM ==========================================
set INCLUDE_FLAGS=-Iinclude

if "%MODE%"=="debug" (
    set CUDA_FLAGS=-g -G
    set OUTPUT=build\main_debug.exe
) else (
    set CUDA_FLAGS=-O2 -lineinfo
    set OUTPUT=build\main.exe
)

REM ==========================================
REM Compile
REM ==========================================
nvcc src\main.cu -o %OUTPUT% %CUDA_FLAGS% %INCLUDE_FLAGS%

if %errorlevel% neq 0 (
    echo Build failed
    exit /b 1
)

echo Build successful: %OUTPUT%

endlocal