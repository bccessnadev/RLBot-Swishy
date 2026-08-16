@echo off
setlocal

for %%I in ("%~dp0..\..\..\..") do set "ROCKETFORGE_ROOT=%%~fI"
set "ROCKETFORGE_BIN=%ROCKETFORGE_ROOT%\out\build\rlbot"
set "RLBOT_EXE=%~dp0RLBot.exe"
if not exist "%RLBOT_EXE%" set "RLBOT_EXE=%ROCKETFORGE_BIN%\RLBot.exe"
set "TORCH_LIB=%LOCALAPPDATA%\libtorch\lib"
set "PATH=%~dp0;%ROCKETFORGE_BIN%;%TORCH_LIB%;%PATH%"

REM Set your agent name here
set "AGENT=Swishy_v1.03_Submittal"

cd /d "%~dp0"
"%RLBOT_EXE%" "%AGENT%"
