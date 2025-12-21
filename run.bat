@echo off
REM Set FFmpeg paths
set FFMPEG_BINARY=C:\Users\mayur\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe
set FFPROBE_BINARY=C:\Users\mayur\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe
set PATH=%PATH%;C:\Users\mayur\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin

REM Activate virtual environment and run
call .venv\Scripts\activate.bat
python deploy-code.py
