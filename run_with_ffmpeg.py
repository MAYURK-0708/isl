import os 
import skvideo.io 
 
ffmpeg_path = r"C:\Users\mayur\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin" 
skvideo.setFFmpegPath(ffmpeg_path) 
print(f"FFmpeg configured at: {ffmpeg_path}") 
 
exec(open("deploy-code.py").read())
