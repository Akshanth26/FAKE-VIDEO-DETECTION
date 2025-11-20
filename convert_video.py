from moviepy.editor import VideoFileClip
import os

input_video = r"C:\Users\Akshanth Chouhan\Downloads\invideo-ai-720 We Mic'd Up_ 60 Seconds of Real Talk 2025-10-16.mp4"
output_video = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\final_video.mp4"

if os.path.exists(input_video):
    print("Converting video... This may take a minute.")
    clip = VideoFileClip(input_video)
    clip.write_videofile(output_video, codec='libx264', audio_codec='aac', fps=30)
    clip.close()
    print(f"✓ Conversion complete! Saved to: {output_video}")
else:
    print(f"✗ Error: Video not found at {input_video}")
    print("Please check the file path.")
