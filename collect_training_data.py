"""
Quick Video Data Collection Script
This script helps you collect training videos using your webcam
"""

import cv2
import os
from datetime import datetime

# Configuration
GESTURES = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]
VIDEOS_PER_GESTURE = 10  # Adjust this number
OUTPUT_DIR = "training-data"
VIDEO_DURATION = 3  # seconds
FPS = 30

def create_folders():
    """Create folder structure for training data"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for gesture in GESTURES:
        gesture_dir = os.path.join(OUTPUT_DIR, gesture)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
            print(f"Created folder: {gesture_dir}")

def collect_videos():
    """Collect training videos"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n=== ISL Training Data Collection ===")
    print("Instructions:")
    print("1. Position yourself in frame")
    print("2. Press SPACE to start recording")
    print("3. Perform the gesture")
    print("4. Recording will stop automatically")
    print("5. Press 'q' to quit\n")
    
    for gesture in GESTURES:
        print(f"\n--- Collecting: {gesture} ---")
        print(f"Target: {VIDEOS_PER_GESTURE} videos")
        
        gesture_dir = os.path.join(OUTPUT_DIR, gesture)
        existing_videos = len([f for f in os.listdir(gesture_dir) if f.endswith('.mp4')])
        
        for i in range(existing_videos, existing_videos + VIDEOS_PER_GESTURE):
            print(f"\nVideo {i+1}/{existing_videos + VIDEOS_PER_GESTURE}")
            print("Press SPACE when ready to record...")
            
            # Wait for user to be ready
            recording = False
            while not recording:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display instruction
                cv2.putText(frame, f"Gesture: {gesture}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Video {i+1}/{existing_videos + VIDEOS_PER_GESTURE}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to start recording", (50, 650), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recording = True
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(gesture_dir, f"{gesture}_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, FPS, (1280, 720))
            
            frames = VIDEO_DURATION * FPS
            print(f"Recording... ({VIDEO_DURATION} seconds)")
            
            for frame_num in range(frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Show countdown
                remaining = VIDEO_DURATION - (frame_num / FPS)
                cv2.putText(frame, f"Recording: {remaining:.1f}s", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(frame, (1200, 50), 20, (0, 0, 255), -1)  # Recording indicator
                
                cv2.imshow('Data Collection', frame)
                out.write(frame)
                cv2.waitKey(1)
            
            out.release()
            print(f"Saved: {output_file}")
            
            # Small break between recordings
            print("Get ready for next recording...")
            cv2.waitKey(2000)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Data Collection Complete! ===")
    print(f"Videos saved to: {OUTPUT_DIR}")

def main():
    print("ISL Training Data Collection Tool")
    print("==================================\n")
    
    create_folders()
    
    input("Press ENTER to start camera...")
    
    collect_videos()
    
    print("\nNext steps:")
    print("1. Review collected videos")
    print("2. Add more videos if needed (recommended: 100+ per gesture)")
    print("3. Open train-model.ipynb to train the model")
    print("4. Follow TRAINING_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()
