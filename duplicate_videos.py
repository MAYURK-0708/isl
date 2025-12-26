"""
Duplicate training videos to increase dataset size
Note: This duplicates existing videos. For better accuracy, collect NEW diverse videos.
"""

import os
import shutil

TRAINING_DIR = "training-data"
GESTURES = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]
COPIES_PER_VIDEO = 3  # Make 3 copies of each video (total 4x videos)

def duplicate_videos():
    """Duplicate each video to increase dataset size"""
    print("="*70)
    print("DUPLICATING TRAINING VIDEOS")
    print("="*70)
    print(f"Making {COPIES_PER_VIDEO} copies of each video...")
    print("Note: For better accuracy, collect NEW diverse videos instead!\n")
    
    total_original = 0
    total_copied = 0
    
    for gesture in GESTURES:
        gesture_path = os.path.join(TRAINING_DIR, gesture)
        
        if not os.path.exists(gesture_path):
            print(f"⚠ Folder not found: {gesture_path}")
            continue
        
        # Get original videos
        original_videos = [f for f in os.listdir(gesture_path) 
                          if f.lower().endswith(('.mov', '.mp4', '.avi', '.webm'))]
        
        if len(original_videos) == 0:
            print(f"{gesture}: No videos found")
            continue
        
        print(f"\n{gesture}: {len(original_videos)} original videos")
        total_original += len(original_videos)
        
        # Duplicate each video
        copied_count = 0
        for video in original_videos:
            base_name, ext = os.path.splitext(video)
            source_path = os.path.join(gesture_path, video)
            
            for copy_num in range(1, COPIES_PER_VIDEO + 1):
                # Check if copy already exists
                new_name = f"{base_name}_copy{copy_num}{ext}"
                dest_path = os.path.join(gesture_path, new_name)
                
                if os.path.exists(dest_path):
                    continue
                
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    total_copied += 1
                except Exception as e:
                    print(f"  Error copying {video}: {e}")
        
        # Count total videos now
        total_videos = len([f for f in os.listdir(gesture_path) 
                           if f.lower().endswith(('.mov', '.mp4', '.avi', '.webm'))])
        
        print(f"  Copied: {copied_count} new videos")
        print(f"  Total now: {total_videos} videos")
    
    print("\n" + "="*70)
    print(f"Original videos: {total_original}")
    print(f"New copies made: {total_copied}")
    print(f"Total videos: {total_original + total_copied}")
    print(f"Average per gesture: {(total_original + total_copied) / len(GESTURES):.0f}")
    print("="*70)

def show_status():
    """Show final dataset status"""
    print("\n" + "="*70)
    print("FINAL DATASET STATUS")
    print("="*70)
    
    total = 0
    for gesture in GESTURES:
        gesture_path = os.path.join(TRAINING_DIR, gesture)
        if os.path.exists(gesture_path):
            videos = [f for f in os.listdir(gesture_path) 
                     if f.lower().endswith(('.mov', '.mp4', '.avi', '.webm'))]
            count = len(videos)
            total += count
            print(f"{gesture:20s}: {count:3d} videos")
    
    print("="*70)
    print(f"Total: {total} videos")
    print(f"Average per class: {total/len(GESTURES):.1f}")
    print("="*70)

def main():
    print("="*70)
    print("  DUPLICATE TRAINING VIDEOS")
    print("="*70)
    print("\n⚠️  WARNING: Duplicating videos won't improve accuracy much!")
    print("For better results: Collect NEW diverse videos (different people, angles)")
    print("\nThis script will make copies of existing videos to increase dataset size.")
    print(f"Each video will be copied {COPIES_PER_VIDEO} times.")
    print("="*70)
    
    choice = input("\nProceed with duplication? (y/n): ").lower()
    
    if choice != 'y':
        print("Cancelled.")
        return
    
    duplicate_videos()
    show_status()
    
    print("\n" + "="*70)
    print("NEXT STEP: RETRAIN THE MODEL")
    print("="*70)
    print("Run: python train_6class_model.py")
    print("\nThis will take 1-2 hours.")
    print("="*70)

if __name__ == "__main__":
    main()
