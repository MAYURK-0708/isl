"""
Remove duplicate videos and retrain model with original data
"""

import os
import shutil

TRAINING_DIR = "training-data"
GESTURES = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]

def remove_duplicates():
    """Remove all _copy videos"""
    print("="*70)
    print("REMOVING DUPLICATE VIDEOS")
    print("="*70)
    
    total_removed = 0
    
    for gesture in GESTURES:
        gesture_path = os.path.join(TRAINING_DIR, gesture)
        
        if not os.path.exists(gesture_path):
            print(f"⚠ Folder not found: {gesture_path}")
            continue
        
        # Get all videos
        all_videos = [f for f in os.listdir(gesture_path) 
                     if f.lower().endswith(('.mov', '.mp4', '.avi', '.webm'))]
        
        # Find duplicates (contain "_copy" in filename)
        duplicates = [f for f in all_videos if '_copy' in f.lower()]
        
        print(f"\n{gesture}:")
        print(f"  Total videos: {len(all_videos)}")
        print(f"  Duplicates: {len(duplicates)}")
        
        # Remove duplicates
        removed = 0
        for video in duplicates:
            try:
                os.remove(os.path.join(gesture_path, video))
                removed += 1
                total_removed += 1
            except Exception as e:
                print(f"  Error removing {video}: {e}")
        
        # Count remaining
        remaining = len([f for f in os.listdir(gesture_path) 
                        if f.lower().endswith(('.mov', '.mp4', '.avi', '.webm'))])
        
        print(f"  Removed: {removed}")
        print(f"  Remaining: {remaining}")
    
    print("\n" + "="*70)
    print(f"Total duplicates removed: {total_removed}")
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
    print("  CLEANUP AND RETRAIN")
    print("="*70)
    print("\nThis will:")
    print("1. Remove all duplicate (_copy) videos")
    print("2. Keep only original videos")
    print("3. Retrain the model with clean data")
    print("="*70)
    
    choice = input("\nProceed? (y/n): ").lower()
    
    if choice != 'y':
        print("Cancelled.")
        return
    
    # Remove duplicates
    remove_duplicates()
    
    # Show final status
    show_status()
    
    print("\n" + "="*70)
    print("STARTING TRAINING...")
    print("="*70)
    print("This will take 1-2 hours with the original dataset (21 videos/class)")
    print("Training will start now...")
    print("="*70)
    
    input("\nPress ENTER to start training...")
    
    # Start training
    import subprocess
    import sys
    
    python_exe = sys.executable
    result = subprocess.run([python_exe, "train_6class_model.py"])
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Check training_history.png for accuracy graphs")
        print("2. Server will use new model automatically (lstm-model-6classes-best.hdf5)")
        print("3. Restart server: taskkill /F /IM python.exe & start 'ISL Server' cmd /k '.venv\\Scripts\\python.exe app.py'")
        print("4. Test at http://localhost:8000")
        print("="*70)
    else:
        print("\n❌ Training failed. Check error messages above.")

if __name__ == "__main__":
    main()
