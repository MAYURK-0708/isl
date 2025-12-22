"""
Quick Setup Script for 6-Class Training
This creates the folder structure and helps you organize training data
"""

import os
import shutil

# Define the 6 gestures
GESTURES = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]
TRAINING_DIR = "training-data"

def create_training_folders():
    """Create folder structure for all 6 gestures"""
    print("Creating training data folders...")
    
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)
        print(f"✓ Created {TRAINING_DIR} directory")
    
    for gesture in GESTURES:
        gesture_dir = os.path.join(TRAINING_DIR, gesture)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
            print(f"✓ Created folder: {gesture_dir}")
        else:
            # Count existing videos
            video_count = len([f for f in os.listdir(gesture_dir) 
                             if f.endswith(('.mp4', '.avi', '.mov', '.webm'))])
            print(f"✓ Folder exists: {gesture_dir} ({video_count} videos)")
    
    print("\nFolder structure created successfully!")

def copy_from_include_dataset(include_path):
    """
    Copy videos from INCLUDE dataset to training folders
    
    Args:
        include_path: Path to the extracted INCLUDE "Greetings" folder
    """
    if not os.path.exists(include_path):
        print(f"❌ INCLUDE dataset not found at: {include_path}")
        print("Download it first:")
        print("  wget https://zenodo.org/record/4010759/files/Greetings_1of2.zip")
        print("  wget https://zenodo.org/record/4010759/files/Greetings_2of2.zip")
        print("  unzip Greetings_1of2.zip")
        print("  unzip Greetings_2of2.zip")
        return
    
    # Mapping of INCLUDE folders to our gesture names
    include_mapping = {
        "50. Alright": "alright",
        "51. Good Morning": "good morning",
        "52. Good afternoon": "good afternoon",
        "48. Thank You": "thank you",  # If you need more Thank You videos
        "46. Hello": "Hello"  # If you need more Hello videos
    }
    
    print(f"\nCopying videos from INCLUDE dataset: {include_path}")
    
    for include_folder, gesture in include_mapping.items():
        source_folder = os.path.join(include_path, include_folder)
        dest_folder = os.path.join(TRAINING_DIR, gesture)
        
        if not os.path.exists(source_folder):
            print(f"⚠ Skipping {include_folder} - folder not found")
            continue
        
        # Get all video files
        videos = [f for f in os.listdir(source_folder) 
                 if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
        
        print(f"\nCopying {len(videos)} videos from '{include_folder}' to '{gesture}'...")
        
        copied = 0
        for video in videos:
            source = os.path.join(source_folder, video)
            dest = os.path.join(dest_folder, video)
            
            # Skip if file already exists
            if os.path.exists(dest):
                continue
            
            shutil.copy2(source, dest)
            copied += 1
        
        print(f"✓ Copied {copied} new videos to {gesture}/")
    
    print("\n✓ INCLUDE dataset videos copied successfully!")

def show_status():
    """Display current status of training data"""
    print("\n" + "="*60)
    print("TRAINING DATA STATUS")
    print("="*60)
    
    total_videos = 0
    for gesture in GESTURES:
        gesture_dir = os.path.join(TRAINING_DIR, gesture)
        if os.path.exists(gesture_dir):
            videos = [f for f in os.listdir(gesture_dir) 
                     if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
            count = len(videos)
            total_videos += count
            
            # Status indicator
            if count >= 100:
                status = "✓ Excellent"
            elif count >= 50:
                status = "⚠ Good"
            elif count >= 20:
                status = "⚠ Minimum"
            else:
                status = "❌ Need more"
            
            print(f"{gesture:20s}: {count:3d} videos  {status}")
        else:
            print(f"{gesture:20s}:   0 videos  ❌ Folder missing")
    
    print("="*60)
    print(f"Total videos: {total_videos}")
    print(f"Average per class: {total_videos/6:.1f}")
    print("\nRecommendation: 100+ videos per gesture for best accuracy")
    print("="*60)

def main():
    print("="*60)
    print("  6-CLASS ISL TRAINING DATA SETUP")
    print("="*60)
    print("\nGestures: Hello, How are you, thank you,")
    print("          alright, good morning, good afternoon")
    print("="*60)
    
    # Create folders
    create_training_folders()
    
    # Ask about INCLUDE dataset
    print("\n" + "="*60)
    print("Do you want to copy videos from INCLUDE dataset?")
    print("="*60)
    print("This requires downloading the dataset first.")
    print("Download from: https://zenodo.org/record/4010759")
    choice = input("\nDo you have the INCLUDE dataset? (y/n): ").lower()
    
    if choice == 'y':
        include_path = input("Enter path to 'Greetings' folder (or press Enter for './Greetings'): ").strip()
        if not include_path:
            include_path = "Greetings"
        copy_from_include_dataset(include_path)
    
    # Show current status
    show_status()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Add more videos to each folder (use collect_training_data.py)")
    print("2. Open train-model.ipynb")
    print("3. Update actions array to 6 classes")
    print("4. Run all cells to train the model")
    print("5. Update app.py with new model weights path")
    print("\nSee RETRAIN_6_CLASSES.md for detailed instructions!")
    print("="*60)

if __name__ == "__main__":
    main()
