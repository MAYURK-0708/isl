"""
Extract and organize downloaded INCLUDE dataset
Run this AFTER manually downloading the zip files
"""

import os
import zipfile
import shutil

# Correct folder mappings based on INCLUDE dataset structure
FOLDER_MAPPING = {
    " Hello": "Hello",  # Note: INCLUDE has space before Hello
    " How Are You": "How are you",
    " Thank You": "thank you",
    " Alright": "alright",
    " Good Morning": "good morning",
    " Good afternoon": "good afternoon",
    # Also try without leading space
    "Hello": "Hello",
    "How Are You": "How are you",
    "Thank You": "thank you",
    "Alright": "alright",
    "Good Morning": "good morning",
    "Good afternoon": "good afternoon",
    # Also try with numbers
    "46. Hello": "Hello",
    "47. How Are You": "How are you",
    "48. Thank You": "thank you",
    "50. Alright": "alright",
    "51. Good Morning": "good morning",
    "52. Good afternoon": "good afternoon",
}

def extract_zip(zip_path):
    """Extract zip file"""
    print(f"\nExtracting: {os.path.basename(zip_path)}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total = len(file_list)
            
            print(f"  Extracting {total} files...")
            
            for i, file in enumerate(file_list):
                zip_ref.extract(file, ".")
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{total}", end='\r')
            
            print(f"  ✓ Extracted {total} files")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def find_video_folders():
    """Find all folders containing videos"""
    video_folders = {}
    
    print("\nScanning for video folders...")
    
    # Scan current directory and subdirectories
    for root, dirs, files in os.walk("."):
        # Skip certain directories
        if any(skip in root for skip in ['.venv', '.git', '__pycache__', 'training-data']):
            continue
        
        # Check if folder has videos
        videos = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if len(videos) > 0:
            folder_name = os.path.basename(root)
            video_folders[root] = (folder_name, len(videos))
            print(f"  Found: {root} ({len(videos)} videos)")
    
    return video_folders

def organize_videos(video_folders):
    """Organize videos into training-data structure"""
    training_dir = "training-data"
    
    # Create gesture folders
    gestures = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]
    for gesture in gestures:
        os.makedirs(os.path.join(training_dir, gesture), exist_ok=True)
    
    print("\n" + "="*70)
    print("ORGANIZING VIDEOS")
    print("="*70)
    
    total_copied = 0
    
    # Try to match folders to gestures
    for folder_path, (folder_name, video_count) in video_folders.items():
        print(f"\nProcessing: {folder_name} ({video_count} videos)")
        
        # Find matching gesture
        target_gesture = None
        for source_name, gesture_name in FOLDER_MAPPING.items():
            if source_name.lower() in folder_name.lower():
                target_gesture = gesture_name
                break
        
        if not target_gesture:
            # Try fuzzy matching
            folder_lower = folder_name.lower()
            if 'hello' in folder_lower:
                target_gesture = "Hello"
            elif 'how' in folder_lower and 'are' in folder_lower:
                target_gesture = "How are you"
            elif 'thank' in folder_lower:
                target_gesture = "thank you"
            elif 'alright' in folder_lower or 'all right' in folder_lower:
                target_gesture = "alright"
            elif 'good' in folder_lower and 'morning' in folder_lower:
                target_gesture = "good morning"
            elif 'good' in folder_lower and 'afternoon' in folder_lower:
                target_gesture = "good afternoon"
        
        if target_gesture:
            print(f"  → Copying to: {target_gesture}")
            dest_folder = os.path.join(training_dir, target_gesture)
            
            # Copy all videos
            videos = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            copied = 0
            for video in videos:
                source = os.path.join(folder_path, video)
                dest = os.path.join(dest_folder, video)
                
                if os.path.exists(dest):
                    continue
                
                try:
                    shutil.copy2(source, dest)
                    copied += 1
                    total_copied += 1
                except Exception as e:
                    print(f"    Error copying {video}: {e}")
            
            print(f"  ✓ Copied {copied} videos")
        else:
            print(f"  ⚠ Could not match to any gesture")
    
    return total_copied

def show_status():
    """Show final dataset status"""
    print("\n" + "="*70)
    print("FINAL DATASET STATUS")
    print("="*70)
    
    training_dir = "training-data"
    gestures = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]
    
    total = 0
    for gesture in gestures:
        folder = os.path.join(training_dir, gesture)
        if os.path.exists(folder):
            videos = [f for f in os.listdir(folder) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
            count = len(videos)
            total += count
            print(f"{gesture:20s}: {count:3d} videos")
    
    print("="*70)
    print(f"Total: {total} videos")
    print(f"Average per class: {total/6:.1f}")
    print("="*70)

def main():
    print("="*70)
    print("  EXTRACT AND ORGANIZE INCLUDE DATASET")
    print("="*70)
    
    # Check for zip files
    zip_files = ["Greetings_1of2.zip", "Greetings_2of2.zip"]
    found_zips = [f for f in zip_files if os.path.exists(f)]
    
    if len(found_zips) == 0:
        print("\n❌ No zip files found!")
        print("Please download them first using manual_download_guide.py")
        return
    
    print(f"\nFound {len(found_zips)} zip file(s)")
    for zf in found_zips:
        print(f"  ✓ {zf}")
    
    # Extract zip files
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING ZIP FILES")
    print("="*70)
    
    for zip_file in found_zips:
        extract_zip(zip_file)
    
    # Find video folders
    print("\n" + "="*70)
    print("STEP 2: FINDING VIDEO FOLDERS")
    print("="*70)
    
    video_folders = find_video_folders()
    
    if len(video_folders) == 0:
        print("\n❌ No video folders found after extraction!")
        return
    
    # Organize videos
    print("\n" + "="*70)
    print("STEP 3: ORGANIZING VIDEOS")
    print("="*70)
    
    total_copied = organize_videos(video_folders)
    
    print(f"\n✓ Total videos copied: {total_copied}")
    
    # Show final status
    show_status()
    
    # Cleanup option
    print("\n" + "="*70)
    choice = input("Delete extracted folders and zip files to save space? (y/n): ").lower()
    
    if choice == 'y':
        print("\nCleaning up...")
        for zip_file in found_zips:
            if os.path.exists(zip_file):
                os.remove(zip_file)
                print(f"  ✓ Deleted: {zip_file}")
        
        # Delete extracted folders (but keep training-data)
        for folder in os.listdir("."):
            if os.path.isdir(folder) and folder not in ['training-data', '.venv', '.git', '__pycache__', 'static', 'lstm-model', 'crnn-model-v1-initial-attempt-files', 'input-video']:
                try:
                    shutil.rmtree(folder)
                    print(f"  ✓ Deleted: {folder}")
                except:
                    pass
    
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE!")
    print("="*70)
    print("\nNext step: Train the model")
    print("  python train_6class_model.py")
    print("="*70)

if __name__ == "__main__":
    main()
