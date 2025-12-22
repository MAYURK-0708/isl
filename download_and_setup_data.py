"""
Download and Setup INCLUDE Dataset for 6-Class Training
This script downloads the INCLUDE dataset and organizes it for training
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Dataset URLs
DATASET_URLS = [
    "https://zenodo.org/record/4010759/files/Greetings_1of2.zip",
    "https://zenodo.org/record/4010759/files/Greetings_2of2.zip"
]

# Mapping of INCLUDE folders to our gesture names
FOLDER_MAPPING = {
    "46. Hello": "Hello",
    "47. How Are You": "How are you",
    "48. Thank You": "thank you",
    "50. Alright": "alright",
    "51. Good Morning": "good morning",
    "52. Good afternoon": "good afternoon"
}

def download_file(url, dest_path):
    """Download file with progress"""
    print(f"\nDownloading: {os.path.basename(dest_path)}")
    print(f"From: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"  Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='\r')
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print(f"\n✓ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return False

def extract_zip(zip_path, extract_to="."):
    """Extract zip file"""
    print(f"\nExtracting: {os.path.basename(zip_path)}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted successfully!")
        return True
    except Exception as e:
        print(f"❌ Error extracting: {e}")
        return False

def organize_videos():
    """Organize videos into training-data folder structure"""
    print("\n" + "="*70)
    print("ORGANIZING VIDEOS FOR TRAINING")
    print("="*70)
    
    # Create training-data directory
    training_dir = "training-data"
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    
    # Create gesture folders
    for gesture in FOLDER_MAPPING.values():
        gesture_dir = os.path.join(training_dir, gesture)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
            print(f"✓ Created folder: {gesture_dir}")
    
    # Copy videos from Greetings folder to training-data
    greetings_dir = "Greetings"
    
    if not os.path.exists(greetings_dir):
        print(f"\n❌ Greetings folder not found!")
        return False
    
    total_copied = 0
    
    for include_folder, gesture_name in FOLDER_MAPPING.items():
        source_folder = os.path.join(greetings_dir, include_folder)
        dest_folder = os.path.join(training_dir, gesture_name)
        
        if not os.path.exists(source_folder):
            print(f"\n⚠ Warning: Folder not found: {source_folder}")
            continue
        
        # Get all video files
        video_files = [f for f in os.listdir(source_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"\nCopying {len(video_files)} videos: {include_folder} → {gesture_name}")
        
        copied = 0
        for video_file in video_files:
            source = os.path.join(source_folder, video_file)
            dest = os.path.join(dest_folder, video_file)
            
            # Skip if already exists
            if os.path.exists(dest):
                continue
            
            try:
                shutil.copy2(source, dest)
                copied += 1
                total_copied += 1
            except Exception as e:
                print(f"  Error copying {video_file}: {e}")
        
        print(f"  ✓ Copied {copied} videos")
    
    print("\n" + "="*70)
    print(f"✓ TOTAL VIDEOS COPIED: {total_copied}")
    print("="*70)
    
    return True

def show_final_status():
    """Show final dataset status"""
    print("\n" + "="*70)
    print("FINAL DATASET STATUS")
    print("="*70)
    
    training_dir = "training-data"
    total_videos = 0
    
    for gesture in FOLDER_MAPPING.values():
        gesture_dir = os.path.join(training_dir, gesture)
        if os.path.exists(gesture_dir):
            videos = [f for f in os.listdir(gesture_dir) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
            count = len(videos)
            total_videos += count
            print(f"{gesture:20s}: {count:3d} videos")
    
    print("="*70)
    print(f"Total videos: {total_videos}")
    print(f"Average per class: {total_videos/6:.1f}")
    print("="*70)

def cleanup_temp_files():
    """Clean up downloaded zip files (optional)"""
    print("\n" + "="*70)
    choice = input("Delete downloaded zip files to save space? (y/n): ").lower()
    
    if choice == 'y':
        for url in DATASET_URLS:
            zip_file = os.path.basename(url)
            if os.path.exists(zip_file):
                os.remove(zip_file)
                print(f"✓ Deleted: {zip_file}")
        
        if os.path.exists("Greetings"):
            shutil.rmtree("Greetings")
            print(f"✓ Deleted: Greetings folder")

def main():
    print("="*70)
    print("  INCLUDE DATASET DOWNLOAD AND SETUP")
    print("="*70)
    print("\nThis will:")
    print("1. Download INCLUDE ISL dataset (~2-3 GB)")
    print("2. Extract the zip files")
    print("3. Organize videos into training-data folder")
    print("\nRequired disk space: ~5 GB")
    print("Estimated time: 10-30 minutes (depends on internet speed)")
    print("="*70)
    
    choice = input("\nProceed with download? (y/n): ").lower()
    
    if choice != 'y':
        print("Setup cancelled.")
        return
    
    # Step 1: Download dataset
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATASET")
    print("="*70)
    
    for url in DATASET_URLS:
        zip_file = os.path.basename(url)
        
        if os.path.exists(zip_file):
            print(f"\n✓ Already downloaded: {zip_file}")
        else:
            success = download_file(url, zip_file)
            if not success:
                print("\n❌ Download failed! Check your internet connection.")
                return
    
    # Step 2: Extract files
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING FILES")
    print("="*70)
    
    for url in DATASET_URLS:
        zip_file = os.path.basename(url)
        if os.path.exists(zip_file):
            extract_zip(zip_file)
    
    # Step 3: Organize videos
    print("\n" + "="*70)
    print("STEP 3: ORGANIZING VIDEOS")
    print("="*70)
    
    success = organize_videos()
    
    if not success:
        print("\n❌ Failed to organize videos!")
        return
    
    # Show final status
    show_final_status()
    
    # Cleanup
    cleanup_temp_files()
    
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE!")
    print("="*70)
    print("\nNext step: Run training")
    print("  python train_6class_model.py")
    print("="*70)

if __name__ == "__main__":
    main()
