"""
Fix video organization - find the extracted Greetings folders and copy videos
"""

import os
import shutil
import glob

def find_greetings_folders():
    """Find all possible Greetings folders"""
    possible_paths = [
        "Greetings",
        "./Greetings",
        "../Greetings",
        "INCLUDE_ISL",
        "dataset",
    ]
    
    # Also search in current directory for any folder with numbers
    for item in os.listdir("."):
        if os.path.isdir(item) and any(char.isdigit() for char in item):
            print(f"Found folder with numbers: {item}")
            # Check if it contains video files
            if any(f.endswith(('.mp4', '.avi', '.mov')) for f in os.listdir(item) if os.path.isfile(os.path.join(item, f))):
                return item
    
    return None

def scan_and_organize():
    """Scan current directory for gesture folders and organize them"""
    print("Scanning for video folders...")
    
    # All possible folder name variations
    gesture_variations = {
        "Hello": ["46. Hello", "46.Hello", "Hello", "46 Hello", "hello"],
        "How are you": ["47. How Are You", "47.How Are You", "How Are You", "47 How Are You", "how are you"],
        "thank you": ["48. Thank You", "48.Thank You", "Thank You", "48 Thank You", "thank you"],
        "alright": ["50. Alright", "50.Alright", "Alright", "50 Alright"],
        "good morning": ["51. Good Morning", "51.Good Morning", "Good Morning", "51 Good Morning"],
        "good afternoon": ["52. Good afternoon", "52.Good afternoon", "Good afternoon", "52 Good afternoon"]
    }
    
    training_dir = "training-data"
    total_copied = 0
    
    # Get all folders in current directory
    all_folders = [f for f in os.listdir(".") if os.path.isdir(f)]
    
    print(f"\nFound {len(all_folders)} folders in current directory")
    print("\nSearching for gesture video folders...")
    
    for target_gesture, possible_names in gesture_variations.items():
        dest_folder = os.path.join(training_dir, target_gesture)
        
        print(f"\n{target_gesture}:")
        print(f"  Target folder: {dest_folder}")
        
        found = False
        for possible_name in possible_names:
            if possible_name in all_folders:
                source_folder = possible_name
                print(f"  ✓ Found source: {source_folder}")
                
                # Count and copy videos
                videos = [f for f in os.listdir(source_folder) 
                         if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                
                print(f"  Found {len(videos)} videos")
                
                copied = 0
                for video in videos:
                    source = os.path.join(source_folder, video)
                    dest = os.path.join(dest_folder, video)
                    
                    if os.path.exists(dest):
                        continue
                    
                    try:
                        shutil.copy2(source, dest)
                        copied += 1
                        total_copied += 1
                    except Exception as e:
                        print(f"    Error copying {video}: {e}")
                
                print(f"  Copied: {copied} videos")
                found = True
                break
        
        if not found:
            print(f"  ⚠ No matching folder found")
            print(f"  Looked for: {', '.join(possible_names[:3])}")
    
    print(f"\n{'='*70}")
    print(f"Total videos copied: {total_copied}")
    print(f"{'='*70}")

def show_all_folders_with_videos():
    """Show all folders that contain video files"""
    print("\n" + "="*70)
    print("ALL FOLDERS WITH VIDEOS")
    print("="*70)
    
    for item in os.listdir("."):
        if os.path.isdir(item) and item not in ['.venv', '.git', '__pycache__', 'static', 'lstm-model']:
            try:
                videos = [f for f in os.listdir(item) 
                         if f.lower().endswith(('.mp4', '.avi', '.mov')) 
                         and os.path.isfile(os.path.join(item, f))]
                
                if len(videos) > 0:
                    print(f"{item:40s}: {len(videos):3d} videos")
            except:
                pass

def main():
    print("="*70)
    print("  FIX VIDEO ORGANIZATION")
    print("="*70)
    
    show_all_folders_with_videos()
    
    print("\n" + "="*70)
    input("Press ENTER to start organizing videos...")
    
    scan_and_organize()
    
    # Show final status
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    
    training_dir = "training-data"
    gestures = ["Hello", "How are you", "thank you", "alright", "good morning", "good afternoon"]
    
    for gesture in gestures:
        folder = os.path.join(training_dir, gesture)
        if os.path.exists(folder):
            videos = [f for f in os.listdir(folder) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
            print(f"{gesture:20s}: {len(videos):3d} videos")

if __name__ == "__main__":
    main()
