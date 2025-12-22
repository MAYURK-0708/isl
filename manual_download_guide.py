"""
Simple script to manually guide you through downloading the dataset
Since automatic download seems to have issues, use this manual method
"""

import os
import webbrowser

def main():
    print("="*70)
    print("  MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nThe automatic download had issues. Please download manually:")
    print("\nSTEP 1: Download these 2 files")
    print("-" * 70)
    print("1. https://zenodo.org/record/4010759/files/Greetings_1of2.zip")
    print("2. https://zenodo.org/record/4010759/files/Greetings_2of2.zip")
    print("\nEach file is ~1.5 GB. Save them to this folder:")
    print(f"  {os.getcwd()}")
    
    choice = input("\nOpen download links in browser? (y/n): ").lower()
    if choice == 'y':
        webbrowser.open("https://zenodo.org/record/4010759/files/Greetings_1of2.zip")
        webbrowser.open("https://zenodo.org/record/4010759/files/Greetings_2of2.zip")
        print("\n✓ Opened browser tabs for downloads")
    
    print("\n" + "="*70)
    print("STEP 2: After downloading")
    print("="*70)
    print("1. Make sure both .zip files are in this folder:")
    print(f"   {os.getcwd()}")
    print("2. Run this command to extract and organize:")
    print("   python extract_and_organize.py")
    print("="*70)
    
    # Check if files exist
    print("\nChecking for downloaded files...")
    file1 = "Greetings_1of2.zip"
    file2 = "Greetings_2of2.zip"
    
    if os.path.exists(file1):
        print(f"✓ Found: {file1}")
    else:
        print(f"✗ Missing: {file1}")
    
    if os.path.exists(file2):
        print(f"✓ Found: {file2}")
    else:
        print(f"✗ Missing: {file2}")
    
    if os.path.exists(file1) and os.path.exists(file2):
        print("\n✓ Both files found! You can run extract_and_organize.py now")
    else:
        print("\n⚠ Please download the missing files first")

if __name__ == "__main__":
    main()
