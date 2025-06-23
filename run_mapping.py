import os
import sys
import time
from tqdm import tqdm

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def main():
    print_header("Cross-Camera Player Mapping")
    
    # Check if required files exist
    required_files = ['best.pt', 'broadcast.mp4', 'tacticam.mp4']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required files are in the current directory.")
        return
    
    # Check if player_mapping.py exists
    if not os.path.exists('player_mapping.py'):
        print("Error: player_mapping.py not found in the current directory.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs('output_frames', exist_ok=True)
    
    # Run the player mapping script
    print("\nStarting player mapping process...")
    print("This may take several minutes depending on your hardware.")
    
    try:
        start_time = time.time()
        
        # Import the player mapping module
        print("\nImporting player_mapping module...")
        sys.path.append(os.getcwd())
        import player_mapping
        
        # Run the main function with progress tracking
        print("\nRunning player mapping...")
        player_mapping.main()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print_header("Process Completed Successfully")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"\nOutput files:")
        print(f"  - Output frames: {os.path.abspath('output_frames')}")
        print(f"  - Output video: {os.path.abspath('output_video.mp4')}")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()