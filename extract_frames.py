import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, interval=30, max_frames=None):
    """Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {video_path}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"Extracting frames every {interval} frames")
    
    # Calculate frames to extract
    frames_to_extract = list(range(0, total_frames, interval))
    if max_frames is not None and len(frames_to_extract) > max_frames:
        frames_to_extract = frames_to_extract[:max_frames]
    
    # Extract frames
    extracted_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for frame_idx in tqdm(frames_to_extract, desc="Extracting frames"):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Save frame
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx:06d}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument("video", help="Path to video file or 'all' to process both videos")
    parser.add_argument("--output", "-o", default="extracted_frames", help="Output directory")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Extract every Nth frame")
    parser.add_argument("--max", "-m", type=int, default=None, help="Maximum number of frames to extract")
    args = parser.parse_args()
    
    if args.video.lower() == "all":
        # Process both videos
        videos = ["broadcast.mp4", "tacticam.mp4"]
        for video in videos:
            video_name = os.path.splitext(os.path.basename(video))[0]
            output_dir = os.path.join(args.output, video_name)
            extract_frames(video, output_dir, args.interval, args.max)
    else:
        # Process single video
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = os.path.join(args.output, video_name)
        extract_frames(args.video, output_dir, args.interval, args.max)

if __name__ == "__main__":
    main()