import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from collections import defaultdict
from tqdm import tqdm

# Load the YOLOv11 model
print("Loading model...")
model = YOLO('best.pt')
print("Model loaded successfully!")

# Define video paths
broadcast_path = 'broadcast.mp4'
tacticam_path = 'tacticam.mp4'

# Function to extract frames from video
def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is None:
        max_frames = total_frames
    
    frames = []
    frame_indices = []
    count = 0
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_indices.append(count)
        count += 1
        
    cap.release()
    return frames, fps

# Function to detect players in frames
def detect_players(frames, model):
    all_detections = []
    
    # Add progress bar
    for frame in tqdm(frames, desc="Detecting players", unit="frame"):
        results = model(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            
            for box, conf, cl in zip(boxes, confs, cls):
                x1, y1, x2, y2 = box
                class_id = int(cl)
                confidence = float(conf)
                
                # Class 0 is player, class 1 is ball (based on your model)
                if class_id == 0:  # Only consider players
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id
                    })
        
        all_detections.append(detections)
    
    return all_detections

# Function to extract features for each player
def extract_features(frames, detections):
    all_features = []
    
    for frame, frame_detections in tqdm(zip(frames, detections), desc="Extracting features", total=len(frames), unit="frame"):
        frame_features = []
        
        for det in frame_detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Extract player patch
            player_patch = frame[y1:y2, x1:x2]
            
            # Calculate color histogram as a simple feature
            if player_patch.size > 0:
                # Resize for consistency
                player_patch = cv2.resize(player_patch, (64, 128))
                
                # Calculate color histograms for each channel
                hist_b = cv2.calcHist([player_patch], [0], None, [32], [0, 256])
                hist_g = cv2.calcHist([player_patch], [1], None, [32], [0, 256])
                hist_r = cv2.calcHist([player_patch], [2], None, [32], [0, 256])
                
                # Normalize histograms
                hist_b = cv2.normalize(hist_b, hist_b).flatten()
                hist_g = cv2.normalize(hist_g, hist_g).flatten()
                hist_r = cv2.normalize(hist_r, hist_r).flatten()
                
                # Combine histograms
                hist_features = np.concatenate([hist_b, hist_g, hist_r])
                
                # Add spatial information (normalized position)
                h, w, _ = frame.shape
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                spatial_features = np.array([center_x, center_y, width, height])
                
                # Combine all features
                features = np.concatenate([hist_features, spatial_features])
                
                frame_features.append({
                    'features': features,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
            else:
                # If patch is empty, add placeholder
                frame_features.append({
                    'features': np.zeros(32*3 + 4),
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
        
        all_features.append(frame_features)
    
    return all_features

# Function to calculate similarity between features
def calculate_similarity(feature1, feature2):
    # Use cosine similarity for the histogram part
    hist1 = feature1[:96]  # First 96 elements are the histogram features
    hist2 = feature2[:96]
    
    hist_sim = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-6)
    
    # Use Euclidean distance for spatial features
    spatial1 = feature1[96:]  # Last 4 elements are spatial features
    spatial2 = feature2[96:]
    
    spatial_dist = np.linalg.norm(spatial1 - spatial2)
    spatial_sim = 1 / (1 + spatial_dist)  # Convert distance to similarity
    
    # Combine similarities with weights
    # Give more weight to histogram features as they're more discriminative
    combined_sim = 0.7 * hist_sim + 0.3 * spatial_sim
    
    return combined_sim

# Function to match players across cameras
def match_players(broadcast_features, tacticam_features, similarity_threshold=0.6):
    # We'll use the first frame to establish initial mappings
    broadcast_initial = broadcast_features[0]
    tacticam_initial = tacticam_features[0]
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(tacticam_initial), len(broadcast_initial)))
    
    for i, tac_feat in enumerate(tacticam_initial):
        for j, bc_feat in enumerate(broadcast_initial):
            similarity_matrix[i, j] = calculate_similarity(
                tac_feat['features'], bc_feat['features']
            )
    
    # Create mapping using greedy approach
    mapping = {}
    used_broadcast_indices = set()
    
    # Sort tacticam players by confidence
    tacticam_indices = sorted(
        range(len(tacticam_initial)), 
        key=lambda i: tacticam_initial[i]['confidence'], 
        reverse=True
    )
    
    for tac_idx in tacticam_indices:
        # Find best match in broadcast view
        similarities = similarity_matrix[tac_idx]
        sorted_indices = np.argsort(-similarities)  # Sort in descending order
        
        for bc_idx in sorted_indices:
            if bc_idx not in used_broadcast_indices and similarities[bc_idx] > similarity_threshold:
                mapping[tac_idx] = bc_idx
                used_broadcast_indices.add(bc_idx)
                break
    
    return mapping

# Function to visualize results
def visualize_results(broadcast_frames, tacticam_frames, broadcast_detections, 
                     tacticam_detections, player_mapping):
    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Assign colors to player IDs
    np.random.seed(42)  # For reproducibility
    colors = {}
    
    # Process frames with progress bar
    total_frames = min(len(broadcast_frames), len(tacticam_frames))
    for frame_idx in tqdm(range(total_frames), desc="Creating visualization", unit="frame"):
        # Get frames and detections
        broadcast_frame = broadcast_frames[frame_idx].copy()
        tacticam_frame = tacticam_frames[frame_idx].copy()
        
        broadcast_dets = broadcast_detections[frame_idx]
        tacticam_dets = tacticam_detections[frame_idx]
        
        # Draw broadcast view detections
        for i, det in enumerate(broadcast_dets):
            x1, y1, x2, y2 = det['bbox']
            player_id = i  # In broadcast view, player_id is just the detection index
            
            if player_id not in colors:
                colors[player_id] = tuple(map(int, np.random.randint(0, 255, 3)))
            
            color = colors[player_id]
            cv2.rectangle(broadcast_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(broadcast_frame, f"ID: {player_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tacticam view detections with mapped IDs
        for i, det in enumerate(tacticam_dets):
            x1, y1, x2, y2 = det['bbox']
            
            # Get mapped ID if available
            if i in player_mapping:
                player_id = player_mapping[i]
                if player_id not in colors:
                    colors[player_id] = tuple(map(int, np.random.randint(0, 255, 3)))
                color = colors[player_id]
            else:
                player_id = f"T{i}"  # Unmapped player
                if player_id not in colors:
                    colors[player_id] = tuple(map(int, np.random.randint(0, 255, 3)))
                color = colors[player_id]
            
            cv2.rectangle(tacticam_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(tacticam_frame, f"ID: {player_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Combine frames side by side
        h1, w1 = broadcast_frame.shape[:2]
        h2, w2 = tacticam_frame.shape[:2]
        
        # Resize to same height if needed
        if h1 != h2:
            if h1 > h2:
                scale = h1 / h2
                new_w = int(w2 * scale)
                tacticam_frame = cv2.resize(tacticam_frame, (new_w, h1))
            else:
                scale = h2 / h1
                new_w = int(w1 * scale)
                broadcast_frame = cv2.resize(broadcast_frame, (new_w, h2))
        
        combined_frame = np.hstack((broadcast_frame, tacticam_frame))
        
        # Save frame
        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, combined_frame)
    
    print(f"Saved {frame_idx+1} frames to {output_dir}")
    
    # Create video from frames
    create_video_from_frames(output_dir, 'output_video.mp4', fps=30)

# Function to create video from frames
def create_video_from_frames(frames_dir, output_path, fps=30):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print("No frames found to create video")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video with progress bar
    for frame_file in tqdm(frame_files, desc="Creating video", unit="frame"):
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        out.write(frame)
    
    out.release()
    print(f"Created video: {output_path}")

# Main function
def main():
    start_time = time.time()
    
    # Extract frames (limit to 100 frames for faster processing)
    print("Extracting frames from videos...")
    broadcast_frames, broadcast_fps = extract_frames(broadcast_path, max_frames=100)
    tacticam_frames, tacticam_fps = extract_frames(tacticam_path, max_frames=100)
    
    if broadcast_frames is None or tacticam_frames is None:
        print("Error extracting frames from videos")
        return
    
    print(f"Extracted {len(broadcast_frames)} broadcast frames and {len(tacticam_frames)} tacticam frames")
    
    # Detect players
    print("Detecting players in broadcast view...")
    broadcast_detections = detect_players(broadcast_frames, model)
    
    print("Detecting players in tacticam view...")
    tacticam_detections = detect_players(tacticam_frames, model)
    
    # Extract features
    print("Extracting features from broadcast view...")
    broadcast_features = extract_features(broadcast_frames, broadcast_detections)
    
    print("Extracting features from tacticam view...")
    tacticam_features = extract_features(tacticam_frames, tacticam_detections)
    
    # Match players
    print("Matching players across views...")
    player_mapping = match_players(broadcast_features, tacticam_features)
    
    print(f"Found {len(player_mapping)} player mappings")
    for tac_id, bc_id in player_mapping.items():
        print(f"Tacticam player {tac_id} maps to Broadcast player {bc_id}")
    
    # Save player mapping to JSON file for future evaluation
    import json
    with open('player_mapping_results.json', 'w') as f:
        # Convert keys to strings for JSON serialization
        json_mapping = {str(k): int(v) for k, v in player_mapping.items()}
        json.dump(json_mapping, f, indent=4)
    print(f"Saved player mapping to player_mapping_results.json")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(
        broadcast_frames, tacticam_frames,
        broadcast_detections, tacticam_detections,
        player_mapping
    )
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()