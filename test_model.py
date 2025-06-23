import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLOv11 model
print("Loading model...")
model = YOLO('best.pt')
print("Model loaded successfully!")

# Function to test model on a single frame
def test_model_on_frame(video_path, frame_number=0):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {frame_number} from {video_path}")
        cap.release()
        return
    
    # Release video capture
    cap.release()
    
    # Run inference
    print(f"Running inference on frame {frame_number} from {video_path}...")
    results = model(frame)
    
    # Process results
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    
    # Draw bounding boxes
    output_frame = frame.copy()
    
    # Define class names and colors
    class_names = {0: 'player', 1: 'ball'}
    colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for players, Red for ball
    
    # Draw detections
    for box, conf, cl in zip(boxes, confs, cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cl)
        confidence = float(conf)
        
        class_name = class_names.get(class_id, f"class_{class_id}")
        color = colors.get(class_id, (255, 0, 0))
        
        # Draw bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(output_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert from BGR to RGB for matplotlib
    output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(output_frame_rgb)
    plt.title(f"Detections on {video_path} (Frame {frame_number})")
    plt.axis('off')
    
    # Save the output
    output_path = f"test_detection_{video_path.split('.')[0]}_frame_{frame_number}.jpg"
    cv2.imwrite(output_path, output_frame)
    print(f"Saved detection result to {output_path}")
    
    # Print detection summary
    player_count = sum(1 for c in cls if int(c) == 0)
    ball_count = sum(1 for c in cls if int(c) == 1)
    
    print(f"Detection summary:")
    print(f"- Players detected: {player_count}")
    print(f"- Balls detected: {ball_count}")
    print(f"- Total detections: {len(boxes)}")

# Test on both videos
if __name__ == "__main__":
    print("Testing model on broadcast view...")
    test_model_on_frame('broadcast.mp4', frame_number=50)
    
    print("\nTesting model on tacticam view...")
    test_model_on_frame('tacticam.mp4', frame_number=50)
    
    print("\nTest completed!")