# Cross-Camera Player Mapping

This project implements a solution for mapping players across different camera angles in sports videos. It maintains consistent player IDs across broadcast and tactical camera views.

## Features

- Player detection using YOLOv11
- Feature extraction for player matching
- Cross-camera player ID mapping
- Visualization of results with consistent color coding

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the following files in your project directory:
   - `best.pt` - The YOLOv11 model trained for player detection
   - `broadcast.mp4` - The broadcast camera view video
   - `tacticam.mp4` - The tactical camera view video

## Usage

### Running the Player Mapping

#### Windows Users
Simply double-click the `run_mapping.bat` file to run the player mapping solution.

#### Command Line
Run the player mapping script:

```bash
python run_mapping.py
```

The script will:
1. Extract frames from both videos
2. Detect players in each frame
3. Extract visual and spatial features for each player
4. Match players across camera views
5. Generate visualization frames with consistent player IDs
6. Create an output video showing the mapping results

### Testing the Model

To test the YOLOv11 model on a single frame from each video:

```bash
python test_model.py
```

This will generate test detection images showing the model's performance.

### Extracting Frames

To extract frames from the videos for manual inspection:

```bash
# Extract frames from both videos
python extract_frames.py all

# Extract frames from a specific video
python extract_frames.py broadcast.mp4

# Extract frames with custom settings
python extract_frames.py tacticam.mp4 --interval 10 --max 50
```

## Output

- `output_frames/` - Directory containing visualization frames
- `output_video.mp4` - Video showing side-by-side comparison with consistent player IDs
- `player_mapping_results.json` - JSON file containing the player mapping results

### Evaluating Mapping Performance

If you have ground truth data for player mapping, you can evaluate the performance using the evaluation script:

```bash
python evaluate_mapping.py --mapping player_mapping_results.json --ground-truth ground_truth.json
```

This will calculate the mapping accuracy and generate a confusion matrix visualization.

## How It Works

### Player Detection
The system uses a fine-tuned YOLOv11 model to detect players in each frame of both videos.

### Feature Extraction
For each detected player, the system extracts:
- Color histogram features (RGB channels)
- Spatial features (normalized position and size)

### Player Matching
Players are matched across cameras using a similarity metric that combines:
- Visual similarity (color histogram comparison)
- Spatial similarity (position and size)

### Visualization
The system visualizes the results by:
- Assigning consistent colors to each player ID
- Drawing bounding boxes and ID labels
- Creating side-by-side comparison frames
- Generating an output video

## Customization

You can adjust the following parameters in the script:
- `max_frames` - Number of frames to process
- `similarity_threshold` - Threshold for player matching
- Feature extraction parameters
- Visualization settings