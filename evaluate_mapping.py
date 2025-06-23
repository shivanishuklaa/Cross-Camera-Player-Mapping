import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_mapping(mapping_file):
    """Load player mapping from a JSON file"""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_ground_truth(gt_file):
    """Load ground truth mapping from a JSON file"""
    with open(gt_file, 'r') as f:
        return json.load(f)

def calculate_accuracy(mapping, ground_truth):
    """Calculate mapping accuracy compared to ground truth"""
    correct = 0
    total = len(ground_truth)
    
    for tac_id, bc_id in ground_truth.items():
        if tac_id in mapping and mapping[tac_id] == bc_id:
            correct += 1
    
    return correct / total if total > 0 else 0

def calculate_confusion_matrix(mapping, ground_truth, num_players):
    """Calculate confusion matrix for player mapping"""
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_players, num_players), dtype=int)
    
    # Fill confusion matrix
    for tac_id, bc_id in mapping.items():
        if tac_id in ground_truth:
            true_bc_id = ground_truth[tac_id]
            confusion_matrix[int(true_bc_id)][int(bc_id)] += 1
    
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, output_file=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Player Mapping Confusion Matrix')
    plt.colorbar()
    
    num_players = confusion_matrix.shape[0]
    tick_marks = np.arange(num_players)
    plt.xticks(tick_marks, [f'BC{i}' for i in range(num_players)], rotation=45)
    plt.yticks(tick_marks, [f'GT{i}' for i in range(num_players)])
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2
    for i in range(num_players):
        for j in range(num_players):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Ground Truth Player ID')
    plt.xlabel('Predicted Player ID')
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved confusion matrix to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate player mapping performance")
    parser.add_argument("--mapping", "-m", required=True, help="Path to mapping JSON file")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--num-players", "-n", type=int, default=22, help="Number of players")
    parser.add_argument("--output", "-o", default="confusion_matrix.png", help="Output file for confusion matrix")
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.mapping):
        print(f"Error: Mapping file {args.mapping} not found")
        return
    
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file {args.ground_truth} not found")
        return
    
    # Load data
    mapping = load_mapping(args.mapping)
    ground_truth = load_ground_truth(args.ground_truth)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(mapping, ground_truth)
    print(f"Mapping accuracy: {accuracy:.2%}")
    
    # Calculate and plot confusion matrix
    confusion_matrix = calculate_confusion_matrix(mapping, ground_truth, args.num_players)
    plot_confusion_matrix(confusion_matrix, args.output)

if __name__ == "__main__":
    main()