#! /usr/bin/env python3

import json
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_json_file>")
        sys.exit(1)

    json_path = sys.argv[1]

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract epochs and losses
    steps = [entry['epoch'] * 2893 for entry in data]       # 2893 steps per epoch
    losses = [entry['loss'] for entry in data]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    # Save to file
    output_filename = "loss_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
