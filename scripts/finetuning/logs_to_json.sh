#!/bin/bash

# Script to parse the finetuning logs to get epoch-wise loss

input_file=$1
output_file=$2

# Extract JSON-like blocks, fix quotes, and filter valid entries
grep -oE "\{[^}]+\}" "$input_file" | \
	sed "s/'/\"/g" | \
	jq -c 'select(.loss and .epoch)' | \
	jq -s '.' > "$output_file"
