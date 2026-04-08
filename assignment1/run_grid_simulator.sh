#!/bin/bash
# Run script for grid-based drone RL simulator

echo "========================================"
echo "Grid-Based Drone RL Simulator"
echo "========================================"
echo ""
echo "Starting simulator with Q-Learning..."
echo ""

# Run the grid simulator
cd "$(dirname "$0")"
uv run python src/main_grid.py "$@"
