#!/bin/bash
# Start VNC server with virtual display and run GRID simulator

echo "Starting Virtual X Display and VNC Server..."

# Start Xvfb on display :99 with good resolution for grid simulator
Xvfb :99 -screen 0 1400x900x24 -ac &
XVFB_PID=$!
export DISPLAY=:99

echo "Waiting for Xvfb to start..."
sleep 3

# Check if Xvfb is running
if ! ps -p $XVFB_PID > /dev/null; then
    echo "✗ Xvfb failed to start"
    exit 1
fi

echo "✓ Xvfb started on display :99 (PID: $XVFB_PID)"

# Start x11vnc to share display :99
x11vnc -display :99 -forever -nopw -listen 0.0.0.0 -rfbport 5900 -noxdamage -nowf -noscr -speeds lan &
X11VNC_PID=$!

echo "Waiting for x11vnc to start..."
sleep 3

# Check if x11vnc is running
if ! ps -p $X11VNC_PID > /dev/null; then
    echo "✗ x11vnc failed to start"
    kill $XVFB_PID
    exit 1
fi

echo "✓ x11vnc started on port 5900 (PID: $X11VNC_PID)"
echo ""
echo "======================================================================"
echo "VNC Server Ready!"
echo "Connect with TigerVNC viewer: localhost:5900 or 192.168.8.37:5900"
echo "======================================================================"
echo ""
echo "Starting Grid Drone RL Simulator..."

# Run the GRID simulator
cd /home/corsight/src/assignment1
/home/corsight/.local/bin/uv run python src/main_grid.py

# Cleanup on exit
echo "Simulator exited. Stopping VNC and Xvfb..."
kill $X11VNC_PID
kill $XVFB_PID
