# VNC Remote Viewing Guide

View the drone simulator through VNC (Virtual Network Computing) for remote access or headless servers.

---

## Quick Start

### Step 1: Start VNC Server

On the server (where the simulator runs):

```bash
cd /home/corsight/src/assignment1
./start_vnc_grid_simulator.sh
```

You'll see:
```
✓ Xvfb started on display :99
✓ x11vnc started on port 5900
======================================================================
VNC Server Ready!
Connect with TigerVNC viewer: localhost:5900 or 192.168.8.37:5900
======================================================================
```

### Step 2: Connect with VNC Viewer

#### Option A: On the Same Machine (localhost)

```bash
vncviewer localhost:5900
```

#### Option B: From Another Computer

```bash
vncviewer 192.168.8.37:5900
```

Replace `192.168.8.37` with your server's IP address.

---

## Detailed Instructions

### 1. Install VNC Viewer (Client Side)

**On Linux:**
```bash
# TigerVNC (recommended)
sudo apt-get install tigervnc-viewer

# Or RealVNC
sudo apt-get install realvnc-vnc-viewer
```

**On Windows:**
- Download TigerVNC: https://tigervnc.org/
- Or RealVNC: https://www.realvnc.com/en/connect/download/viewer/

**On macOS:**
```bash
brew install tiger-vnc
```

### 2. Start the VNC Server

On your server:

```bash
cd /home/corsight/src/assignment1
./start_vnc_grid_simulator.sh
```

**What this does:**
- Starts `Xvfb` (virtual X display on :99)
- Starts `x11vnc` (VNC server on port 5900)
- Launches the grid simulator
- Display is 1400x900 resolution

### 3. Connect with VNC

**Method 1: TigerVNC GUI**
1. Open TigerVNC Viewer
2. Enter server address: `192.168.8.37:5900`
3. Click "Connect"
4. You'll see the simulator window!

**Method 2: Command Line**
```bash
vncviewer 192.168.8.37:5900
```

**Method 3: SSH Tunnel (Secure)**
```bash
# On your local machine, create SSH tunnel
ssh -L 5900:localhost:5900 corsight@192.168.8.37

# Then in another terminal, connect to localhost
vncviewer localhost:5900
```

---

## Connection Details

| Parameter | Value |
|-----------|-------|
| **Server IP** | 192.168.8.37 |
| **VNC Port** | 5900 |
| **Display** | :99 |
| **Resolution** | 1400x900 |
| **Password** | None (local network only) |

---

## Using the Simulator via VNC

Once connected, you'll see the simulator window. You can:

- **Use all keyboard controls** (SPACE, F, H, 1, 2, 3, X, R, S, L, ESC)
- **Click buttons** in the bottom menu
- **Place obstacles** by clicking grid cells
- **Everything works exactly the same** as direct display

### Performance Tips

For best VNC performance:

1. **LAN Connection**: Use wired ethernet if possible
2. **Compression**: TigerVNC has good compression by default
3. **Quality vs Speed**: Adjust in VNC viewer settings
4. **Fast-Forward Mode**: Press `F` - training still runs fast even over VNC

---

## Stopping the VNC Server

To stop the simulator and VNC:

1. Press `ESC` in the simulator window, or
2. Press `Ctrl+C` in the terminal where you started the script

The script automatically cleans up:
- Stops x11vnc
- Stops Xvfb
- Closes all processes

---

## Troubleshooting

### "Connection refused" or "Connection timed out"

**Check if VNC server is running:**
```bash
ps aux | grep x11vnc
```

**Check if port 5900 is open:**
```bash
sudo netstat -tlnp | grep 5900
```

**Start the server if not running:**
```bash
./start_vnc_grid_simulator.sh
```

### "Cannot open display :99"

**Check if Xvfb is running:**
```bash
ps aux | grep Xvfb
```

**Restart the VNC script:**
```bash
# Kill any existing processes
pkill -f Xvfb
pkill -f x11vnc

# Start fresh
./start_vnc_grid_simulator.sh
```

### Firewall blocking connection

**Open port 5900 (if on different machine):**
```bash
sudo ufw allow 5900/tcp
```

Or for temporary testing:
```bash
sudo iptables -I INPUT -p tcp --dport 5900 -j ACCEPT
```

### Display is too small/large

Edit the script and change resolution:
```bash
# In start_vnc_grid_simulator.sh, line 7:
Xvfb :99 -screen 0 1920x1080x24 -ac &  # Larger
# or
Xvfb :99 -screen 0 1024x768x24 -ac &   # Smaller
```

### VNC tools not installed

**Install on Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install xvfb x11vnc
```

**Verify installation:**
```bash
which Xvfb x11vnc
```

### Simulator is slow over VNC

This is normal for network viewing. Options:

1. **Use fast-forward mode**: Press `F` in simulator
2. **Lower quality**: In VNC viewer settings, reduce quality
3. **Use SSH tunnel**: More efficient than direct VNC
4. **Run training headless**: Train without viewing, then view results

---

## Alternative: SSH with X Forwarding

For simple viewing without VNC:

```bash
# From your local machine
ssh -X corsight@192.168.8.37

# Then run simulator
cd /home/corsight/src/assignment1
./run_grid_simulator.sh
```

**Note**: X forwarding is usually slower than VNC for graphical apps.

---

## VNC Viewer Alternatives

If TigerVNC doesn't work, try:

1. **RealVNC Viewer**: https://www.realvnc.com/
2. **TightVNC**: http://www.tightvnc.com/
3. **Remmina** (Linux): `sudo apt-get install remmina`
4. **Built-in VNC** (macOS): Safari → vnc://192.168.8.37:5900

---

## Security Notes

**Current setup is NOT secure** - designed for local network only:
- No password authentication
- No encryption
- Listening on all interfaces (0.0.0.0)

**For production/internet use, add security:**

1. **Password protection:**
```bash
# Create password file
x11vnc -storepasswd your_password ~/.vnc/passwd

# Use it in script (edit line 23):
x11vnc -display :99 -forever -rfbauth ~/.vnc/passwd -rfbport 5900
```

2. **SSH tunnel (recommended):**
```bash
# Only listen on localhost
x11vnc -display :99 -forever -nopw -localhost -rfbport 5900

# Connect via SSH tunnel
ssh -L 5900:localhost:5900 corsight@192.168.8.37
vncviewer localhost:5900
```

3. **Firewall rules:**
```bash
# Only allow from specific IP
sudo ufw allow from 192.168.8.100 to any port 5900
```

---

## Testing VNC Connection

Quick test before starting simulator:

```bash
# Terminal 1: Start just the VNC server
Xvfb :99 -screen 0 1400x900x24 &
x11vnc -display :99 -forever -nopw -rfbport 5900 &

# Terminal 2: Connect
vncviewer localhost:5900

# You should see an empty gray screen (success!)
# Press Ctrl+C to stop test
```

---

## Summary

**Start VNC + Simulator:**
```bash
cd /home/corsight/src/assignment1
./start_vnc_grid_simulator.sh
```

**Connect from another machine:**
```bash
vncviewer 192.168.8.37:5900
```

**Stop everything:**
Press `ESC` in simulator or `Ctrl+C` in terminal

---

**That's it! You're now viewing the simulator remotely via VNC!** 🖥️➡️📺
