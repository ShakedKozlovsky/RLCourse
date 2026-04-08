# Getting Started - Grid Drone RL Simulator

Quick guide to verify installation and start training!

---

## ✅ Installation Verification

### Step 1: Check UV Installation
```bash
uv --version
```
Expected: `uv 0.11.2` or higher

### Step 2: Verify Python Version
```bash
python3 --version
```
Expected: `Python 3.11.0` or higher

### Step 3: Check Dependencies
```bash
cd assignment1
uv sync
```
Expected: All dependencies installed successfully

---

## ✅ Functionality Verification

### Test 1: Run with Help Flag
```bash
uv run python3 src/main_grid.py --help
```

Expected output:
```
usage: main_grid.py [-h] [--config CONFIG] [--load LOAD]
                    [--grid-size GRID_SIZE GRID_SIZE]

Grid-based Drone RL Simulator
```

### Test 2: Import Test
```bash
uv run python3 -c "from src.environment.grid_env import GridDroneEnv; print('✓ Environment OK')"
uv run python3 -c "from src.rl.qlearning_agent import QLearningAgent; print('✓ Agent OK')"
uv run python3 -c "import pygame; print('✓ Pygame OK')"
```

Expected: All imports succeed

---

## 🚀 First Run

### Basic Launch
```bash
cd assignment1
uv run python3 src/main_grid.py
```

You should see:
- Window opens (1400×900)
- Grid with obstacles visible
- Dashboard on right side
- Menu bar at bottom
- Drone at start position (red circle with propellers)
- Goal at target position (green square)

### Start Training
Press `SPACE` to start training!

You'll see:
- Drone moving around the grid
- Episode counter incrementing
- Reward values updating
- Success rate increasing over time

---

## 🎮 Quick Controls Test

Try these immediately:

### Training Controls
- `SPACE` - Start/pause (try toggling)
- `F` - Fast forward mode (watch it zoom!)
- `H` - Toggle heatmap (see visited cells)

### Obstacle Tools
- `1` - Select building tool
- Click on grid - Place building
- `X` - Select eraser
- Click on building - Remove it

### Agent Management
- `S` - Save agent (check console for confirmation)
- `L` - Load agent (loads from saved_models/)
- `R` - Reset entire game

---

## 🔧 Troubleshooting

### Issue: "No module named pygame"
```bash
cd assignment1
uv sync
```

### Issue: "Display not found"
If running headless (no display):
```bash
# Start VNC server first
vncserver :1 -geometry 1920x1080

# Then run with display
DISPLAY=:1 uv run python3 src/main_grid.py
```

See `VNC_GUIDE.md` for detailed VNC setup.

### Issue: Window doesn't open
Check if pygame works:
```bash
uv run python3 -c "import pygame; pygame.init(); print('Pygame version:', pygame.version.ver)"
```

### Issue: Slow performance
- Normal mode: 30 FPS (expected)
- Fast forward: 1000 FPS (use `F` key)
- If still slow, check CPU usage

---

## 📚 Next Steps

### Learn the Interface
Read: `USAGE_GUIDE.md` - Detailed usage instructions

### Understand the Architecture
Read: `docs/CURRENT_ARCHITECTURE.md` - Technical deep dive

### Check Compliance
Read: `FINAL_COMPLIANCE_REPORT.md` - Code quality summary

### Explore Configuration
Edit: `configs/*.yaml` - Customize behavior

---

## ✅ Success Checklist

- [ ] UV installed and working
- [ ] Python 3.11+ verified
- [ ] Dependencies installed (uv sync)
- [ ] Simulator window opens
- [ ] Training starts with SPACE
- [ ] Tools work (place/remove obstacles)
- [ ] Save/load functions
- [ ] Console shows episode updates

**All checked?** You're ready to train! 🎉

---

## 🎯 Training Tips

### For Best Results
1. **Let it train**: 1000+ episodes for good convergence
2. **Watch the metrics**: Success rate should increase
3. **Try different layouts**: Add obstacles to test adaptation
4. **Save progress**: Use `S` to save good agents
5. **Use fast forward**: Speed up training with `F`

### What Good Training Looks Like
- Episode 0-100: Random exploration, low success
- Episode 100-500: Starting to learn, ~30% success
- Episode 500-1000: Clear improvement, 50-70% success
- Episode 1000+: Converged behavior, 80%+ success

### Signs of Issues
- Success rate stuck at 0%: Check if goal is reachable
- Q-table not growing: Check discretization bins
- Crashes on obstacle add: Check bounds validation

---

## 📞 Quick Reference

**Start simulator:**
```bash
uv run python3 src/main_grid.py
```

**With custom grid:**
```bash
uv run python3 src/main_grid.py --grid-size 25 25
```

**Load saved agent:**
```bash
uv run python3 src/main_grid.py --load saved_models/agent.pkl
```

**VNC mode:**
```bash
DISPLAY=:1 uv run python3 src/main_grid.py
```

---

**Ready to go!** Press SPACE and watch your agent learn! 🚀
