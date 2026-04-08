# Drone RL Simulator - Grid-Based Q-Learning

A 2D grid-based drone navigation simulator using **Q-Learning with Bellman updates** to optimize flight paths through dynamic environments with obstacles and wind.

---

## 📚 Documentation

All documentation is in the `docs/` folder:

- **[README.md](README.md)** - This file - Quick start and overview
- **[docs/PRD.md](docs/PRD.md)** - Product Requirements Document
- **[docs/PLAN.md](docs/PLAN.md)** - Technical Implementation Plan
- **[docs/TODO.md](docs/TODO.md)** - Complete task list (1000+ tasks)
- **[docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md)** - Detailed Architecture Guide
- **[docs/FILE_MANIFEST.md](docs/FILE_MANIFEST.md)** - Complete File Structure
- **[docs/FINAL_COMPLIANCE_REPORT.md](docs/FINAL_COMPLIANCE_REPORT.md)** - Compliance Summary
- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Installation and verification guide
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Detailed usage instructions
- **[docs/VNC_GUIDE.md](docs/VNC_GUIDE.md)** - Headless setup for remote viewing
- **[docs/COMPLIANCE_STATUS.md](docs/COMPLIANCE_STATUS.md)** - Compliance progress tracking
- **[docs/SUBMISSION_COMPLIANCE.md](docs/SUBMISSION_COMPLIANCE.md)** - Initial compliance check

---

## Features

- **Q-Learning Algorithm**: Tabular Q-Learning with Bellman equation updates
- **Epsilon-Greedy Exploration**: Balanced exploration-exploitation (ε: 1.0 → 0.01)
- **2D Grid Environment**: Discrete 20x20 grid with configurable obstacles
- **Interactive UI**: Bottom menu with 10 clickable buttons
- **Dynamic Obstacles**: Place buildings, traps, and wind zones in real-time
- **Visit Heatmap**: Visualize which cells the drone explores most
- **Thematic Colors**: Beautiful color-coded environment elements
- **Save/Load System**: Manual agent persistence with notifications
- **VNC Support**: Run headless for remote viewing
- **Real-Time Stats**: Episode metrics, rewards, success rate, Q-table size

---

## Quick Start

```bash
cd assignment1
./run_grid_simulator.sh
```

The simulator will open immediately with the drone at the start position, goal in green, and interactive controls at the bottom.

---

## Prerequisites

Only **UV** (Python package manager) is required:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

**That's it!** UV automatically handles Python version, dependencies, and virtual environment.

---

## Controls

### Keyboard Shortcuts
- `SPACE` - Start/pause training
- `F` - Fast-forward mode (1000 FPS)
- `H` - Toggle visit heatmap overlay
- `1` - Select building placement tool
- `2` - Select trap placement tool
- `3` - Select wind zone placement tool
- `X` - Select eraser tool
- `R` - Complete reset (clears all progress, no save)
- `S` - Save agent manually
- `L` - Load saved agent
- `ESC` - Quit simulator

### Bottom Menu Buttons
Click buttons to toggle actions and tools. Selected tools highlight in blue.

---

## How It Works

### Q-Learning Algorithm

The simulator uses **tabular Q-Learning** with the Bellman update equation:

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

Where:
- `Q(s,a)` = Q-value for state s and action a
- `α` = Learning rate (0.1)
- `r` = Reward received
- `γ` = Discount factor (0.99)
- `max Q(s',a')` = Maximum Q-value in next state

### Epsilon-Greedy Exploration

- **Epsilon (ε)** starts at 1.0 (100% random exploration)
- Decays by 0.995 each episode
- Minimum epsilon: 0.01 (1% exploration, 99% exploitation)

### Environment

**Grid**: 20x20 discrete cells  
**Actions**: Stay, Up, Down, Left, Right (5 discrete actions)  
**Obstacles**:
- **Buildings** (brown): Collision = episode ends, -100 reward
- **Traps** (orange-red): -50 reward, continues
- **Wind Zones** (blue): Probabilistic push effect
- **Goal** (green): +100 reward, episode ends successfully

**Rewards**:
- Reach goal: +100
- Hit building: -100 (terminal)
- Hit trap: -50
- Move closer to goal: +1
- Move away from goal: -1
- Each step: -0.1 (time penalty)

---

## Training Progress

### What to Expect

**Episodes 1-100:**
- Epsilon high (0.90-0.60)
- Mostly random exploration
- Negative rewards
- Many collisions

**Episodes 100-500:**
- Epsilon medium (0.60-0.30)
- Learning patterns
- Rewards improving
- Some successful episodes

**Episodes 500-2000:**
- Epsilon low (0.30-0.05)
- Exploiting learned policy
- Positive average rewards
- 50-70% success rate

**Episodes 2000+:**
- Epsilon minimal (0.05-0.01)
- Optimal or near-optimal policy
- High success rate (70-85%)
- Shortest paths to goal

### Monitoring Training

Watch the **dashboard panel** (right side) for:
- **Episode**: Current episode number
- **Reward**: Total reward this episode
- **ε (Epsilon)**: Current exploration rate
- **Steps**: Actions taken this episode
- **Goal Rate**: Success percentage (last 100 episodes)
- **Reward Chart**: Line graph of last 100 episodes
- **Legend**: Color guide for obstacles

---

## Interactive Obstacle Editor

### Placing Obstacles

1. Click a tool button (BUILD, TRAP, WIND) or press `1`, `2`, `3`
2. Tool highlights in blue when selected
3. Click grid cells to place obstacles
4. Click tool button again to deselect

### Erasing Obstacles

1. Click ERASE button or press `X`
2. Click obstacles to remove them
3. Protected cells (start, goal) cannot be erased

### Best Practices

- Place obstacles **before** starting training
- Pause training (`SPACE`) to edit during learning
- Reset (`R`) after major changes for clean training
- Save (`S`) successful configurations

---

## Heatmap Visualization

Press `H` or click HEAT button to toggle the visit heatmap.

**Colors**:
- **Blue** = Rarely or never visited
- **Purple** = Moderate visits
- **Red** = Frequently visited

The heatmap reveals:
- Which paths the drone prefers
- Unexplored areas
- Inefficient exploration patterns
- Successful route clusters

**Tip**: Reset heatmap with `R` when starting new training runs.

---

## Saving and Loading

### Save Agent

Press `S` or click SAVE button to save:
- Q-table (all learned Q-values)
- Epsilon value
- Episode count
- Training steps

Saves to: `saved_models/qlearning_grid.pkl`

### Load Agent

Press `L` or click LOAD button to restore a saved agent and continue training or testing.

**Note**: No automatic checkpoints! Save manually when you want to preserve progress.

---

## Complete Reset

Press `R` or click RESET button for a full reset:
- ✓ New Q-Learning agent (empty Q-table)
- ✓ Epsilon reset to 1.0
- ✓ Episode counter to 0
- ✓ All statistics cleared
- ✓ Reward history cleared
- ✓ Goal rate reset to 0%
- ✓ Heatmap cleared
- ✓ Environment reset to start

**Does NOT save** before resetting. Use this for a completely fresh start.

---

## VNC Remote Viewing

For headless servers or remote viewing:

```bash
./start_vnc_grid_simulator.sh
```

Then connect with VNC viewer to `localhost:5900`

Requirements:
- `Xvfb` (virtual framebuffer)
- `x11vnc` (VNC server)

Install on Ubuntu/Debian:
```bash
sudo apt-get install xvfb x11vnc
```

---

## Configuration

Edit `configs/` YAML files to customize:

### `configs/environment.yaml`
```yaml
grid:
  width: 20
  height: 20
  max_steps: 200

rewards:
  goal_reached: 100
  building_collision: -100
  trap_collision: -50
```

### `configs/training.yaml`
```yaml
q_learning:
  learning_rate: 0.1
  discount_factor: 0.99
  initial_epsilon: 1.0
  final_epsilon: 0.01
  epsilon_decay: 0.995
```

### `configs/rendering.yaml`
```yaml
display:
  window_width: 1400
  window_height: 900
  fps: 10
  fast_forward_fps: 1000
```

---

## Project Structure

```
assignment1/
├── src/
│   ├── environment/
│   │   └── grid_env.py          # 2D grid environment
│   ├── rl/
│   │   └── qlearning_agent.py   # Q-Learning agent
│   ├── visualization/
│   │   └── grid_renderer.py     # 2D pygame renderer
│   ├── utils/
│   │   ├── config.py            # Configuration loader
│   │   ├── logger.py            # Metrics logging
│   │   └── math_utils.py        # Utilities
│   └── main_grid.py             # Entry point
├── configs/                      # YAML configuration
├── saved_models/                 # Saved agents
├── logs/                         # Training logs
├── PRD.md                        # Requirements (created 1st)
├── PLAN.md                       # Architecture (created 2nd)
├── TODO.md                       # 1850+ tasks (created 3rd)
├── pyproject.toml               # UV project definition
└── run_grid_simulator.sh        # Launch script
```

---

## Implemented Features

- ✓ 2D grid-based environment (20x20 configurable)
- ✓ Q-Learning with Bellman equation updates
- ✓ Epsilon-greedy exploration (1.0 → 0.01)
- ✓ Interactive bottom menu with 10 buttons
- ✓ Dynamic obstacles: buildings, traps, wind zones
- ✓ Click-to-select tool system
- ✓ Visit heatmap visualization
- ✓ Thematic color scheme
- ✓ Manual save/load system
- ✓ Complete reset functionality
- ✓ VNC remote viewing support
- ✓ Real-time training metrics and statistics
- ✓ On-screen notifications for actions
- ✓ Reward history chart
- ✓ Goal success rate tracking

---

## Troubleshooting

### "uv: command not found"
Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### "Permission denied" on shell script
Make executable: `chmod +x run_grid_simulator.sh`

### Window doesn't open
Check display: `echo $DISPLAY`  
Try VNC option: `./start_vnc_grid_simulator.sh`

### Training not progressing
- Ensure you pressed `SPACE` to start
- Check epsilon hasn't decayed to 0 (reset if needed)
- Try removing complex obstacles
- Reset (`R`) and restart training

### Agent performs poorly
- Train longer (2000+ episodes recommended)
- Reduce obstacle density
- Check epsilon decay isn't too fast
- Verify reward structure in configs

---

## File Manifest

- **Source Code**: 12 Python files (~2,000 lines)
- **Documentation**: 11 markdown files (~20,000 lines)
- **Planning**: PRD.md, PLAN.md, TODO.md (1850 tasks)
- **Configuration**: 3 YAML files

---

## Project Timeline

This project was implemented following a structured approach with strict phase ordering:

**Phase 1: Requirements (PRD.md)** - Created FIRST
   - Product Requirements Document
   - Complete specifications and success criteria
   - All features defined before any planning

**Phase 2: Architecture (PLAN.md)** - Created SECOND  
   - Technical implementation plan based on PRD
   - System architecture and design decisions
   - Algorithm specifications

**Phase 3: Task Breakdown (TODO.md)** - Created THIRD
   - Detailed TODO list with 1850+ tasks
   - Tasks derived from PLAN architecture
   - Complete before any coding

**Phase 4: Implementation** - Created LAST
   - Foundation & utilities
   - Grid environment with Q-Learning
   - 2D pygame visualization
   - Interactive UI and controls
   - Testing and refinement

---

## Documentation

- **PRD.md** - Complete product requirements (created 1st)
- **PLAN.md** - Technical implementation details (created 2nd)
- **TODO.md** - All 1850+ tasks with completion status (created 3rd)
- **INSTALLATION.md** - Setup guide for professors
- **QUICKSTART.md** - 5-minute quick start guide
- **USAGE_GUIDE.md** - Detailed UI and feature guide
- **PROJECT_SUMMARY.md** - High-level project overview
- **FILE_MANIFEST.md** - Complete file inventory
- **GETTING_STARTED.md** - Beginner's tutorial

---

## Implementation Highlights

### Q-Learning Agent
- 250 lines of code
- Dictionary-based Q-table
- State discretization for table lookup
- Pickle-based persistence
- Statistics tracking

### Grid Environment
- 350 lines of code
- Gymnasium-compatible interface
- 5 cell types (empty, building, trap, goal, wind)
- Dynamic obstacle API
- Heatmap tracking system

### Grid Renderer
- 700+ lines of code
- Pygame-CE based 2D rendering
- 10-button interactive menu
- Thematic color system
- Notification system
- Dashboard with charts

### Main Application
- 614 lines of code
- Event-driven architecture
- Tool selection system
- Save/load management
- Statistics aggregation

---

## Algorithm Details

### State Space
- Drone position (x, y)
- Goal position (x, y)
- Distance to goal (Manhattan)
- Surrounding cells (8 neighbors)
- Current wind at position

### Action Space
- 0: Stay in place
- 1: Move up
- 2: Move down
- 3: Move left
- 4: Move right

### Q-Table Structure
```python
q_table = {
    (state_tuple, action): q_value,
    ...
}
```

Keys are `(state, action)` tuples, values are Q-values (floats).

---

## Performance

- **Training Speed**: ~10,000 episodes in <30 minutes
- **Frame Rate**: 10 FPS (normal), 1000 FPS (fast-forward)
- **Memory Usage**: <100 MB (Q-table: 500-2000 states typically)
- **GPU**: Not required (CPU only)
- **Python Version**: 3.11+ required

---

## Support

For questions or issues:

1. Check `README.md` (this file) for comprehensive documentation
2. Review `PLAN.md` for technical architecture
3. See `TODO.md` for implementation details
4. Read `QUICKSTART.md` for a 5-minute overview
5. Check `INSTALLATION.md` for setup issues

---

## License

Educational project for reinforcement learning demonstration.

---

**Ready to fly?**

```bash
cd assignment1
./run_grid_simulator.sh
```

Watch the drone learn to navigate! 🚁
