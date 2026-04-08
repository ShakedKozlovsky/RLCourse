# Grid Drone Simulator - Usage Guide

## 🎨 New Visual Interface

The simulator now has a **beautiful, intuitive interface** with:

### Visual Layout
```
┌─────────────────────────────────────────────────────────┐
│  Grid Panel (65%)          │   Dashboard Panel (35%)   │
│                             │                           │
│  🚁 Drone with propellers   │   📊 Episode Stats       │
│  🏢 Red Buildings           │   📈 Reward Chart        │
│  ⚠️  Dark Red Traps         │   🎯 Goal Rate           │
│  💨 Blue Wind Zones         │   📋 Legend              │
│  🟩 Green Goal              │                           │
│  ⬜ Wind Arrows             │                           │
│                             │                           │
├─────────────────────────────────────────────────────────┤
│              ✏️ EDITOR MODE - Selected: building        │
├─────────────────────────────────────────────────────────┤
│  [SPACE]  [F]    [1]     [2]    [3]    [X]   [S]   [L] │
│  Play/    Fast   🏢      ⚠️     💨     🗑️    💾   📂  │
│  Pause    Mode   Build   Trap   Wind   Erase Save Load │
└─────────────────────────────────────────────────────────┘
```

## 🖱️ Simple Click Behavior

### Adding Obstacles (NEW - Much Simpler!)

1. **Click a tool button** (or press number key)
   - 🏢 Building (Press '1' or click button)
   - ⚠️ Trap (Press '2' or click button)
   - 💨 Wind Zone (Press '3' or click button)
   - 🗑️ Eraser (Press 'X' or click button)

2. **Click again to DESELECT** the tool
   - Button returns to normal color
   - You can't accidentally place things

3. **Click on grid** to place/remove
   - With tool selected: Place on empty cells
   - With eraser: Remove any obstacle
   - Visual feedback in console

### Visual Feedback
- **Selected tool**: Button turns **blue** with highlight
- **Hovering**: Button lights up when mouse over
- **Active editor**: Green banner shows "✏️ EDITOR MODE"
- **Console messages**: Emoji indicators for all actions

## 🎮 Bottom Menu Options

| Button | Key | Action | Description |
|--------|-----|--------|-------------|
| **Play/Pause** | SPACE | Training | Start or pause the training |
| **Fast Mode** | F | Speed | Train at max speed (1000+ FPS) |
| **🏢 Building** | 1 | Tool | Add red building obstacles |
| **⚠️ Trap** | 2 | Tool | Add dangerous traps |
| **💨 Wind** | 3 | Tool | Add wind zones |
| **🗑️ Eraser** | X | Tool | Remove any obstacle |
| **💾 Save** | S | File | Save trained agent |
| **📂 Load** | L | File | Load saved agent |

## ✨ Enhanced Drone Icon

The drone now has a **realistic appearance**:
- Yellow circular body
- White propeller arms (cross shape)
- 4 small propellers at the ends
- White outline for visibility

Much better than the old diamond shape!

## 🎯 Workflow Examples

### Example 1: Create Obstacle Course
```
1. Start simulator
2. Click "🏢 Building" button (turns blue)
3. Click on grid cells to place buildings
4. Click "🏢 Building" again to deselect
5. Click "⚠️ Trap" button
6. Click to place traps
7. Click "SPACE" to start training!
```

### Example 2: Fix Mistakes
```
1. You placed a building by mistake
2. Click "🗑️ Eraser" button (or press X)
3. Click the building to remove it
4. Click "🗑️ Eraser" again to deselect
5. Continue editing or training
```

### Example 3: Quick Training
```
1. Start simulator
2. Click "SPACE" or press SPACE
3. Watch drone learn!
4. Click "Fast Mode" for faster training
5. Click "💾 Save" when done
```

## 🎨 Visual Improvements

### Before (Old):
- ⬜ Basic diamond for drone
- Simple red squares for buildings
- No visual menu
- Keyboard-only controls
- No visual feedback

### After (New):
- 🚁 Realistic drone with propellers
- Color-coded obstacles with meaning
- Beautiful bottom menu with icons
- Click OR keyboard controls
- Visual selection feedback
- Hover effects
- Clear status indicators

## 💡 Tips

1. **Mouse Users**: Just click the buttons at the bottom!
2. **Keyboard Users**: All shortcuts still work (1, 2, 3, X, etc.)
3. **Selection**: Tool stays selected until you click it again
4. **Safety**: Can't accidentally place - must select tool first
5. **Visual**: Watch the button colors to see what's selected
6. **Status**: Green "EDITOR MODE" banner shows when active

## 🚀 Quick Start

```bash
cd /home/corsight/src/assigment1
./start_vnc_grid_simulator.sh  # With VNC
# or
./run_grid_simulator.sh        # Direct
```

Then:
1. Look at the pretty menu at the bottom 😊
2. Click "SPACE" button to start
3. Enjoy watching the drone learn!

---

**The interface is now much more intuitive and visually appealing!** 🎉
