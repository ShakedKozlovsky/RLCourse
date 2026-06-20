# PRD — Custom 2-D cleaning-robot simulator

> **The spec's central engineering constraint**: build this from scratch. No Gymnasium. No Gazebo. Just NumPy + shapely + math.

## 1. Goal

Provide a deterministic, reproducible, headless-friendly 2-D physics simulator of a differential-drive vacuum robot navigating real apartment floorplans loaded from HouseExpo. The simulator must:

1. Accept a `(v, ω) ∈ [−1, 1]²` action vector.
2. Update robot pose under differential-drive kinematics.
3. Detect collisions against wall polygons.
4. Track which grid cells the robot has cleaned.
5. Generate LIDAR observations (handled by `sensor/lidar.py`).
6. Render to PNG / GIF for visualisation (matplotlib `Agg` backend).

## 2. World representation

The HouseExpo JSON gives us a list of vertex sequences for the apartment outline + interior obstacles. We convert it into two parallel representations:

| Representation | Used by | Why |
|---|---|---|
| `shapely.geometry.Polygon` per wall / obstacle | LIDAR ray-casting, collision checks | Sub-grid precision |
| 2-D NumPy bitmap (`uint8` occupancy grid) | Coverage tracking, render fast path | O(1) cell lookup |

The pixel-to-metre conversion comes from `configs/setup.json::env.map_pixels_per_metre`.

```
World:
    polygons: list[shapely.Polygon]     # walls
    bounding_box: (x_min, y_min, x_max, y_max)   # metres
    grid: np.ndarray[H, W] of uint8     # 0=unvisited, 1=visited, 255=obstacle
    pixels_per_metre: float
```

## 3. Robot model — differential-drive kinematics (ADR-003)

Pose: `(x, y, θ)` ∈ R² × S¹.
Action: `(v_norm, ω_norm) ∈ [−1, 1]²` → scaled to `(v, ω) = (v_norm · v_max, ω_norm · ω_max)`.

Discrete update at `dt = 0.1 s`:

```
x'  = x + v · cos(θ) · dt
y'  = y + v · sin(θ) · dt
θ'  = θ + ω · dt          # wrapped to [−π, π]
```

This is the unicycle model. It is **pure** — no global state, no randomness. Lives in `simulator/kinematics.py::step_unicycle()`.

## 4. Collision detection (ADR-002)

For a candidate next pose `(x', y')`:

```
disk = shapely.Point(x', y').buffer(robot_radius)
collide = any(disk.intersects(p) for p in world.polygons)
```

If `collide` is True, the env **does not advance the pose** (the robot stays put) and applies the collision penalty.

## 5. Cleaning logic

For each step:

```
cell_i, cell_j = world.cell_index(robot.x, robot.y)
# A small disk of cells gets marked cleaned (cleaning radius from config)
for di, dj in self._cleaning_kernel:
    if 0 ≤ cell_i+di < H and 0 ≤ cell_j+dj < W:
        if world.grid[i, j] == 0:    # was unvisited
            world.grid[i, j] = 1     # mark visited
            new_cells += 1
```

The reward function uses `new_cells` as the headline driver.

## 6. Episode termination

| Condition | Done |
|---|---|
| Coverage ≥ `coverage_target` (default 0.10 — tuned in Layer 18) | True (success) |
| Step count ≥ `max_episode_steps` (default 500) | True (timeout) |
| Robot leaves the bounding box | True (out-of-bounds — collision penalty) |

## 7. Reset

```
reset():
    grid.fill(0)
    grid[obstacle cells] = 255
    robot.pose = random valid pose inside the apartment
    step_count = 0
    return obs
```

The random spawn samples uniformly in the bounding box and **rejects** poses inside walls (max 100 attempts).

## 8. Test plan

| Test | Pass criterion |
|---|---|
| Kinematics — zero action | new_pose == pose |
| Kinematics — max forward (v=1, ω=0, dt=0.1, v_max=0.5) | x advances by 0.05 m |
| Kinematics — max turn (v=0, ω=1, dt=0.1, ω_max=1.5) | θ advances by 0.15 rad |
| Kinematics — combined | matches closed-form for 1 step |
| Collision — point inside polygon | True |
| Collision — point outside | False |
| Collision — on edge | False (touch is not collide; `intersects` not `contains`) |
| Cleaning — first visit to cell | new_cells = 1 |
| Cleaning — second visit | new_cells = 0 |
| World grid round-trip | `cell_index(continuous_to_cell(i,j))` returns `(i,j)` for cell centres |

## 9. Non-goals

- 3-D maps
- Multi-floor apartments
- Friction / tyre slip
- Sensor noise on LIDAR (left as easy future extension)
