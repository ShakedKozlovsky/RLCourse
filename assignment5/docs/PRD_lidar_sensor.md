# PRD — LIDAR sensor (virtual)

> Per-mechanism PRD for the simulated 2-D LIDAR. Companion to [PRD_simulator.md](PRD_simulator.md).

## 1. Goal

Produce realistic, deterministic distance readings to walls / obstacles from the robot's current pose. The 24 LIDAR returns are the dominant component of the agent's observation vector — slide 8's "advantage data from experience replay" depends on these being **reproducible across episodes**.

## 2. Specification

| Parameter | Default | Source |
|---|---|---|
| `n_lidar_beams` | 24 | spec § 1 — "סדרת קריאות חיישני מרחק" (typical SLAM LIDAR has 8–32 beams) |
| `lidar_max_range_m` | 5.0 | Roomba-realistic; comparable to commercial 2-D LIDARs |
| `fov_degrees` | 360 | Spec uses an omnidirectional sensor |
| Output range | `[0, 1]` (after normalisation by max range) | Stable for the network |
| Noise model | None (deterministic) | Future extension; ensures reproducibility tests pass |

## 3. Algorithm — ray casting

```
For each beam angle θ_k = robot.θ + 2π · k / n_beams, k = 0..n_beams-1:
    ray = LineString from (x, y) extending lidar_max_range_m along θ_k
    intersections = ray.intersection(union(world.polygons))
    if intersections is empty:
        d_k = lidar_max_range_m
    else:
        d_k = min distance from (x, y) to intersection points
    normalised_d_k = d_k / lidar_max_range_m  ∈ [0, 1]
```

Implementation uses `shapely.geometry.LineString.intersection(MultiPolygon)`. The shapely `prepared` geometry is cached on the `World` instance.

## 4. Observation shape

```
obs = [
    lidar_beam_0, lidar_beam_1, ..., lidar_beam_23,    # 24 entries
    x_normalised, y_normalised,                          # 2 entries (position within bbox)
    sin(θ), cos(θ),                                       # 2 entries (orientation, avoids 2π discontinuity)
    coverage_fraction,                                    # 1 entry (% of grid already cleaned)
]                                                          # total: 29 entries
```

The `sin(θ), cos(θ)` representation is standard for RL on continuous angles — it avoids the discontinuity at θ = ±π that would confuse the network.

## 5. Test plan

| Test | Pass criterion |
|---|---|
| Square room, 4 cardinal beams from centre | All 4 beams return half the room's side length |
| Square room, robot at corner | At least one beam returns the room diagonal length |
| 8-beam scan from centre of an empty square | All 8 returns equal within tolerance |
| Robot facing wall at 1 m distance | The forward beam returns 1.0 m (within 1 cm) |
| Empty world (no polygons) | All beams return `lidar_max_range_m` |

## 6. Performance budget

Each `step()` calls `LidarSensor.scan()` once. With 24 beams and shapely-prepared geometry, a single scan should take < 1 ms on an apartment with < 100 polygons. We confirm this in a benchmark integration test.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Slow `intersection()` on complex polygons | `shapely.prepared.prep()` cached on World |
| Beam grazing a polygon edge gives 0-length intersection | Use `min(distance)` filter; treat 0 as "no hit" |
| Per-step shapely allocations dominate the loop | Reuse the LineString factory; profile in Layer 3 |
