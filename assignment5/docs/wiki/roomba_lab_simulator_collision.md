---
path: /home/corsight/src/RL_Course/assignment5/src/roomba_lab/simulator/collision.py
module: roomba_lab.simulator.collision
tags: [type-module]
---

# roomba_lab.simulator.collision

## Summary
Collision detection — checks whether a robot disk fits inside the apartment

## Public functions

- `roomba_lab.simulator.collision.is_collision` — Return True if a disk of radius `robot_radius` at `pose` is NOT fully inside
- `roomba_lab.simulator.collision.point_in_polygon` — Strict containment check — used by the random-spawn rejection sampler.

## Imports

- [[__future__|__future__]]
- [[shapely_geometry|shapely.geometry]]
- [[roomba_lab_simulator_kinematics|roomba_lab.simulator.kinematics]]