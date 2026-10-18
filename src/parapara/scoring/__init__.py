"""Scoring engine (PLAN.md §5): upper body only; footwork is never scored.

Pure Python + NumPy, runs in the game process.

- :mod:`.schedule`    -- when each keyframe is due (routine + beat clock, or a recorded clip)
- :mod:`.matching`    -- front-weighted pose error and pose score
- :mod:`.judge`       -- streaming judgments: timing grades, holds, claps, flow
- :mod:`.scorekeeper` -- points, combo multiplier, posture and flow bonuses
"""
