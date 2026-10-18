"""The demo dancer's motion (PLAN.md §7).

- :mod:`.animator` -- routine + beat position → full dancer pose, with motion
  styles per kind of move, two-step footwork and the upcoming-move preview.

Pure Python + NumPy (no cv2/mediapipe/pygame): rendering lives in
``parapara.game.dancer_view``, so a nicer renderer (e.g. a 3-D avatar) can
replace the stick figure later without touching the motion.
"""
