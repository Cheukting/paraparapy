# Credits

The game shows a short version of this page (`C` in the menu of
`parapara-play`); this file has the full attributions with links.

## Music (demo songs)

The demo songs are not in the repository; `uv run parapara-fetch-demo-songs`
downloads them from the Free Music Archive (see `assets/songs/catalog.json`).
All four are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); no changes were
made to the recordings.

- "80s Motivational Chiptune" by Shane Ivers
  (https://freemusicarchive.org/music/shane-ivers/single/80s-motivational-chiptune/),
  licensed under CC BY 4.0
- "Goa Trolls" by Soulbringer
  (https://freemusicarchive.org/music/Soulbringer/Soulbringer/Soulbringer_-_Goa_Trolls/),
  licensed under CC BY 4.0
- "Lets Go To A Rave" by Defrini
  (https://freemusicarchive.org/music/defrini/songtrash-vol3-revenge-of-the-interplanetary-rubbish-bin/lets-go-to-a-rave/),
  licensed under CC BY 4.0
- "Onion March" by Shane Ivers
  (https://freemusicarchive.org/music/shane-ivers/single/onion-march/),
  licensed under CC BY 4.0

Songs players import themselves are their own music for personal use; the
game shows them as "your own music".

## Pose tracking

- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) Pose
  Landmarker, Apache License 2.0.
- Pose models: BlazePose GHUM 3D
  ([model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)),
  Apache License 2.0. Out of scope per the model card: surveillance and
  identity recognition (the game does neither).
- MediaPipe sends Google usage statistics (no images); see the
  [MediaPipe Tasks privacy notice](https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice)
  and the game's privacy screen (`P` in the menu).

## Software

- [pygame-ce](https://github.com/pygame-community/pygame-ce): GNU LGPL 2.1
- [NumPy](https://numpy.org/): BSD 3-Clause
- [OpenCV](https://opencv.org/) (a MediaPipe dependency): Apache License 2.0
- [librosa](https://librosa.org/) (beat-map tools, not needed to play): ISC

## Dance moves

The building blocks follow EKS-D's
["How to Dance"](https://kirakira.org/XD/learn/how-to-dance/) guide (the
wrist roll is not in it and was checked against a real dancer instead).

## Logo

The ParaParaPy logo (`assets/branding/`) was designed by the ParaParaPy
creator.
