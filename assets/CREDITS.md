# Project-page assets

## Recorded rail-slide sound

`rail-slide.wav` is an edited excerpt of **Drawers with metal runners** by
**Elemes**, published January 16, 2019 on Freesound:
https://freesound.org/people/Elemes/sounds/456731/

License: **CC0 1.0 Universal** (public-domain dedication).
https://creativecommons.org/publicdomain/zero/1.0/

Source: the author's high-quality MP3 preview,
https://cdn.freesound.org/previews/456/456731_4059103-hq.mp3

This is a real recording of an Ikea chest of drawers on metal runners,
recorded with a Zoom Q3HD, with background-noise reduction by the author.
The page uses 29.58–30.28 seconds: converted to mono, high-pass filtered at
140 Hz, low-pass filtered at 5.2 kHz, gently compressed and level-adjusted,
with short end fades. `rail-sound.js` crossfades the loop seam, blends a
reversed copy for the opposite direction, and fades playback with scrolling.
No synthesized noise or oscillator is mixed into the sound.

`rail-recording.js` contains a base64 copy of `rail-slide.wav` so direct
`file://` previews can decode the recording without a cross-origin fetch.
Regenerate this file if the WAV changes; keep the audio bytes identical.

## Hugging Face icon

`huggingface.svg` is the official Hugging Face logo, downloaded from
https://huggingface.co/front/assets/huggingface_logo.svg

Brand assets: https://huggingface.co/brand
Used to identify the project's dataset hosted on Hugging Face.

## GitHub and arXiv icons

`github.svg` and `arxiv.svg` use the brand paths from Simple Icons,
distributed under CC0 1.0 Universal. Their fill is set to white for the
project page's dark link buttons.

- https://github.com/simple-icons/simple-icons/blob/develop/icons/github.svg
- https://github.com/simple-icons/simple-icons/blob/develop/icons/arxiv.svg
- https://github.com/simple-icons/simple-icons/blob/develop/LICENSE.md

The marks identify the code repository and paper; their respective owners
retain their trademark rights.
