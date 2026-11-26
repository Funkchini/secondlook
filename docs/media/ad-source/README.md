# SecondLook 30s launch ad

Code-built animation of the approved 30-second script (nine scenes, applicant #135305).

- `ad.html`: the whole ad as one HTML page; `render(t)` draws the frame at time `t` seconds.
- `music.py`: synthesises the music bed and sound accents into `music.wav` (no samples).
- `render.js`: renders 900 frames (30 fps) with Playwright and encodes the MP4 with ffmpeg.

All numbers on screen come from the app's own demo data (`app/demo_data`). Brand: Deep Navy #0A2E52,
Fintech Teal #21A6B8, status colours only for decisions, Montserrat + Inter, logos from `app/static` unchanged.

To rebuild: copy `app/static/logo_*.png` and the Montserrat/Inter woff2 files (URLs in `.streamlit/config.toml`)
next to `ad.html` as `montserrat.woff2` / `inter.woff2`, then run from a folder holding `build/` (these files)
and `out/`: `python3 music.py && node render.js`.
