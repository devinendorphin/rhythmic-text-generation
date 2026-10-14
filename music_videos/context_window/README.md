# Context Window — music video

A 2:29 audio-reactive video for the Suno track *Context Window*. The text in the
video is the raw generated text that went into Suno's lyrics field
([`lyrics.txt`](lyrics.txt)), typed into a fixed monospace window in time with the
track.

- **`Context_Window.mp4`**: 1080p30, compressed to 28 MB. The film grain is smoothed
  out at this size.
- **Full-quality master**: 283 MB, with the grain intact. It's attached to the
  [`context-window-v1` release](../../../../releases/tag/context-window-v1) because git
  refuses files over 100 MB.

## What happens

| Time | Music | Video |
|---|---|---|
| 0:00–0:04 | intro | The window draws itself in from its corners while the cursor blinks |
| 0:04–0:25 | first section | Paragraph 1 types in, paced by spectral flux. Each glyph cycles through random glyphs before it settles |
| 0:25 | section change | The bracketed line lands as an amber header bar |
| 0:25–0:48 | build | Older glyphs cool and dim. On kicks, arcs reach from the cursor back to earlier glyphs and light them up |
| 0:48.35–0:49.25 | bass dropout | The lone `/` gets the frame to itself while everything else dims |
| 0:49–1:39 | main body | The window fills, and lines scroll off the top as particles. The plain-English lines type steadily and hold still |
| 1:39–1:53 | low section | The window shrinks from 12 rows to 6, and the lines it drops never come back |
| 1:53–2:27 | final push | The window reopens, with more flicker, row tears and colour fringing |
| 2:27 | music cuts | Text decays back into noise, oldest first. The cursor blinks twice, then holds |

The counter under the window shows the number of glyphs currently in view.

## Rebuilding

The renderer uses only numpy, scipy, OpenCV, Pillow, fontTools and ffmpeg. There is
no ML step: the timing comes entirely from signal features of the audio.

```bash
cp /path/to/Context_Window.wav song.wav
python3 analyze.py                         # writes feat.npz (RMS, bass/mid/hi bands, spectral flux at 30 fps)
python3 mv.py still 25.2 48.6 106.0        # test frames → still_<t>.png
python3 mv.py range 0 2242 part1.mp4 &     # render in two halves
python3 mv.py range 2242 4485 part2.mp4
printf "file 'part1.mp4'\nfile 'part2.mp4'\n" > list.txt
ffmpeg -f concat -safe 0 -i list.txt -i song.wav -map 0:v -map 1:a \
  -c:v libx264 -crf 22 -pix_fmt yuv420p -c:a aac -b:a 320k -shortest Context_Window_master.mp4
```

Section times (`BREAK0/1`, `LOW0/1`, `T_END`) and when each paragraph types
(`SCHED`) are hand-set from the audio analysis, at the top of `mv.py`. Glyphs fall
back across DejaVu, FreeFont, Noto CJK, Unifont and Noto Color Emoji (emoji are
rendered monochrome). Codepoints that no font covers render as a box.
