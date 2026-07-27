# extracts

Passages referenced by `findings/davinci-002-creative-tests.md`, kept so the
numbers there are reproducible.

- `davinci002_song_mode_raw.txt` — the song-mode passage exactly as generated,
  note glyphs intact. Use this one for `caption_furniture()`: it carries the
  5 empty `♪♪` spans and the absence of any other caption apparatus.
- `davinci002_song_mode.txt` — the same passage lineated on the note
  delimiters, one lyric span per line. Use this for the rhythm metrics and for
  `form_drift()`, which need line structure.
- `davinci002_prose_head.txt` — the same session before the first note glyph.
  The within-session control: same seed, same hand, same sampling regime,
  no song mode.

All three are `davinci-002` output, not human writing. The seed prompt is not
included.
