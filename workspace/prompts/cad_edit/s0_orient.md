# S0 — Upright the part

Goal: place the part as a person would normally set it down, so later renders of the four base views (`iso` / corner views) are stable. This step **only uprights; it does not edit**.

## Inputs

1. **Seven probe images** (raw STEP, not yet upright) — 1 iso + 6 orthographic faces:
   - `[_orient_probe_iso]` corner (1,1,1) looking at center
   - `[_orient_probe_front]` looking from −Y (+Z up)
   - `[_orient_probe_back]` looking from +Y (+Z up)
   - `[_orient_probe_left]` looking from −X (+Z up)
   - `[_orient_probe_right]` looking from +X (+Z up)
   - `[_orient_probe_top]` looking from +Z (+Y up)
   - `[_orient_probe_bottom]` looking from −Z (+Y up)

2. **Edit Description** (text): helps identify the part and which face is the everyday resting / working face. Use it to judge up/down. **Do not** edit the model from the description — openings, sizes, etc. belong to later steps.

Axis names are the current STEP coordinates, not post-upright world axes.

## Process

1. From the probe images + Edit Description, identify the object
2. Decide how a person would set it down: which face up, which face down (do not treat the current on-screen orientation as the answer; words like “bottom / back / top” in the description can help)
3. Map the downward face onto the current axes and write `bottom_axis`
   - Use front/back/left/right/top/bottom to match axes: e.g. if the bottom face is the large central plane in the `top` image and corresponds to the `bottom` image, the bottom normal is likely ±Z

After this, the code rotates `bottom_axis` to −Z and captures the four official corner views.

## Output

Return **only** one JSON object, no other text:

```json
{
  "object": "short object name",
  "top_face": "up-facing face/feature",
  "bottom_face": "down-facing face/feature",
  "bottom_axis": "-Y",
  "already_upright": false,
  "reason": "one sentence (how the description confirmed up/down)"
}
```

`bottom_axis` must be one of: `+X` `-X` `+Y` `-Y` `+Z` `-Z`
Set `already_upright` true only when the resting face already points down and matches a natural placement; otherwise false.
