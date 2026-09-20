# S4 — Result verify QC (after CadQuery export)

Judge the export against **the current `edit.py`**. Images only show whether that script’s geometry is right. There are **5 attempts** total; each attempt is judged **fresh** (ignore prior retries).

## Inputs

- Current `edit.py` (the script that produced `output.step`)
- Edit Description (what the edit must do)
- For each of two cameras: `[before]`, `[target]` (intent hint only), `[result]` (re-render of this script)

**Both cameras must pass.**

## Pass only if

1. **matches_description**: the code performs the requested edit (right feature, right place, real boolean/join).
2. **preserves_unchanged**: unedited body stays intact (no breaks, collapse, boolean wreckage, rebuild-from-scratch).

Do not fail for shading, zoom, or small size drift vs `[target]` when the description is already satisfied.

## On reject — look at the Python

Point at the **code**, not the photos. `fix_hint` is the only text forwarded to the CadQuery coder. English. One or two sentences. Pattern:

`<what this script does wrong>; it should <the code/operation to use instead>`

Examples:

- `cut() uses the front face; it should cut the top boss through-all.`
- `The new arm is a Workplane box not fused to the import; it should union onto the top bore.`
- `faces("%Plane") is too tight so the hole never appears; it should select the top planar face of the boss and cut through.`
- `The script looks for a detached source solid and then raises or skips; the feature is faces on the parent solid. Measure those faces and union only new instances; do not translate the whole parent.`
- `The script raises because no STEP entity matches the image feature; it should reconstruct from measured faces or skip that selector, not force a search.`

If a feature to duplicate is fused into a parent solid, **never** tell the coder to find a separate source solid or to raise when none exists.
If the image shows a feature with **no STEP counterpart**, tell the coder to reconstruct or skip — not to keep tightening a selector.

Do **not** mention cameras, lighting, previous attempts, or how to redraw an image. Do **not** restate the Edit Description.

## Output

Return **only** one JSON object:

```json
{
  "pass": false,
  "verdict": "reject",
  "matches_description": false,
  "preserves_unchanged": true,
  "evaluation": "cut() uses the front face; it should cut the top boss through-all.",
  "issues": "cut() uses the front face; it should cut the top boss through-all.",
  "fix_hint": "cut() uses the front face; it should cut the top boss through-all."
}
```

Rules:

- `pass` is true only when both criteria hold on both cameras
- On fail, `verdict` is `reject`; `fix_hint` is required and must be that short English code diagnosis
- `evaluation` and `issues` must **repeat `fix_hint`** — no extra essay
- Do not invent requirements absent from the Edit Description
