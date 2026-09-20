# S1 Turn 2 — Target render direction

This turn **does not generate an image**. Pick the two base images for later img2, and specify how the edited appearance should be photographed / what must be visible.

## Inputs (what the code actually feeds)

- **Previous Turn 1 reply**: the design intent in this session (`s1_intent.md` output)
- **4 CAD images**: `02_views/` `iso` / `z_corner` / `x_corner` / `y_corner` (already attached on Turn 1; this turn is text-only, relying on session context)
- **View direction legend**: `view_legend.md` (spliced into this turn)
- **Edit Description**: `description.txt` (already given on Turn 1; the intent should reflect it)

## Task

From the inputs above, pick **two different** `base_views` as the primary bases for the two later img2 runs (one target each). Together they should show the attachment / edit region from complementary angles.

`must_show` is **delta only** (what the description requires). Unmentioned appearance must match the CAD base images; do not rewrite the part type in `must_show`.

If Turn 1 restyled an existing feature into a different feature, this turn follows the CAD images and drops that restyle.

Copy / mirror / pattern: copies must look like the CAD source feature; only placement / count / join may change.

## Requirements

- Change only what the intent and Edit Description support; unmentioned features stay.
- **Pick two `base_views`**: both show the attachment / edit region, and they **differ**; do not default to `iso`; do not pick the same corner twice.
- Prefer complementary pairs (e.g. top edit: `z_corner` + `iso`; bottom: `x_corner` + `y_corner`).
- Each generation uses the same camera as its `base_view`: sharp silhouette, part centered, white/neutral background, technical shading (not artistic lighting).
- Phrases like `mounted from the top` in the Edit Description mean **which interface on the part**, not “draw it pointing toward the top of the image”. Pose must match the CAD views.
- If the intent is an **L / bent arm + arm length**: `must_show` must reveal the bend and the reaching working arm, not only a straight pin along the top-bore axis.
- Dimensions marked `unknown` should look plausible; do not invent precise labels.

## Output

Return **only** one JSON object (no markdown fences, no prose):

```json
{
  "base_views": ["z_corner", "iso"],
  "base_reasons": {
    "z_corner": "attachment region is clearest in this PNG",
    "iso": "second angle that still shows the edit without hiding the join"
  },
  "camera": "same angle as each chosen base_view",
  "shading": "olive shaded technical, white background",
  "keep": "unmentioned CAD features must look like the base PNG",
  "must_show": "only the described delta, visible after the edit"
}
```

`base_views` must be exactly 2, each one of: `iso` | `z_corner` | `x_corner` | `y_corner`. No duplicates.

The code runs img2 **independently** on each base (they do not see each other) to write `03_target_render/target_<view>.png`. There is **no per-image QC**. After both exist, a helper checks that they correspond (same 3D edit, two cameras) **and** match the Edit Description. If only one passes, that image guides regenerating the other, then the pair is QCed once more. If still only one passes, only that target is passed to the harness. If both fail twice, **no** target is passed; the harness continues from CAD views + description.
