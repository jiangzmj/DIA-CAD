# S1 — Target pair QC (same edit, two cameras, matches description)

The two targets were generated **independently**, with **no per-image QC**. Judge them **together**: whether they show **the same 3D edit of this CAD part** from two cameras, **and** whether each matches the Edit Description.

Ground truth is each camera’s **CAD before PNG + Edit Description**. Two targets that match each other but not the CAD part / description must **fail**.

## Inputs

- Edit Description
- Chosen `base_views` (exactly two cameras)
- For each camera, in this order:
  1. CAD **input** PNG (unedited part, that camera)
  2. generated **edited** target from that camera (may be missing)
- View legend — only to read which CAD PNG is which side

If a camera’s target is missing, that view fails.

## Criteria (all must hold to pass the pair)

1. **matches_description** / **cad_consistent**: each present target is still that camera’s CAD part with **only** the described delta. Fail a view if it restyles unmentioned features (feature type, attachment face, invented openings, dropped joins) or does not show the described edit.
2. **same_edit**: both targets add/remove/resize the **same** features, on the **same** part interface, with the same overall new geometry. It must look like one object, two photos.
3. **viewpoint_only**: visible differences are explained by camera angle / occlusion. Fail if one target has a different arm, extra/missing hole, different attachment face, or a redesign the other view does not show.

`same_edit` alone is not enough. A view can pass individually while the pair still fails correspondence.

## Do not fail / pass for the wrong reasons

- Pose, framing, and which faces are visible **should** differ — that is the point of two cameras.
- Ignore shading, lighting, background, and minor edge-quality differences.
- Do not require both images to be pixel-similar or the same crop.
- `mounted from the top` is attachment language, not “must point screen-up”.

## If they fail

Report **per-view** pass/fail in `view_pass`. Then:

- **One view ok, one not**: set `keep_view` to the good camera and `retry_view` to the bad one. The kept image will guide regenerating the other camera.
- **Both fail description / CAD identity**: set `both_fail` true, leave `keep_view` null, and put **both** cameras in `retry_views`. Do **not** pick a keep just to have one.
- **Both match description but do not correspond**: pick the better CAD/description match as `keep_view` and the drifted one as `retry_view`.

The controller QCs the pair at most twice. After a second failure it keeps only a passing view, or drops both so the harness proceeds from CAD views + description alone.

## Output

Return **only** one JSON object (no markdown fences, no prose):

```json
{
  "pass": false,
  "cad_consistent": false,
  "matches_description": false,
  "same_edit": false,
  "viewpoint_only": false,
  "both_fail": false,
  "view_pass": {
    "z_corner": true,
    "iso": false
  },
  "issues": "what differs between the two targets, or how they left the CAD part / description",
  "fix_hint": "how the failed camera(s) should match the description and (if one is kept) the kept view",
  "keep_view": "z_corner",
  "retry_view": "iso",
  "retry_views": ["iso"]
}
```

Rules:
- `pass` is true **only** when both views pass `view_pass`, and `cad_consistent`, `matches_description`, `same_edit`, and `viewpoint_only` are all true
- `view_pass` must include both `base_views`
- On one-view fail: `keep_view` and `retry_view` must be the two `base_views`, and they must differ
- On both-fail: `both_fail` true, `keep_view` null, `retry_views` both cameras
- Do not invent requirements absent from the Edit Description
