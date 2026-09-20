# S1 — Target render QC (unused standalone)

S1 no longer QCs each generated target by itself. After **both** cameras are rendered, `s1_target_pair_qc.md` judges correspondence **and** Edit Description match together.

The text below is kept only as the per-image rubric that pair QC still applies to each camera.

---

# S1 — Target render QC

Judge whether this `target_<base_view>.png` is an acceptable **edited** appearance of **this CAD part** from the chosen base camera.

Ground truth is **the CAD PNGs + Edit Description**. Any earlier design-intent JSON is **not** ground truth. If the target matches a restyled paraphrase but not the CAD part, **fail**.

## Inputs

- Edit Description text (the **only** allowed delta — feature/attachment intent, not camera instructions)
- CAD reference views of the **input** model — identity of the unedited part
- Chosen `base_view` for this generation
- Last two images: `[CAD same camera]` then `[target]` — compare those two first
- Other CAD views are supporting identity only
- View legend — only for which CAD PNG is which side

## How to judge (do this in order)

1. **Identity** — looking at the same-camera CAD PNG: what is the body, how do existing features attach, solid vs opening, fillets/joins. The target must still be that part.
2. **Allowed delta** — only what the Edit Description asks (add / remove / resize / mirror / pattern / join). Copy operations must duplicate the CAD source feature, not substitute a different one.
3. **Camera** — target pose/silhouette framing must match the chosen CAD PNG. A different angle is a fail even if the design looks plausible.
4. **Unmentioned restyle is a fail** — new feature type, moved attachment face, invented holes, dropped fillets, rebuilt body.

`matches_description` is not “looks edited”. It is “the described operation is visible **on this CAD part**”.

## Criteria (all must hold to pass)

1. **matches_description**: the described delta is present, on the correct part interface from the CAD views.
2. **preserves_unchanged**: unmentioned geometry stays as in the CAD views (no invented edits, no random shape drift).
3. **camera_matches_base_view**: same camera / part pose as the chosen CAD PNG.
4. **same_part_identity**: still the same part; not a redesigned substitute that merely has a similar story (e.g. “there are two of something”).

## Do not fail / pass for the wrong reasons

- Phrases like `mounted from the top` do **not** require the new geometry to point screen-up or along world +Z; they require correct **attachment to the part’s top interface** as seen in the CAD views.
- If the Edit Description has **arm length** + fix/hold product language, a **straight coaxial stick** only upward is usually **wrong**; expect an **L/bent reach arm** unless the description clearly asks for a straight pin.
- Ignore minor shading/lighting/background differences. Do **not** ignore feature-type or attachment-face changes.
- This check is **per image**. A later pair QC compares the two targets as the same 3D edit — that later check cannot rescue a restyle.

## Output

Return **only** one JSON object (no markdown fences, no prose):

```json
{
  "pass": false,
  "matches_description": false,
  "preserves_unchanged": true,
  "camera_matches_base_view": false,
  "same_part_identity": false,
  "issues": "short list of failures",
  "fix_hint": "concrete guidance for regenerating the target image"
}
```

Rules:
- `pass` is true **only** when all four criteria are true
- Do not set `pass` true with empty issues if any criterion is doubtful — fail and say what to keep from the CAD PNG
- If failing, `fix_hint` must say what to fix next (what to add/remove/keep **relative to the CAD PNG**)
- Do not invent requirements absent from the Edit Description
- Do not invent a “must be vertical” requirement from the word `top`
