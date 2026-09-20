# S3 — CadQuery code generation

Generate runnable CadQuery Python that builds the edited 3D CAD model from the structured edit intent.

## Inputs

- `edit.json` (design intent / operations) provided below
- Reference model: `input.step` at the given path
- CAD views of the original part (`[iso]`, `[z_corner]`, …)
- Target renders of the intended edit (`[target_<view>]`, one per chosen base_view)

CAD views are part identity. Target renders hint the described delta only; ignore a restyled body.
Do not invent features absent from the images and not jointly supported by `edit.json` / the Edit Description.

## Reference model policy

Read and use `input.step` as the reference for overall size, structural layout, interface locations, assembly relationships, and geometric style.
Unless the design intent explicitly requires a change, **preserve** key structural features and mounting interfaces from the reference model.
**Never** replace the imported part with a placeholder box or a from-scratch primitive that is not the original solid.

## Code requirements

- Valid CadQuery Python only
- Define a top-level variable named `result` (Workplane / Solid / Compound) for export
- Prefer loading/editing the reference STEP when practical; otherwise rebuild from `edit.json` while matching reference scale and interfaces
- Do not invent geometry that is not supported by `edit.json` and the target renders
- No `exec` / `eval` / network / subprocess
- Prefer resilient face/solid selection: if the first candidate face is too small or missing, widen the search across solids instead of raising `ValueError` and exiting
- If the STEP contains coincident duplicate panels at the edit site, apply the boolean to **all** overlapping solids so one twin cannot hide the cut

## Image feature with no STEP counterpart

If a target/CAD image shows a feature that **has no matching B-Rep entity** (no detached solid, no face of that kind to select):

- Do **not** force a selector until it fails or `raise`
- Reconstruct from measured nearby faces, or from the image/intent dimensions, then `union`/`cut` onto the import
- Skip that selector entirely when the feature is decorative / not in the solid at all

## Duplicating a feature that is fused into a parent solid

When the feature to pattern/copy is **already boolean-unioned into a larger body** (one solid, not a detached source solid):

- Do **not** require a separate source solid, and do **not** `raise` when that search is empty
- Do **not** translate or copy the whole parent solid (that duplicates the entire part)
- Read the feature from **faces on the parent** (type, axis/plane, size, height)
- Build **only the new instances** from those measurements, then `union` them onto the imported solid
- Copying an existing sibling solid is valid only if the STEP actually has that solid detached
- Added supports (ribs, legs, gussets) must be the requested secondary features — not a stand-in for the patterned feature, and not oversized blobs

## Output

Return a single CadQuery script (prefer one fenced ```python``` block). Do not export STEP yourself — only produce the code that defines `result`.
