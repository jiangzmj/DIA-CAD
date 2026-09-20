# S2 — Structured edit.json

You are given multi-view CAD images of the input model, two target renders of the intended edit (one per chosen base view), and an Edit Description.

Produce **only** one JSON object (no markdown fences, no prose) that is schema-valid for the pipeline and encodes design intent.

## Hard schema (required)

```json
{
  "version": 1,
  "edit_type": "modify | add | remove | replace | other",
  "operations": [
    {
      "op": "string",
      "target": "string",
      "params": {}
    }
  ],
  "notes": "optional short string"
}
```

Each item in `operations` **must** include `op` and `target`. Put quantitative / relational detail under `params`.

## Design-intent content (required inside the same JSON)

Also include a `design_intent` object with:

- **shape**: geometric operations, feature types, parameters, quantities, boolean relations
- **position**: target faces/edges, directions, datums, relative placement, symmetry/pattern
- **purpose**: functional check that the modified shape and mounting realize the intended use (not operation-level instructions)

## Priority rules (same as intent stage)

- CAD `base_view` images are part identity. Unmentioned features stay as there.
- Edit Description is the allowed delta.
- Target renders constrain the changed region only if they still look like that CAD part; ignore a restyled body.
- Do not invent features absent from images and not jointly supported by the Edit Description.
- Unmeasurable values → `"unknown"`; do not fabricate dimensions.

Map each concrete edit into `operations[]`, and keep the richer shape/position/purpose under `design_intent`.
