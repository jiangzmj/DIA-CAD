# S2 — Plan intent (think first; do not write CadQuery)

You propose a frozen edit plan. The harness will ask for CadQuery on the **next** turn. You do **not** declare success. You do **not** write code on this turn.

If the supplied views do not reveal the feature, return `action: "REQUEST_VIEWS"` with
`requested_views` chosen from the camera catalog in the request. The controller renders
them and asks for the intent again. Do not guess a camera name.

## Inputs (already attached, labeled)

1. **Edit Description** — what must change, and what must stay.
2. **STEP geometry summary + FeatureProbe candidates** — OCP facts and optional Palmetto recognizer candidates. Candidate ids are evidence, not ground truth. **Never** select `faces[i]` or `.vals()[n]`.
3. **Four original CAD views** — role=`base_view`.
4. **Target reference images** (zero, one, or two; role=`target_reference`). They may be absent if pair QC dropped them. When present, use them only for the **described delta**, and only if they still look like the same CAD part. Unmentioned CAD features stay as in the `base_view` images. If a target restyles the body, ignore that restyle. If none are attached, plan from the Edit Description and the four CAD views.
5. **Previous diagnosis** (if any) — if this is a replan after visual mismatch, fix **where** / **what**, not CadQuery.

## Decide, in order

- **target_description** — quote/paraphrase the user's target in geometry language without choosing a modeling operation.
- **change_type** — `add` | `remove` | `resize` | `move` | `duplicate` | `modify` | `unknown`.
- **what** — feature type, quantities and dimensions. Mark unmeasurable values `"unknown"`. Do not invent numbers.
- **where** — target face/edge/solid: `semantic_location` and stable `candidate_refs`, axis, origin. Use candidate ids only when their measurements and the images agree.
- **target_relationship** — `detached_solid` | `fused_feature` | `ambiguous` | `unknown`. This is your interpretation, not a tool decision. Explain uncertainty in `ambiguities`.
- **how** — one registry operation (not geometry primitives): `add_hole` | `add_boss` | `cut_slot` | `extend_cylinder` | `chamfer` | `fillet` | `modify_feature`. Optional `pattern` `{kind, count, around, increment_deg}`.
- **why** — one sentence of function, not a modeling recipe.
- **spatial_constraints / geometric_constraints** — explicit location and measurement evidence.
- **preserve_constraints** — unmentioned structure that must remain unchanged.
- **ambiguities** — unresolved interpretation or missing measurement. Never silently guess.
- **constraints** — legacy mirror: `keep`, `forbidden`, `unknowns`.

Treat the user's edit verb as evidence. `create` / `add` normally means new geometry; do not
silently reinterpret it as resizing or replacing a nearby existing feature merely because
that feature is at a similar semantic location. When the target shows a new instance of an
existing design, keep **copy source** and **destination** separate: compare all plausible
sibling solids by morphology and dimensions, then place the selected design at the requested
destination. A candidate near the destination is not automatically the source.

Treat collision removal as an outcome, not automatically as a subtractive edit. Compare the
base and target views for the involved component: if a detached component keeps the same
silhouette and dimensions but changes position, freeze `change_type: "move"` and preserve its
shape; if its shape changes locally, freeze the corresponding remove/resize intent. Use
FeatureProbe `solid_relations` as collision evidence, while still using the target to decide
which component is allowed to change.

Do not infer a world axis from words such as “above”, “front”, or “vertical” alone. Resolve the
image direction through the supplied camera/view metadata and confirm it against measured
candidate centers and boundary sides. For copies around an existing feature, separate the
source center from the copy translation axis in the frozen intent.
Named cameras are controller aliases, not semantic claims about the product: `front` is the
upright-render -Y axis view. Images use `upright_render`; FeatureProbe and CadQuery use the
untouched `original_step` frame. Convert screen directions with `step_look_from` and
`step_view_up` before freezing a modeling axis.

For patterns, resolve the grammatical and geometric scope together. “This design” means the
completed result of the first requested design operation. Choose the smallest repeatable unit
that matches both the target references and connectivity. If repeating only a terminal
bearing/hole/boss would create floating disconnected features while the target shows connected
radial branches, include the minimal source-derived support segment in the patterned unit.
Do not pattern the complete imported parent unless the request or target actually duplicates
that whole part. Put the resolved unit in `how.pattern.scope`, and state the expected final
connectivity in `target_relationship` and `preserve_constraints`.

If the request cannot be done with those operations, set `action: "UNSUPPORTED_EDIT"`. The controller confirms that claim.

## Output

Return **only** one JSON object. **No** `cadquery_code`.

```json
{
  "action": "PLAN_INTENT",
  "target_description": "bearing feature at the flat unfilleted end",
  "change_type": "duplicate",
  "target_relationship": "fused_feature",
  "candidate_refs": ["ocp_solid_000_face_0012_abcd1234"],
  "spatial_constraints": ["same bearing axis", "circular pattern around bearing axis"],
  "geometric_constraints": [{"instance_count": 8}, {"increment_deg": 45}],
  "preserve_constraints": ["filleted end", "imported arm profile"],
  "ambiguities": [],
  "what": {
    "feature": "annular_bearing",
    "parameters": {"outer_radius_mm": 10, "inner_radius_mm": 5, "height_mm": 6, "instance_count": 8}
  },
  "where": {
    "semantic_location": "flat_end_without_fillet",
    "feature_id": "",
    "candidate_refs": ["ocp_solid_000_face_0012_abcd1234"],
    "axis": [0, 0, 1],
    "origin_mm": [80, 17, 6]
  },
  "how": {
    "operation": "add_boss",
    "pattern": {"kind": "circular", "count": 8, "increment_deg": 45, "around": "bearing_axis", "scope": "smallest connected completed design shown in target"}
  },
  "why": "bearing array at the unfilleted end for mounting",
  "constraints": {
    "keep": ["filleted end", "imported arm profile"],
    "forbidden": ["do not replace the imported solid"],
    "unknowns": []
  }
}
```

Do not emit `box` / `cut` / `union` as `how.operation`. Those belong in CadQuery later.
