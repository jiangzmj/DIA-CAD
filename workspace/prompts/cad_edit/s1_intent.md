# S1 Turn 1 — Design intent (fixed views + Edit Description)

You are given **4 fixed CAD images** of the **input** model
(`iso`, `z_corner`, `x_corner`, `y_corner`), an Edit Description, and a
**view direction legend** (front/back/left/right/up/down mapping for each file).
Each attached image is preceded by a caption such as `[iso]`.

The model has already been uprighted so **+Z is natural up** in the images.

This turn writes a **delta**, not a new part description. Later image generation
and QC treat the CAD PNGs as identity. If you restyle existing features here,
downstream images will copy the restyle.

## Two vocabularies — do not mix them

1. **Legend front/back/left/right/up/down** — only names **which camera sees which side of the AABB** after upright. Use this to read the four PNGs and to pick two `base_views`.
2. **Edit Description English** (e.g. `mounted from the top`, `from above`, `underside`) — **design attachment language** relative to the **part’s own features** (which face/boss/hole the new geometry connects to). It is **not** a camera label, and it does **not** mean “extrude along world +Z / screen-up”.

Never translate description phrases like “from the top” into `primary_direction: "up"` or “rod straight toward screen-up” solely because legend up = +Z. First find the **attachment interface on the part in the images**, then decide the new feature’s axis from that interface’s local geometry.

## Fixation rod / arm length (do not collapse to a straight stick)

When the Edit Description asks for a **fixation rod/arm** with **arm length** (or similar reach) **to fix / hold a product**:

- `mounted from the top` = **where it mounts** (e.g. into the top boss/bore), **not** “the whole rod sticks straight up along +Z”.
- `arm length N mm` = **reach of the working arm** (how far it extends to the workpiece), **not** the length of a coaxial pin sticking out of the bore along the bore axis.

## Priority

1. **Identity** — unmentioned features stay as they appear in the CAD PNGs (cross-section, attachment face, solid vs hollow, fillets, how a feature joins the body). Do not reclassify them.
2. **Delta** — Edit Description says what may change (add / remove / resize / mirror / pattern / join).
3. **Do not invent** features that are not visible in the CAD images and not jointly required by the description.

When the description conflicts with clearly visible CAD geometry, keep the CAD identity and apply only the described operation.

Copy / mirror / pattern / duplicate: the copies must look like the **source feature in the CAD PNGs**, not a redesigned substitute. The change is placement / count / boolean join, not a new feature type.

## Process

### Step 1 — CAD identity

From the four views, list existing features **as seen** (do not upgrade them to a more specific type you cannot confirm).

### Step 2 — Description delta

What the Edit Description actually requires. Everything else is `keep`.

### Step 3 — Decision output

Emit a single JSON object (no prose outside JSON):

- **keep**: existing features / body that must remain visually the same
- **change**: only the described operation (what is added, removed, mirrored, patterned, joined)
- **shape** / **position** / **purpose**: details of that delta
- **base_views**: exactly **two different** views that together **show the attachment / edit region** from complementary angles — each must be `iso` | `z_corner` | `x_corner` | `y_corner`. Pick by which PNGs show that region clearly; **do not default to iso**; do not repeat the same view.

If a feature or parameter cannot be confirmed from the images, mark it `"unknown"` — do **not** invent values.
