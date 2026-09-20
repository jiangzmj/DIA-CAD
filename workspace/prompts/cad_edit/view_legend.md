# CAD four-view direction legend (after upright)

The part is already upright: **+Z = up, −Z = down** (camera directions for these four images only).
Body-relative axes (relative to the part AABB):

| Axis | Direction |
|------|-----------|
| −Y | **front** |
| +Y | **back** |
| −X | **left** |
| +X | **right** |
| +Z | **up** |
| −Z | **down** |

The four official views look toward the AABB center from unit-box corners (not pure six-face orthographic). File names:

| File / caption | Camera corner (relative to center) | Main facing (front/back/left/right/up/down) | Best for |
|----------------|--------------------------------------|---------------------------------------------|---------|
| `[iso]` / `iso.png` | (+X, +Y, +Z) | **right · back · up** oblique | overall shape |
| `[z_corner]` / `z_corner.png` | (−X, −Y, +Z) | **left · front · up** oblique | front + top |
| `[x_corner]` / `x_corner.png` | (+X, −Y, −Z) | **right · front · down** oblique | front + bottom |
| `[y_corner]` / `y_corner.png` | (−X, +Y, −Z) | **left · back · down** oblique | back + bottom |

## How to pick base views

Choose **two different** PNGs. Only criterion: **where the edit is on the part, and which two complementary cameras show that region**. Do not default to `iso`. Do not pick the same view twice.

- Need **top / top boss / top hole** → `z_corner` + `iso`
- Need **front** → `z_corner` + `x_corner`
- Need **bottom** → `x_corner` + `y_corner`
- Need **back** → `y_corner` + `iso`

## Boundary with the Edit Description

This table's front/back/left/right/up/down **only names CAD cameras**. It does not interpret English in the Edit Description.

- `mounted from the top` / `from the top` = the new feature attaches to the part's **top mounting interface** (find that face/hole in the images)
- It does **not** mean “the new feature must grow toward the top of the screen / world +Z”
- Pick two `base_views` to see that interface. Do not treat the word `top` as “extrude along legend up”
