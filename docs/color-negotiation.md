# Color negotiation, render intents, and brightness

> **Status.** Increments A + B landed: render intent is threaded to the input
> stage; `perceptual`, `relative`, and `absolute` are advertised and honored via
> the two input-stage knobs — white-point adaptation (absolute carries source
> white verbatim) and reference-white anchoring (perceptual/relative anchor the
> content's reference white to the per-output reference-white level;
> `decode_luminance_scale`). The per-output level is config-based, defaulting to
> 80 nits (SDR) / 203 nits (HDR, BT.2408) so HDR/PQ content stays ~pass-through
> while SDR-on-HDR brightens to match. ext-linear/scRGB keeps its literal
> encoding scale (not anchored). The out-of-gamut/over-peak operator stays the
> panel LUT's measured degradation. **Hardware (2026-05-29):** SDR-on-HDR
> brightening and MPV HDR both look correct on DP-1 (OLED). A Firefox-HDR
> dim/dull report was traced to a *Firefox update* regressing its own HDR
> compositing (rolling Firefox back fixed it; MPV HDR through the same anchored
> path is fine) — not prism. A speculative PQ→pass-through revert was made and
> then reverted back. The full anchoring (incl. PQ) stands until there's real
> evidence it misbehaves. Pending: `relative_bpc`, `saturation`,
> `absolute_no_adaptation`; EDID/ambient-derived brightness; per-intent LUT
> variants; mastering/target-volume tone mapping.

How prism turns the `wp_color_management_v1` negotiation into deterministic
on-screen behavior. [color-management.md](color-management.md) is the *why* of the
pipeline (decode-per-element / encode-per-output, the BT.2020 absolute-nits
intermediate, calibration). This doc is the layer on top: given what a client
declares about its content and how it asks for that content to be mapped, what
does prism actually do — and how much of that is *forced by the spec* versus
*ours to decide*.

The motivating problem is that there are a ridiculous number of degrees of
freedom in color mapping. The discipline of this doc is to sort them into three
bins and refuse to treat a determined quantity as a free one:

1. **Spec-determined / self-describing** — the protocol (via ICC intent
   semantics and its luminance model) defines the behavior. We implement it; we
   don't get a choice.
2. **Panel-determined** — fixed by what the panel can physically do, which the
   gamut-probe + LUT calibration measures exactly.
3. **Policy** — genuinely open. This is where the design effort and the
   configuration surface belong.

Most of the brightness/mastering work is bins 1 and 2 wearing a bin-3 costume.
The job is to shrink bin 3 to the few knobs that are truly subjective.

## What the protocol mandates vs. leaves open

The compositor's *hard* obligations are small (from
`color-management-v1.xml`):

- **Advertise** capabilities on bind: `supported_intent`, `supported_feature`,
  `supported_tf_named`, `supported_primaries_named`, then `done`.
- **Support the perceptual intent.** All others are optional
  (*"Compositors must support the perceptual rendering intent. Other rendering
  intents are optional."*).
- **Accept any *ready* image description** a client sets, as double-buffered
  surface state.
- **Expose** a per-output image description (what the output expects) and a
  per-surface *preferred* description (a hint, *"not automatically used for
  anything"*).
- **Handle unmanaged surfaces** — only a *should*: *"compositor implementation
  defined. Compositors should handle such surfaces as sRGB, but may handle them
  differently."*

Everything that makes content *look right* — the conversion from each surface's
color volume to each panel, tone mapping, gamut mapping, how SDR and HDR coexist,
where SDR white sits in nits — the protocol declines to specify. It hands us the
colorimetric *data* and the client's *intent* and gets out of the way. So our
responsibility is almost entirely the mapping policy; the wire contract is the
easy part (and is largely built — see color-management.md "What's wired today").

## The negotiated inputs (the self-describing channels)

A complete parametric image description carries these, and each one constrains
the mapping in a specific way:

| Channel | Set via | What it tells us |
|---|---|---|
| **Primaries + white point** | `set_primaries{,_named}` | The *primary color volume* — the chromaticity basis the pixel values are encoded against, **including the encoding white point**. |
| **Transfer function** | `set_tf_named` / `set_tf_power` | How to linearize the electrical signal to optical. |
| **Primary-volume luminance** (min/max) | `set_luminances` | The dynamic range the encoding spans (e.g. PQ ⇒ 0.005…10000). |
| **Reference white luminance** | `set_luminances` (`reference_lum`) | The luminance of *diffuse white* — the brightness facet of the white point. SDR ≈ 80–100, PQ default 203. |
| **Target / mastering volume** | `set_mastering_display_primaries`, `set_mastering_luminance` | The *actually used* color + luminance volume the content was mastered for. May be **smaller** than the primary volume. |
| **max_cll / max_fall** | `set_max_cll` / `set_max_fall` | Actual peak and frame-average content light level. |

Two channels deserve emphasis because they're the crux of this work:

**White point is two facets, communicated separately.** *Chromaticity* lives in
the primaries; *luminance* lives in `reference_lum`. Render intent decides what
to do with each — whether to chromatically adapt the chromaticity, and what to
anchor the luminance to. Having both facets explicitly is what makes the intents
mechanically implementable rather than guessed.

**The target/mastering volume is the mastering-negotiation lever.** The primary
volume is just the *container* (e.g. full BT.2020 / 10000-nit PQ). The target
volume is what the content actually occupies (e.g. P3 / 1000 nit). The spec is
explicit that the target volume *"is the actually displayable color volume"* and
may be smaller *"to minimize gamut and tone mapping distances."* So we tone/gamut
map from the **target** volume, not the container — which is tighter and more
faithful. `max_cll`/`max_fall` refine it further (true peak vs. theoretical
peak). Using these is how "mastering negotiation" actually pays off; ignoring
them means mapping a 1000-nit clip as if it were a 10000-nit clip and crushing it
needlessly.

The protocol also defines when colorimetry is **undefined** (and thus where we're
free to do anything): signal outside the TF range, tristimulus outside the target
volume, or outside the primary volume when `extended_target_volume` is
unsupported. Non-finite values are always undefined.

## Render intents as self-describing behaviors

The protocol defers to ICC.1:2022 for intent meaning and notes the principles
*"apply with all types of image descriptions, not only those with ICC file
profiles."* Crucially, an intent is not a vague mood — for the colorimetric
intents it is a **deterministic** function of the declared data and the panel
capability. Each intent sets three knobs:

- **(A) Chromatic adaptation** of the source white chromaticity to the panel
  white — on or off.
- **(B) Luminance anchoring** — map source `reference_lum` to the output's chosen
  reference-white level, *or* reproduce absolute nits literally.
- **(C) Out-of-volume operator** — how colors/luminances outside the panel's
  reachable set are handled: hard clip vs. perceptual compression.

| Intent | (A) Adapt WP | (B) Luminance | (C) Out-of-volume | Determined? |
|---|---|---|---|---|
| **perceptual** (mandatory) | yes | anchor to output ref white | **compress** (smooth gamut + tone roll-off) | curve is **policy** |
| **relative** (media-relative colorimetric) | yes | anchor to output ref white | clip to panel boundary | **deterministic** |
| **relative + BPC** | yes | anchor + map source black → panel black | clip | **deterministic** |
| **saturation** | yes | anchor to output ref white | compress, preserve saturation | curve is **policy** |
| **absolute** (ICC-absolute colorimetric) | no (media-white ratio only) | **absolute nits**, clip to peak | clip | **deterministic** |
| **absolute_no_adaptation** (v2) | no, none at all | absolute nits | clip | **deterministic** |

The four colorimetric intents are fully determined by (declared description +
measured panel). Only perceptual and saturation contain a free curve — and even
those are bounded: they must agree with the colorimetric intents in-gamut and
only differ in *how* they compress what's out of reach.

**Absolute is the faithful anchor, and the gamut mesh is what makes it exact.**
The user-facing promise of absolute is "show me the real colors, don't
reinterpret." That means: source chromaticity reproduced without adaptation,
source luminance reproduced in literal nits, and anything the panel can't reach
clipped to the panel boundary — *the measured boundary*, not a nominal gamut. The
gamut probe gives us that boundary as a measured mesh, so the clip is precise and
defensible (we clip to where the panel actually lands, hue-preserving), rather
than to an idealized BT.2020/P3 triangle the panel doesn't actually fill. This is
the single strongest argument that the calibration work we just finished is the
right substrate for intent handling: absolute colorimetric is *only* as good as
your knowledge of the destination volume, and we measured it.

## The panel as destination: capability from the gamut mesh

For every intent, the thing we map *into* is the panel's reachable volume. The
gamut-probe + LUT calibration characterizes it directly:

- **Reachable chromaticity boundary** — the measured cube-surface mesh, with
  per-patch trust and fold (clamping) detection.
- **Black point** — measured XYZ floor (for BPC and for the bottom of the tone
  curve).
- **Peak** — per-channel panel peak nits (the top of the tone curve / absolute
  clip ceiling).

This is bin 2: not a choice, a measurement. The mapping problem for any intent is
therefore "(source target-volume) → (measured panel volume)," with the intent
selecting the operator. Both the absolute clip and the perceptual compression
target *the same measured boundary*; they differ only in the operator applied
approaching it.

## Reference-white anchoring (the brightness keystone)

The spec's luminance model contains a quietly load-bearing requirement
(`set_luminances`):

> *Compositors should make sure that all content is anchored, meaning that an
> input signal level of `reference_lum` on one image description and another
> input signal level of `reference_lum` on another image description should
> produce the same output level.*

This is the whole brightness model in one sentence. It says: pick **one output
reference-white level** (in panel nits), and map every (non-absolute) surface's
declared `reference_lum` to *that same level*. An SDR app's 80-nit white and an
HDR app's 203-nit diffuse white then land on the same on-screen brightness, with
each app's content scaled around its own white accordingly.

- The **behavior** (all reference whites coincide) is spec-mandated.
- The **value** (what nit level is "white" on this panel right now) is the one
  genuinely subjective knob — the per-output brightness control. This subsumes
  the "SDR-on-HDR reference white nits (hardcoded 200 in niri; should be
  per-output)" item from color-management.md: it's not an SDR-on-HDR special
  case, it's *the* output reference-white level that everything anchors to.
- **Absolute intents are the documented exception** — a client asking for
  absolute is explicitly opting out of anchoring; its `reference_lum` is
  reproduced literally (clipped to peak).

So brightness management is: one per-output reference-white nit level + the
anchoring math, with absolute as the opt-out. That is far less open-ended than
"design a brightness system" makes it sound.

## Where each behavior lives in prism's pipeline

Mapping the three intent knobs onto the existing two-stage pipeline:

- **(A) White-point adaptation → the decode primary matrix.** `description_to_
  params` already builds the source→BT.2020 matrix; whether a Bradford CAT is
  folded in *is* the adapt/don't-adapt choice. **Today it always adapts**
  (`primaries_to_bt2020` Bradford-adapts any non-D65 white), so there is
  currently no way to express absolute/no-adaptation — the matrix has to become
  intent-aware.
- **(B) Luminance anchoring → the decode luminance scale.** Today `sdr_white_
  nits = reference_lum` literally, with no per-output anchor target. That's
  absolute-ish luminance behavior glued onto relative white-point behavior — a
  hybrid that corresponds to *no* named intent. Anchoring means scaling so source
  `reference_lum` → output reference-white level (except absolute).
- **(C) Out-of-volume operator → stays in the per-output LUT (by design).** The
  gamut-bake reform projects out-of-gamut/below-floor onto the measured boundary
  (hue-preserving) and rolls above-peak onto the white surface, inside the
  calibration LUT. It is one operator shared by all intents — which is acceptable
  (see "Architectural decisions"): intents diverge on (A)/(B), not (C), and a
  measured projection is a good shared clip. Where an intent eventually needs a
  *different* curve, the answer is a per-policy baked LUT variant, not moving (C)
  upstream into the hot loop.

**The crux is smaller than it looks, because decode is already per-output.** All
three knobs need the per-surface intent *and* the per-panel measured volume in
the same place. That place is the input/decode stage — and the structure to put
them there already exists: each output owns its own `Renderer` + `Intermediate`,
`render_frame` runs once per output, and a surface visible on two panels is
lowered and decoded *independently into each output's intermediate*. Decode even
already takes one panel parameter today (`output_peak_nits_rgba`, for its clamp).
So intent and panel-volume don't currently meet only because they aren't *threaded*
to decode yet — not because of a structural barrier. This is data plumbing plus a
bake re-split, not a pipeline rebuild.

## Architectural decisions (this round)

The split of responsibility is fixed as follows:

- **The intermediate is a stable BT.2020 absolute-nits metric, 1:1 with real
  light.** A value of *N* means "emit *N* nits of BT.2020 light," and the LUT
  guarantees that *any intermediate value in-gamut on a panel reproduces that
  exact calibrated color* — **panel-independently**. Same intermediate value →
  same real light on every panel. This is the universal contract and the space
  blending happens in.
- **Output LUTs realize the request and degrade gracefully — "make requests
  happen."** In-gamut, the LUT is a faithful device characterization (measured
  inverse panel model). Out-of-gamut / above-peak, it applies the *measured
  graceful degradation* we already built (hue-preserving projection onto the
  measured boundary; above-peak-Y rolled onto the white surface). The LUT
  therefore **does** carry the default gamut/tone operator — deliberately,
  because that operator is measured, good, and costs one trilinear tap.
- **The input/decode side decides what light to *request*.** It realizes the
  negotiated intent: TF decode, primaries matrix with an intent-driven CAT toggle
  for white point, and luminance anchoring to the per-output reference-white
  level. It is **panel-independent except for cheap per-output scalars** (peak,
  reference-white level) — it does **not** run a per-pixel mesh-based gamut cap.
  Out-of-gamut points are handed to the LUT's default. It only behaves
  *differently per panel* when a deliberate compromise is required (a gamut that
  genuinely cannot be delivered, or a negotiated white-point shift on a weaker
  panel) — policy, not a per-frame mesh query.

Why this division (the reasoning behind leaning on the LUT):

- **Correct order.** Gamut/tone mapping at the *output* of the large linear
  working space (blend-big, map-at-output) is more correct than capping on input:
  blending happens unconstrained in absolute light, and only the final per-panel
  realization is squeezed. Capping on input would gamut-map *before* blend — the
  worse order.
- **Hot-loop economics.** A per-fragment mesh boundary test in decode is exactly
  the cost to avoid, and it duplicates what the LUT already does downstream for
  free.
- **Intents mostly share one out-of-gamut operator.** Render intents differ
  primarily in white-point adaptation (A) and luminance anchoring (B) — both
  input-stage and cheap — and can share one OOG operator (C). A hue-preserving
  projection-to-boundary is a reasonable shared clip for relative and absolute,
  and an acceptable first cut for perceptual.

Consequences this commits us to:

1. **Intermediate contents are panel-independent except by deliberate policy.**
   In-gamut, the same client produces the same intermediate value and the LUT
   renders it identically everywhere. Contents diverge per panel *only* from
   chosen per-output policy (a different reference-white/brightness level, or an
   explicit gamut/white-point compromise) — never from a mandatory per-pixel cap.
2. **Blend-big, map-at-output.** Blending happens in the panel-independent
   working space (which may legitimately hold values out-of-gamut for a given
   panel); the LUT gamut-maps post-blend. Standard and correct.
3. **The gamut bake is leveraged, not re-split.** The measured-boundary
   degradation *stays in the LUT* as the shared default operator; the recent
   calibration work is the load-bearing substrate here, untouched. The mesh stays
   out of the hot loop — consumed at bake time, not per frame.
4. **Over-peak roll-off is largely already the LUT's job.** Input chooses the
   anchor (where reference white sits); the LUT rolls off whatever still exceeds
   this panel's peak (the above-peak-onto-white behavior). The decode
   hard-clamp-to-peak demotes to a safety net.
5. **Per-intent OOG differences, when they arise, become baked LUT variants.**
   The genuine exception is an intent wanting a *different curve* (true perceptual
   soft-compression that rolls off inside the boundary vs. colorimetric
   hard-clip-at-boundary). Because config-change LUT rebuilds are acceptable and
   LUTs are already per-output, the escape hatch is a per-policy baked LUT variant
   selected at encode — **not** a hot-loop mesh cap. Worst case is more LUTs on
   disk; the expensive input path is never forced.

**To verify before relying on it:** that the LUT input domain (BT.2020 cube
extent + PQ shaper envelope) covers everything the intermediate can hold *after
blending* — additive blends and over-peak highlights can exceed the measured
range, where the de-facto behavior is the trilinear edge clamp. Confirm that edge
clamp is the graceful behavior, or extend the cube envelope so it is.

## Configuration

- **Config files are the source of truth.** Brightness (per-output reference-white
  nit level), advertised intents, unmanaged-surface policy, tone/gamut operator
  choices, and per-panel quirks live in KDL per-output config (EDID-keyed, same
  mechanism as today's calibration config).
- **Architecture stays open to online IPC modification.** Don't bake config in as
  compile-time-only. The `prism-tune`/`prism-ipc` live-reload pattern already used
  for calibration is the template; the goal is a future live settings/tune GUI
  that adjusts brightness and mapping policy against the running compositor.
- **Rebuilding shaders / LUTs on config change is acceptable.** The encode chain
  is already synthesized per-output; regenerating shaders or re-baking a LUT on a
  config/calibration change is fine and removes the pressure to make every knob a
  hot per-frame uniform. Per-surface, per-commit state (intent) still needs to be
  dynamic, but the expensive panel-dependent precomputation can be rebuilt on
  change rather than evaluated every frame.

## Deterministic now vs. the policy frontier

| Determined (implement, don't debate) | Policy (design + config) |
|---|---|
| White-point adaptation on/off per intent | The perceptual / saturation compression curves |
| Reference-white anchoring math | The per-output reference-white **nit level** (brightness) |
| Absolute = no-adapt + literal nits + clip-to-mesh | Whether to inverse-tone-map SDR→HDR (default: no) |
| Colorimetric intents (relative, abs, +BPC, abs-no-adapt) | Per-panel treatment of **unmanaged** surfaces |
| Mapping *from* the target/mastering volume, refined by max_cll | How aggressively to trust `max_cll` (panel quirks, see LU28R55) |
| Clipping to the **measured** panel boundary | Which intents to advertise (perceptual mandatory) |

## Baseline behavior today (honest current state)

Independent of declared intent, every managed surface currently gets: white point
Bradford-adapted to D65/BT.2020 (relative-like), `reference_lum` reproduced as
literal nits (absolute-like), gamut hard-clipped via the panel LUT's fixed
hue-preserving projection. The committed render intent is stored on surface state
but **dropped** before the renderer (`description_to_params` has no intent input;
`SurfaceColorParams` has no intent field). Only `perceptual` is advertised.

So the gap is not "no color management" — it's "one fixed hybrid mapping that
matches no named intent and ignores the client's stated preference." Closing it
means (a) threading intent through to the renderer, (b) making the decode matrix +
luminance scale intent-aware, and (c) deciding option 1 vs. 2 for the
out-of-volume operator.

## Open decisions (for discussion)

The big decisions are made above: input realizes the requested light (intent:
white point + anchoring), the LUT realizes-and-degrades (keeping the shared OOG
operator), the intermediate is the universal BT.2020 absolute metric, and config
is file-based with IPC-online-modification kept open. The remaining work is mostly
threading intent to the input stage. What's still open:

1. **Brightness value derivation.** Config + IPC is decided as the *mechanism*;
   how the per-output reference-white nit level is *chosen* is open — static config
   value, or derived (EDID HDR caps + ambient sensor) with config override. The
   anchoring math is identical regardless.
2. **Which intents to advertise next.** Relative (+BPC) are cheap and deterministic
   once intent is threaded. Absolute is the faithful one you want;
   absolute_no_adaptation is nearly free alongside it. Saturation rarely matters
   outside business graphics. ICC-profile intents pull in the deferred ICC creator.
3. **Unmanaged-surface policy per panel.** Keep the flat "sRGB @ output ref white"
   default everywhere, or differentiate (e.g. how a legacy sRGB window sits on a
   wide-gamut HDR panel: clamp to sRGB-in-panel, or let it fill?).
4. **Per-intent LUT variants — now or deferred?** All intents share the LUT's one
   OOG operator today. Build per-policy LUT variants (e.g. perceptual
   soft-compression vs. colorimetric hard-clip) now, or defer until an advertised
   intent demonstrably needs a different curve? Deferring costs nothing structural
   — it's an encode-time LUT selection added later.

Verification owed before relying on the LUT for all OOG handling: confirm the LUT
input domain covers the post-blend intermediate range (see "Architectural
decisions"). This is a check, not a decision.
