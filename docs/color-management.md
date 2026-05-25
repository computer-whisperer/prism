# Color management

The design rationale for prism's defining feature: first-class, HDR-native color
management. This is an inventory of the architectural primitives, the decisions
behind them, and the domain knowledge (amdgpu KMS quirks, panel quirks, tone
mapping, calibration) that the implementation rests on. Much of it is now built —
where it is, [architecture.md](architecture.md) describes the as-built mechanics
and [status.md](status.md) tracks what's verified. This doc is the *why*.

> Origin: a brain dump from the phase-1 niri-fork pathfinding, captured so the
> greenfield wouldn't rediscover what we already learned. The phase-1 fork itself
> is not in this checkout.

## The pipeline that has to fit cleanly

Every reference compositor doing real color management converges on this shape:

```
per-client surface
   └─ decode shader (TF + primary matrix, knows the client's color description)
        └─ writes into intermediate (linear-light, fp16, defined color space)
            └─ blend / composite (in linear)
                └─ encode shader (per-output: calibration + tone-map + OETF)
                    └─ scanout buffer (output's native encoding: sRGB or PQ)
                        └─ KMS modeset properties (Colorspace, HDR_OUTPUT_METADATA)
                            └─ panel
```

Two stages of color transform: **per-element decode at draw time**, and
**per-output encode at scanout time**, with compositing in linear light between
them. This is non-negotiable for correct alpha blending and any per-client
color-space handling. prism implements exactly this; the decode/encode mechanics
are in [architecture.md](architecture.md). The "direct scanout" optimization (skip
the intermediate when a client's description already matches the output's) is a
*specialization* of this path, not a separate path.

## Why this didn't fit smithay

This is the architectural reason prism exists rather than continuing the niri fork:

- **Smithay's `GlesRenderer` and the `RenderElement::draw` API assume one default
  texture shader** (sample + matrix + alpha). There's no per-element decode hook.
  Per-element decode would mean replacing or wrapping every texture-drawing element
  type — and they're smithay's types, not niri's.
- **`DrmCompositor::render_frame` owns the scanout buffer** and binds it as the
  render target. The HDR fork had to side-step it: render into an intermediate via
  a separate path, then submit a single fullscreen `ShaderRenderElement`. The shape
  is wrong — the architecture wants "render into output buffer directly" while we
  want "render into intermediate, postprocess, scan out."
- **`MultiRenderer` / `as_gles_renderer()`** complicate the offscreen path and make
  multi-GPU client buffers an uncertain edge case for HDR outputs.

None of these are smithay defects — they're consequences of smithay being designed
for SDR compositors where one texture shader is enough and direct scanout is the
optimization. Color management is a different architectural axis. prism makes the
color description a first-class property of every render element and makes
"decode-per-element / encode-per-output" the default path, even for SDR outputs.

## Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Intermediate color space | **BT.2020 primaries, absolute-nits linear, fp16** (1.0 = 1 nit) | BT.2020 covers all panels we care about. Absolute nits aligns with PQ semantics and lets us reason in real luminance throughout (gamescope's proven approach). fp16 is the only format with the range+precision for absolute-nits encoding (10000 fits). |
| Per-element decode | **Single shader, uniform-driven** (parameterized by source TF + primary matrix + ref luminance) | Simpler than per-(TF, primary) shader variants; avoids shader-cache complexity. KWin's variant approach is an optimization we can adopt later if profiling demands. |
| Per-output OETF | **Pluggable per output**: sRGB for SDR, PQ for HDR, linear for fp16 scanout | Each output declares its target color description; the encode chain's transfer fragment is selected from it. |
| Pipeline unification | **Every output goes through the intermediate + encode**, even SDR | Eliminates the SDR/HDR fork. One extra pass per frame on SDR outputs (negligible at 1080p, bounded at 4K@60). |
| Calibration mechanism | **3D LUT with a PQ shaper** is the default encode calibration path | A single trilinear sample captures both gamut correction and per-channel response without assuming either is closed-form. Subsumes the older `CalibrationMatrix` + per-channel gain/gamma pair; HDR and SDR calibration share one knob (LUT contents). |
| Frog protocol | **Defer** until basic `wp_color_management_v1` works | Frog is the de-facto HDR-games standard today; worth supporting eventually for OLED + Steam. Spec-aligned protocol first. |

## First-class state

**Per-surface** (lives with the Wayland surface, set via `wp_color_management_v1`):

- Transfer function: {sRGB, BT.1886, PQ (ST 2084), HLG, linear, custom-parametric}
- Primaries: {Rec.709/sRGB, DCI-P3, Display-P3, BT.2020, Adobe RGB, custom xy}
- Reference white luminance (nits): SDR 100 by convention; HDR clients specify
- Optional: max content / frame-average light level (CTA 861.G)
- Optional: mastering-display metadata
- Rendering intent: {perceptual, relative-colorimetric, saturation, absolute}

Default for clients that don't speak the protocol: sRGB transfer, sRGB primaries,
100-nit ref white, perceptual intent.

**Per-output** (set by compositor config + EDID):

- Target color description (primaries + TF + peak luminance + ref white)
- Tone-mapping curve choice (BT.2390 EETF default, configurable)
- Calibration: 3D LUT (default path); 3×3 matrix + 1D LUT remain available as
  encode fragments
- SDR-on-HDR reference white nits (we hardcoded 200 in niri; should be per-output)
- HDR static metadata for the panel (max_cll, max_fall → `HDR_OUTPUT_METADATA`)

**Per-frame derived** (computed each frame per output): the decode-shader uniforms
per element and the encode-shader uniforms (tone-map, OETF mode, ref luminance,
LUT binding).

### What's wired today

- **Decode primaries**: `description_to_params` computes the
  surface-primaries→BT.2020 matrix from the client's actual chromaticities
  (`prism_frame::primaries_to_bt2020`, with named Display-P3/sRGB/BT.2020 support)
  and fills `DecodePush::decode_matrix`. Default is sRGB→BT.2020.
- **Source transfer**: sRGB and linear are anchor-tested; PQ EOTF is written
  (exercised by the P010 video path). HLG, BT.1886, and parametric gamma are stubs.
- **Encode**: `[Lut3d, OutputTransferSrgb|Pq|Linear]` per output. The LUT is
  identity until a calibration is loaded; `prism-tune` generates and pushes one.
- **Output capability advertisement**: `wp_color_management_output_v1.get_output`
  returns the output's preferred PQ/BT.2020 description so HDR clients (Firefox)
  engage their HDR path.
- **KMS HDR signaling + per-output config**: an output with a `color.hdr` config
  block gets `HDR_OUTPUT_METADATA` + `Colorspace = BT2020_RGB` + `max_bpc = 10` +
  fp16 scanout + the PQ encode chain at bringup (re-pushed across VT handoff,
  cleared on shutdown so the next session doesn't inherit stale signaling). CTM,
  per-channel response gain/gamma, panel-peak-nits (per channel), SDR reference
  nits, and a 3D-LUT file are all per-output config — resolved from KDL,
  EDID-keyed (`output "Make Model Serial"`), with live IPC reload from `prism-tune`.

## amdgpu / KMS gotchas to bake in

1. **amdgpu has no CRTC-level `DEGAMMA_LUT`.** All linearization happens in shaders.
   The per-plane gen-2 `COLOR_PIPELINE` exposes degamma on DCN3+ but only when
   amdgpu lights it up on stable kernels (not present in 7.0.x). Shader degamma is
   the always-works path; per-plane HW degamma is a future optimization (Navi 21
   DCN3 eventually; Vega 20 DCN1 never).
2. **CRTC `CTM` is post-blend, post-shader, gamma-space.** Useful as an SDR trim
   *only* if coefficients are derived in gamma space. In HDR mode the CRTC sees
   PQ-encoded scanout, so a gamma-space CTM produces wrong math. prism keeps
   calibration in the shader (linear light, post-decode, pre-OETF); the CRTC CTM is
   unused or a final per-channel trim at most.
3. **`HDR_OUTPUT_METADATA` change triggers a full modeset** (~165 ms on our hw).
   Don't toggle per frame; treat it as a per-output mode setting stable across
   config reloads.
4. **`max bpc` is a connector property** (not per-plane). Set at modeset; the kernel
   silently clamps to the EDID-advertised range.
5. **Legacy property writes for `HDR_OUTPUT_METADATA` + `Colorspace`** work through
   the kernel's atomic shim — no explicit `AtomicModeReq` plumbing needed.
6. **`Colorspace` enum on amdgpu**: `{Default, BT709_YCC, opRGB, BT2020_RGB,
   BT2020_YCC}`. No `DCI-P3` option. HDR signaling uses `BT2020_RGB`; SDR uses
   `Default` (panel decodes as sRGB/Rec.709).
7. **`max_cll` lying-low.** Samsung tone-mappers (LU28R55) crush highlights when
   given honest peak values; advertising a lower `max_cll` (e.g. 250 when peak is
   400) gives a more pleasant picture. Per-panel quirk — expose as config.
8. **fp16 (`Xbgr16161616f`) scanout works on both DCN1 (Vega 20) and DCN3
   (Navi 21).** Tiled fp16 negotiates fine via the scanout modifier path.
9. **EDID HDR static metadata** is parseable: per-panel peak luminance, native
   primaries, supported EOTFs. prism reads EDID (`EdidInfo`) at bringup — it keys
   `output "Make Model Serial"` config blocks and seeds per-output defaults.
   *Auto-enabling* HDR from the EDID's advertised HDR capability (rather than from
   an explicit `color.hdr` block) is the one remaining piece — see
   [deferred-work.md](deferred-work.md).

## Tone mapping

Deferred in phase 1; needed for mixed content. Applied in the encode chain between
calibration and the OETF, as a per-output config (curve + parameters):

- **HDR client → SDR output**: tone-map highlights down (BT.2390 EETF reference;
  Reinhard/Hable simpler). Loses HDR-ness, keeps colors plausible.
- **SDR client → HDR output**: scale SDR ref white to a chosen nit level
  (per-output; niri hardcoded 200). Inverse tone-mapping to brighten highlights is
  controversial — usually best left identity.
- **HDR client → HDR output**: identity if peak luminances match; otherwise
  roll-off (the "panel does 400, content mastered for 1000" case).
- **Cross-gamut**: per-element primary conversion handles the matrix; out-of-gamut
  handling is hard-clip today, soft-clip (3D LUT/shader) is the "right" thing later.

Both decode (over-bright input) and encode (above-peak) currently hard-clip — real
tone mapping waits until we have mixed SDR+HDR content to tune against.

## Calibration data flow

`prism-tune` is the closed-loop calibration tool (replaces the phase-1 spyder
scripts that produced gamma-space CRTC CTMs). It is an IPC client of the running
compositor (`prism-ipc`) plus a colorimeter driver plus a patch-rendering surface:

- `prism-tune characterize` — measure a per-channel response curve (gain/gamma).
- `prism-tune calibrate` — derive a correction.
- `prism-tune calibrate-lut3d` — generate a 3D LUT (`.lut` + `.csv`) for the
  default encode path.
- `prism-tune validate-lut3d` — software validation of the color pipeline against
  the generated LUT.

Where calibration lives in the pipeline: between the encode pass's input (linear
BT.2020 absolute nits) and its OETF stage. The 3D LUT uses a PQ shaper on input so
precision is allocated near zero where the eye is sensitive; the trilinear sample
returns panel-native commanded nits, and a downstream `OutputTransfer*` fragment
encodes for scanout.

**Color-aware calibration is the goal**: the tool emits absolute-luminance /
known-primary patches *through* the compositor's real HDR path and measures with
the colorimeter, so corrections are derived against the actual pipeline rather than
a stand-in.

## Lessons from reference compositors

| Compositor | Intermediate | Decode location | Notable |
|---|---|---|---|
| **KWin** | gamma 2.2 (not linear) | per-surface shader | Chose perf over correctness; 10-bit intermediate; relies on HW OETF offload |
| **Mutter** | BT.2020 linear | per-surface shader | Textbook "right thing"; pays the fp16 cost; integrates DRM plane color props |
| **gamescope** | scRGB linear, absolute nits | pre-decode per fullscreen game | Game-focused; extensive inverse-tone-mapping for SDR-on-HDR |
| **Hyprland** | gamma-ish, shader-driven | per-surface shader | Pre-protocol; monitor-rule config for color mode |

prism is closest to Mutter conceptually (BT.2020 linear intermediate, per-surface
decode), with gamescope's absolute-nits encoding to keep the math in real units.
KWin's gamma-2.2 is a forced perf optimization for older hardware / high refresh —
we may revisit it for the 4K@240 OLED, but the plan is "build correct first,
measure, optimize where needed," not "design around the perf hack."

## What we wanted from day 1 (and built)

- Color description as a first-class property of every render element, not opt-in
  metadata. SDR clients default to "sRGB, 100 nit"; HDR clients carry their
  declared description; the renderer dispatches the decode shader from it. ✅
- A renderer designed around "decode per element, encode per output" rather than
  "draw a texture to a buffer." ✅ (the synthesized encode chain)
- An output's target color description as config/data, not a code path — SDR is the
  degenerate case of the HDR machinery. ✅ (`OutputConfig` + `EncodeConfig`)
- Per-output calibration in the data path from the start, not retrofitted. ✅
  (the 3D LUT path + `prism-tune`)
- EDID parsing for per-output defaults. ✅ (`EdidInfo` + EDID-keyed config). The
  one gap is *auto-enabling* HDR from EDID capability vs. an explicit config block —
  see [deferred-work.md](deferred-work.md).

## Reusable artifacts from phase 1

Carried forward into prism (math/knowledge, not code):

- The PQ encode math (`hdr_pq.frag`) — correct, ported to the synthesized chain.
- `M_SRGB_TO_BT2020` / `M_BT2020_TO_SRGB` and the broader primary-conversion set.
- `build_hdr_metadata_blob` byte layout (kernel `struct hdr_output_metadata`,
  32 bytes incl. tail padding) — for when KMS HDR signaling lands.
- Per-panel/kernel quirk knowledge: LU28R55 max_cll lying-low, `modetest -D pci:`
  silently aliasing card0, modeset latency, format-preference order.
