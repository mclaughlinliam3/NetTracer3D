"""
color_schemes.py
================

Shared palette generation for NetworkGraphWidget and UMAPWidget.

Three schemes, exposed through one entry point, `generate_scheme_hex()`:

  'Default'            - the original HSV hue wheel. Byte-for-byte identical
                         to the previous inline implementations, so existing
                         renders are unchanged.
  'Alt Color Scheme'   - perceptually spaced colors seeded from Kelly's
                         maximum-contrast set. Reaches browns, grays, tans and
                         white/black, which a hue sweep cannot.
  'Colorblind Scheme'  - Okabe-Ito, extended by worst-case sampling under
                         simulated deuteranopia (the most common CVD, ~6% of
                         males). n <= 8 returns Okabe-Ito unmodified.

Both non-default schemes are BACKGROUND-AWARE. This matters: the network
widget renders on white, the UMAP widget on white / dark navy / pale green.
On a light canvas colors are darkened away from the background and the
white palette slot flips to black; on a dark canvas the reverse. Nothing
returned will sit close to the background.

Usage
-----
    from color_schemes import generate_scheme_hex, SCHEMES

    hexes = generate_scheme_hex(n, scheme='Colorblind Scheme', background='w')
"""

from typing import List, Tuple
import colorsys
import math

RGB = Tuple[int, int, int]

# --------------------------------------------------------------- schemes ----
SCHEME_DEFAULT = 'Default'
SCHEME_ALT = 'Alt Color Scheme'
SCHEME_COLORBLIND = 'Colorblind Scheme'
SCHEME_CUSTOM = 'Custom'
SCHEME_PREVIOUS = 'Previous'

# Schemes that generate a palette algorithmically.
AUTO_SCHEMES = [SCHEME_DEFAULT, SCHEME_ALT, SCHEME_COLORBLIND]
# Everything offered in a UI picker. 'Previous' is last so index 0 remains
# 'Default' for any caller that still selects by index.
SCHEMES = AUTO_SCHEMES + [SCHEME_CUSTOM, SCHEME_PREVIOUS]

# Used whenever 'Previous' is requested but nothing has been recorded yet.
FALLBACK_SCHEME = SCHEME_DEFAULT

# Palette domains tracked independently. Communities, identities and
# individual nodes are different label spaces, so reusing one's colors for
# another would be meaningless -- each gets its own slot.
DOMAIN_COMMUNITIES = 'communities'
DOMAIN_IDENTITIES = 'identities'
DOMAIN_NODES = 'nodes'
DOMAINS = [DOMAIN_COMMUNITIES, DOMAIN_IDENTITIES, DOMAIN_NODES]

# Kelly's colors of maximum contrast (black/white handled by the bg logic).
_ALT_PALETTE_HEX = [
    "F3C300", "875692", "F38400", "A1CAF1", "BE0032", "C2B280", "848482",
    "008856", "E68FAC", "0067A5", "F99379", "604E97", "F6A600", "B3446C",
    "DCD300", "882D17", "8DB600", "654522", "E25822", "2B3D26",
]

# Okabe-Ito (CVD-safe) + Paul Tol "muted" extras for n > 8.
_CB_PALETTE_HEX = [
    "E69F00", "56B4E9", "009E73", "F0E442", "0072B2", "D55E00", "CC79A7",
    "88CCEE", "999933", "882255", "44AA99", "DDCC77", "AA4499",
]

# Deuteranopia simulation matrix (Vienot et al. 1999), on LINEAR RGB.
_DEUTAN_MATRIX = (
    (0.33066007, 0.66933993, 0.0),
    (0.33066007, 0.66933993, 0.0),
    (-0.02785538, 0.02785538, 1.0),
)

# Lightness weights. Under normal vision hue/chroma carry the categorical
# signal so lightness is discounted; a deuteranope has lost the red-green
# axis, leaving lightness as the main surviving channel, so it weighs more.
_L_WEIGHT_NORMAL = 0.75
_L_WEIGHT_DEUTAN = 1.6

# Separation below which two colors read as "the same" to a deuteranope.
# Calibrated to the tightest pair in Okabe-Ito, a published CVD-safe palette.
CB_CONFUSABLE = 0.083

# Minimum separation any color must keep from the background.
_BG_SEPARATION = 0.20


# ------------------------------------------------------------ conversions ---
def _hex_to_rgb(h: str) -> RGB:
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb: RGB) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*(int(max(0, min(255, c))) for c in rgb))


def _srgb_to_linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> int:
    c = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return max(0, min(255, round(c * 255)))


def _rgb_to_oklab(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v) for v in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def simulate_deuteranopia(rgb: RGB) -> RGB:
    """How an RGB color appears to a deuteranope."""
    lin = [_srgb_to_linear(c) for c in rgb]
    return tuple(
        _linear_to_srgb(max(0.0, min(1.0, sum(_DEUTAN_MATRIX[i][j] * lin[j]
                                              for j in range(3)))))
        for i in range(3)
    )


# -------------------------------------------------------------- distances ---
def _dist(a, b, l_weight: float) -> float:
    dl = (a[0] - b[0]) * l_weight
    return math.sqrt(dl * dl + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _labs_normal(rgb: RGB):
    return (_rgb_to_oklab(rgb),)


def _labs_cvd(rgb: RGB):
    return (_rgb_to_oklab(rgb), _rgb_to_oklab(simulate_deuteranopia(rgb)))


def _dist_normal(a, b) -> float:
    return _dist(a[0], b[0], _L_WEIGHT_NORMAL)


def _dist_cvd(a, b) -> float:
    """Worst case across normal and deuteranopic vision."""
    return min(_dist(a[0], b[0], _L_WEIGHT_NORMAL),
               _dist(a[1], b[1], _L_WEIGHT_DEUTAN))


# --------------------------------------------------------- background prep ---
def normalize_background(background) -> RGB:
    """
    Accept anything the widgets use for a background and return an RGB tuple.

    Handles pyqtgraph single letters ('w', 'k'), hex strings ('#1a1a2e'),
    friendly names ('white', 'black', 'green'), and RGB tuples.
    """
    if background is None:
        return (255, 255, 255)
    if isinstance(background, (tuple, list)) and len(background) >= 3:
        return tuple(int(c) for c in background[:3])
    if isinstance(background, str):
        s = background.strip().lower()
        named = {
            'w': (255, 255, 255), 'white': (255, 255, 255),
            'k': (0, 0, 0), 'black': (0, 0, 0),
            'g': (128, 128, 128), 'grey': (128, 128, 128), 'gray': (128, 128, 128),
            'green': (200, 213, 163),
        }
        if s in named:
            return named[s]
        if s.startswith('#'):
            try:
                return _hex_to_rgb(s)
            except Exception:
                return (255, 255, 255)
    return (255, 255, 255)


def _blend_toward(rgb: RGB, target: RGB, t: float) -> RGB:
    """Blend in linear light, which keeps the result from looking muddy."""
    return tuple(
        _linear_to_srgb(_srgb_to_linear(c) * (1 - t) + _srgb_to_linear(tc) * t)
        for c, tc in zip(rgb, target)
    )


def _push_lightness(rgb: RGB, bound: float, dark_bg: bool) -> RGB:
    """
    On a dark background, lift a color until OKLab L >= bound.
    On a light background, darken it until OKLab L <= bound.
    """
    l = _rgb_to_oklab(rgb)[0]
    if (dark_bg and l >= bound) or (not dark_bg and l <= bound):
        return rgb
    target = (255, 255, 255) if dark_bg else (0, 0, 0)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        lm = _rgb_to_oklab(_blend_toward(rgb, target, mid))[0]
        ok = lm >= bound if dark_bg else lm <= bound
        if ok:
            hi = mid
        else:
            lo = mid
    return _blend_toward(rgb, target, hi)


def _candidate_grid(step_levels: int = 8) -> List[RGB]:
    levels = [round(i * 255 / (step_levels - 1)) for i in range(step_levels)]
    return [(r, g, b) for r in levels for g in levels for b in levels]


# ------------------------------------------------------------- core engine ---
def _build_palette(n_colors: int, base_hex: List[str], background,
                   dist_fn, labs_fn, dedup: float) -> List[str]:
    """
    Curated palette first, then greedy farthest-point sampling to extend.

    `dist_fn` / `labs_fn` decide whether separation is judged under normal
    vision only or worst-case across normal + deuteranopic vision.
    """
    if n_colors <= 0:
        return []

    bg_rgb = normalize_background(background)
    bg = labs_fn(bg_rgb)
    dark_bg = _rgb_to_oklab(bg_rgb)[0] < 0.5

    # Lightness bound: stay clear of the background side of the range.
    bound = 0.45 if dark_bg else 0.72

    # The high-contrast neutral slot: white on dark, black on light.
    contrast_slot = (255, 255, 255) if dark_bg else (0, 0, 0)

    chosen: List[RGB] = []
    chosen_v = []

    def accept(rgb: RGB) -> bool:
        v = labs_fn(rgb)
        if dist_fn(v, bg) < _BG_SEPARATION:
            return False
        if any(dist_fn(v, c) < dedup for c in chosen_v):
            return False
        chosen.append(rgb)
        chosen_v.append(v)
        return True

    # 1. Neutral contrast color first, then the curated palette.
    accept(contrast_slot)
    for h in base_hex:
        if len(chosen) >= n_colors:
            return [_rgb_to_hex(c) for c in chosen[:n_colors]]
        accept(_push_lightness(_hex_to_rgb(h), bound, dark_bg))

    # 2. Extend by greedy farthest-point sampling.
    if len(chosen) < n_colors:
        candidates, cand_v = [], []
        for c in _candidate_grid():
            l = _rgb_to_oklab(c)[0]
            if (dark_bg and l < bound) or (not dark_bg and l > bound):
                continue
            v = labs_fn(c)
            if dist_fn(v, bg) < _BG_SEPARATION:
                continue
            candidates.append(c)
            cand_v.append(v)

        while len(chosen) < n_colors and candidates:
            best_i, best_d = -1, -1.0
            for i, v in enumerate(cand_v):
                d = (min(dist_fn(v, c) for c in chosen_v)
                     if chosen_v else dist_fn(v, bg))
                if d > best_d:
                    best_i, best_d = i, d
            chosen.append(candidates.pop(best_i))
            chosen_v.append(cand_v.pop(best_i))

    # 3. Last resort if the grid is exhausted (very large n): cycle.
    while len(chosen) < n_colors:
        chosen.append(chosen[len(chosen) % max(1, len(chosen))])

    return [_rgb_to_hex(c) for c in chosen[:n_colors]]


# ------------------------------------------------------- public entry point ---
def generate_default_hex(n_colors: int) -> List[str]:
    """The original HSV hue wheel. Preserved exactly for backwards fidelity."""
    colors = []
    for i in range(n_colors):
        hue = i / max(n_colors, 1)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append('#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


def generate_alt_hex(n_colors: int, background='w') -> List[str]:
    """Perceptually spaced colors, background-aware."""
    return _build_palette(n_colors, _ALT_PALETTE_HEX, background,
                          _dist_normal, _labs_normal, dedup=0.10)


def generate_colorblind_hex(n_colors: int, background='w') -> List[str]:
    """Deuteranopia-safe colors, background-aware. n <= 8 is Okabe-Ito."""
    return _build_palette(n_colors, _CB_PALETTE_HEX, background,
                          _dist_cvd, _labs_cvd, dedup=CB_CONFUSABLE)


def generate_scheme_hex(n_colors: int, scheme=SCHEME_DEFAULT,
                        background='w') -> List[str]:
    """
    Return `n_colors` hex strings for the requested scheme.

    Unknown scheme names fall back to 'Default', so a stale saved state or a
    bad combo index can never break a render.
    """
    if n_colors <= 0:
        return []
    if scheme == SCHEME_ALT:
        return generate_alt_hex(n_colors, background)
    if scheme == SCHEME_COLORBLIND:
        return generate_colorblind_hex(n_colors, background)
    # SCHEME_CUSTOM has no algorithmic palette of its own -- it is seeded from
    # Default and then hand-edited, so a bare request falls through to here.
    return generate_default_hex(n_colors)


def audit_colorblind(hex_colors, threshold: float = CB_CONFUSABLE):
    """
    Check a palette for pairs a deuteranope would confuse.
    Returns [(distance, hex_a, hex_b), ...] closest first. Empty is good.
    """
    rgbs = [_hex_to_rgb(h) if isinstance(h, str) else tuple(h) for h in hex_colors]
    vs = [_labs_cvd(c) for c in rgbs]
    out = []
    for i in range(len(rgbs)):
        for j in range(i + 1, len(rgbs)):
            d = _dist_cvd(vs[i], vs[j])
            if d < threshold:
                out.append((round(d, 3), _rgb_to_hex(rgbs[i]), _rgb_to_hex(rgbs[j])))
    return sorted(out)


# =============================================================================
# Standard label -> color contract
# =============================================================================
#
# Every consumer (community_extractor, NetworkGraphWidget, UMAPWidget) funnels
# through `resolve_palette`. It owns the three decisions that used to be
# duplicated and subtly divergent across call sites:
#
#   1. how labels are ordered before colors are handed out,
#   2. how the palette is shuffled,
#   3. how the outlier label (community 0) is special-cased to brown.
#
# The differences between call sites are expressed as arguments rather than
# smoothed over, so existing renders stay pixel-identical.

import random as _random

# Community 0 is treated as an outlier group and always painted brown.
OUTLIER_HEX = '#8B4513'


def normalize_label(value, default='Unknown'):
    """
    Reduce a label to a single hashable scalar.

    Identity dicts store a LIST per node -- often a one-element list produced
    by remove_dupe_ids, sometimes several identities. A node can only be drawn
    in one color, so the first entry wins (deterministic; sets are sorted by
    string form first so the choice is stable across runs).

    Use this wherever a node's stored label is turned into a color-map key, so
    the key matches what resolve_palette generated.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) \
            else list(value)
        if not items:
            return default
        return normalize_label(items[0], default)
    return value


def _unique_sorted(labels):
    """
    Reduce labels to a unique, sorted, hashable list.

    Container values (identity lists) are flattened to their members, so a
    palette is built over the individual identities rather than over the
    unhashable lists themselves -- which is both what users want to see in a
    legend and the only thing set() will accept.

    Sorting falls back to string form while KEEPING the original label objects,
    so callers can still look up with the value they hold.
    """
    uniq = flatten_labels(labels)
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=str)


# =============================================================================
# Last-used palette registry
# =============================================================================
#
# Every call to resolve_palette records what it produced, keyed by domain, so a
# later render can ask for 'Previous' and get visually consistent colors across
# the network graph, the UMAP view and the extractor overlays.
#
# Domains are tracked separately because communities, identities and individual
# nodes are different label spaces -- reusing one's colors for another would be
# meaningless.
#
# Reads and writes are mutex-guarded: NetworkGraphWidget resolves palettes on a
# background QThread while the GUI thread may be reading the same slot.

import sys as _sys
import threading as _threading

# The registry is anchored on the `sys` module rather than on this module's
# globals, because this file legitimately gets imported under two different
# names: the widgets use `import color_schemes`, while community_extractor is
# inside a package and uses `from . import color_schemes`. Python treats those
# as separate module objects with separate globals -- which would give each
# importer its own private registry and quietly defeat the whole point of
# 'Previous'. A process-global store keeps every importer on the same state.
_REGISTRY_KEY = '_anthropic_color_schemes_registry_v1'


def _registry():
    """The one process-wide registry, created on first use."""
    store = getattr(_sys, _REGISTRY_KEY, None)
    if store is None:
        store = {'lock': _threading.RLock(),
                 'palettes': {d: None for d in DOMAINS}}
        setattr(_sys, _REGISTRY_KEY, store)
    # A module reloaded after DOMAINS changed should not lose existing slots.
    for d in DOMAINS:
        store['palettes'].setdefault(d, None)
    return store


def _normalize_domain(domain):
    """Map a domain-ish value to a canonical domain name, or None."""
    if not domain:
        return None
    key = str(domain).strip().lower()
    aliases = {
        'community': DOMAIN_COMMUNITIES, 'communities': DOMAIN_COMMUNITIES,
        'identity': DOMAIN_IDENTITIES, 'identities': DOMAIN_IDENTITIES,
        'node': DOMAIN_NODES, 'nodes': DOMAIN_NODES,
    }
    return aliases.get(key)


def record_palette(domain, scheme, color_map, background='w'):
    """
    Remember a resolved palette as the most recent one for `domain`.

    Called automatically by resolve_palette; exposed for callers that build a
    map by other means and still want it to become the 'Previous' palette.
    """
    domain = _normalize_domain(domain)
    if not domain or not color_map:
        return
    store = _registry()
    with store['lock']:
        store['palettes'][domain] = {
            'scheme': scheme,
            'background': background,
            'color_map': dict(color_map),
        }


def get_last_palette(domain):
    """Return a copy of the stored entry for `domain`, or None."""
    domain = _normalize_domain(domain)
    if not domain:
        return None
    store = _registry()
    with store['lock']:
        entry = store['palettes'].get(domain)
        if not entry:
            return None
        return {
            'scheme': entry['scheme'],
            'background': entry['background'],
            'color_map': dict(entry['color_map']),
        }


def get_last_color_map(domain):
    """Just the {label: hex} map for `domain`, or None."""
    entry = get_last_palette(domain)
    return entry['color_map'] if entry else None


def has_previous(domain):
    """
    True when a palette has been recorded for `domain`.

    UI code uses this to decide whether 'Previous' should be preselected.
    """
    return get_last_palette(domain) is not None


def clear_last_palette(domain=None):
    """Forget the stored palette for a domain, or for every domain."""
    store = _registry()
    with store['lock']:
        if domain is None:
            for d in DOMAINS:
                store['palettes'][d] = None
            return
        d = _normalize_domain(domain)
        if d:
            store['palettes'][d] = None


class _LastPaletteAccessor:
    """
    Attribute-style access to the registry:

        color_schemes.last_palette.communities   -> entry dict or None
        color_schemes.last_palette.identities
        color_schemes.last_palette.nodes
    """

    @property
    def communities(self):
        return get_last_palette(DOMAIN_COMMUNITIES)

    @property
    def identities(self):
        return get_last_palette(DOMAIN_IDENTITIES)

    @property
    def nodes(self):
        return get_last_palette(DOMAIN_NODES)

    def __getitem__(self, domain):
        return get_last_palette(domain)

    def __repr__(self):
        parts = []
        for d in DOMAINS:
            e = get_last_palette(d)
            parts.append(f"{d}={e['scheme'] if e else None}")
        return f"<last_palette {' '.join(parts)}>"


last_palette = _LastPaletteAccessor()


def flatten_labels(source):
    """
    Collect category labels from a dict, iterable, or dict-of-collections.

    Identity dicts sometimes map a node to a LIST of identities rather than a
    single one, so a naive set(d.values()) would raise on unhashable values.
    This flattens one level and skips anything unusable.
    """
    if source is None:
        return []
    values = source.values() if hasattr(source, 'values') else source
    out = []
    for v in values:
        if isinstance(v, (list, tuple, set, frozenset)):
            out.extend(v)
        else:
            out.append(v)
    seen, uniq = set(), []
    for v in out:
        try:
            if v in seen:
                continue
            seen.add(v)
        except TypeError:
            continue
        uniq.append(v)
    return uniq


# Two colors closer than this in OKLab read as the same swatch. Used only to
# keep generated colors away from pinned (custom / previous) ones.
_PIN_COLLISION = 0.06


def _colors_avoiding(n_needed, pinned_hexes, scheme, background):
    """
    Generate `n_needed` colors that avoid everything in `pinned_hexes`.

    When a previous or custom map pins some labels, the remaining labels must
    not be handed a color that duplicates a pinned one -- otherwise two
    categories render identically. Grows the requested palette until enough
    non-colliding colors exist, then gives up gracefully rather than looping.
    """
    if n_needed <= 0:
        return []
    pinned_labs = [_rgb_to_oklab(_hex_to_rgb(h)) for h in set(pinned_hexes)]
    if not pinned_labs:
        return generate_scheme_hex(n_needed, scheme, background)

    def usable(pool):
        out = []
        for h in pool:
            lab = _rgb_to_oklab(_hex_to_rgb(h))
            if all(_dist(lab, p, _L_WEIGHT_NORMAL) >= _PIN_COLLISION
                   for p in pinned_labs):
                out.append(h)
        return out

    n_try = n_needed + len(pinned_labs)
    for _ in range(6):
        pool = usable(generate_scheme_hex(n_try, scheme, background))
        if len(pool) >= n_needed:
            return pool[:n_needed]
        n_try *= 2
    # Exhausted: return what we have, padded from the raw palette so the
    # caller always gets the count it asked for.
    pool = usable(generate_scheme_hex(n_try, scheme, background))
    filler = generate_scheme_hex(n_needed, scheme, background)
    while len(pool) < n_needed:
        pool.append(filler[len(pool) % len(filler)])
    return pool[:n_needed]


def resolve_palette(labels,
                    scheme=SCHEME_DEFAULT,
                    background='w',
                    custom_map=None,
                    shuffle='seeded',
                    shuffle_seed=42,
                    sort_reverse=False,
                    outlier_label=0,
                    outlier_consumes_slot=True,
                    domain=None,
                    record=True):
    """
    Map category labels to hex colors. This is THE entry point.

    Parameters
    ----------
    labels : iterable
        Category labels (community ids, identity strings, node ids...).
        Duplicates are fine; they are reduced to a unique sorted set.
    scheme : str
        'Default' | 'Alt Color Scheme' | 'Colorblind Scheme' | 'Custom'.
        Unknown values fall back to 'Default' rather than raising.
    background : str or tuple
        Canvas background, so the palette can avoid blending into it.
    custom_map : dict, optional
        {label: '#rrggbb'} used when scheme == 'Custom'. Labels missing from
        the map fall back to the Default palette, so a partial map is safe.
    shuffle : {'seeded', 'colors', 'none'}
        'seeded' - shuffle the LABELS with Random(shuffle_seed). Reproducible.
                   Matches the widgets and assign_community_colors.
        'colors' - shuffle the PALETTE with the global RNG (not reproducible).
                   Matches assign_node_colors.
        'none'   - assign in sorted order.
    sort_reverse : bool
        Sort labels descending before assignment (assign_node_colors did this).
    outlier_label : hashable or None
        Label forced to brown. None disables the special case entirely.
    outlier_consumes_slot : bool
        True  - generate a color for the outlier too, then overwrite it.
                (What the widgets did.)
        False - exclude the outlier before generating, so it does not use up
                a palette slot. (What assign_community_colors did.)
    domain : {'communities', 'identities', 'nodes'} or None
        Which registry slot this palette belongs to. Required for
        scheme='Previous' to find anything, and for the result to be
        remembered as the next 'Previous'.
    record : bool
        Set False to resolve without disturbing the registry -- used for
        preview surfaces (e.g. the custom color editor's batch buttons) that
        should not redefine what 'Previous' means.

    Returns
    -------
    dict : {label: '#rrggbb'}
    """
    domain = _normalize_domain(domain)

    # --- 'Previous': reuse the last palette recorded for this domain ---
    #
    # The stored map is applied as an override on top of its own original
    # scheme, so labels seen before keep their exact colors while labels that
    # have appeared since still get sensible ones from the same scheme.
    # With nothing recorded (or no domain), fall through to FALLBACK_SCHEME so
    # a stray 'Previous' can never produce an empty or broken render.
    # A custom map is honoured when the caller explicitly asked for 'Custom',
    # or when it was seeded from a recorded palette below. It is NOT applied
    # to an Alt/Colorblind request just because a stale map is still attached
    # to the widget.
    apply_custom = (scheme == SCHEME_CUSTOM)

    if scheme == SCHEME_PREVIOUS:
        entry = get_last_palette(domain) if domain else None
        if entry:
            previous_map = entry['color_map']
            scheme = entry['scheme']
            if scheme == SCHEME_PREVIOUS:      # defensive: never chain
                scheme = FALLBACK_SCHEME
            merged = dict(previous_map)
            if custom_map:
                merged.update(custom_map)      # an explicit map still wins
            custom_map = merged
            apply_custom = True
        else:
            scheme = FALLBACK_SCHEME
            apply_custom = (scheme == SCHEME_CUSTOM)

    uniq = _unique_sorted(labels)
    if sort_reverse:
        uniq = list(reversed(uniq))
    if not uniq:
        return {}

    has_outlier = outlier_label is not None and outlier_label in uniq

    # Which labels actually need a generated color.
    if has_outlier and not outlier_consumes_slot:
        palette_labels = [l for l in uniq if l != outlier_label]
    else:
        palette_labels = list(uniq)

    n = len(palette_labels)

    # --- ordering ---
    if shuffle == 'seeded' and n:
        order = _random.Random(shuffle_seed).sample(palette_labels, n)
    else:
        order = palette_labels

    # Labels already pinned by a custom / previous map do not consume a
    # generated color, and generated colors must steer clear of theirs.
    pinned = {}
    if apply_custom and custom_map:
        pinned = {l: _normalize_hex(h) for l, h in custom_map.items()
                  if l in palette_labels}
    free = [l for l in order if l not in pinned]

    colors = _colors_avoiding(len(free), pinned.values(), scheme, background) \
        if pinned else (generate_scheme_hex(n, scheme, background) if n else [])

    if shuffle == 'colors' and colors:
        colors = list(colors)
        _random.shuffle(colors)

    if pinned:
        color_map = dict(pinned)
        for i, label in enumerate(free):
            if i < len(colors):
                color_map[label] = colors[i]
    else:
        color_map = {label: colors[i] for i, label in enumerate(order)}

    # --- custom overrides ---
    # Applied last so a hand-picked color always wins, and so a partial custom
    # map degrades gracefully to generated colors for anything it omits.
    if apply_custom and custom_map:
        for label, hx in custom_map.items():
            if label in color_map or label in uniq:
                color_map[label] = _normalize_hex(hx)

    # --- outlier ---
    if has_outlier:
        color_map[outlier_label] = OUTLIER_HEX

    # --- remember, so the next render can ask for 'Previous' ---
    if record and domain:
        record_palette(domain, scheme, color_map, background)

    return color_map


def resolve_palette_interactive(labels, scheme=SCHEME_DEFAULT, background='w',
                                custom_map=None, domain=None, parent=None,
                                category_name=None, on_message=None, **kwargs):
    """
    resolve_palette, but 'Custom' actually TRIGGERS the color editor.

    resolve_palette itself is non-interactive: it applies a custom map if given
    and otherwise ignores the request. This wrapper is the entry point that
    turns scheme='Custom' into a prompt, so a caller can ask for custom colors
    without having to build the map first.

    Falls back to FALLBACK_SCHEME when the editor cannot be shown (headless,
    no PyQt6) or the user cancels, so a custom request can never leave a
    caller with no colors at all.
    """
    scheme = scheme_from_legacy(scheme)
    say = on_message or (lambda msg: print(msg))

    if scheme == SCHEME_CUSTOM and not custom_map:
        labels_for_prompt = list(_unique_sorted(labels))
        if not can_prompt():
            say("Custom colors need a running Qt GUI; "
                f"falling back to '{FALLBACK_SCHEME}'.")
            scheme = FALLBACK_SCHEME
        elif not labels_for_prompt:
            scheme = FALLBACK_SCHEME
        else:
            seed_kwargs = {k: v for k, v in kwargs.items()
                           if k in ('shuffle', 'shuffle_seed', 'sort_reverse',
                                    'outlier_label', 'outlier_consumes_slot')}
            seed_kwargs['domain'] = domain
            picked = prompt_custom_colors(
                labels_for_prompt,
                parent=parent,
                background=background,
                initial_scheme=(SCHEME_PREVIOUS if has_previous(domain)
                                else FALLBACK_SCHEME),
                category_name=category_name or 'Category',
                seeded_kwargs=seed_kwargs,
            )
            if picked:
                custom_map = picked
            else:
                say(f"Custom color selection cancelled; "
                    f"falling back to '{FALLBACK_SCHEME}'.")
                scheme = FALLBACK_SCHEME

    return resolve_palette(labels, scheme=scheme, background=background,
                           custom_map=custom_map, domain=domain, **kwargs)


def resolve_palette_rgb(labels, **kwargs):
    """Same as resolve_palette but values are (r, g, b) tuples, 0-255."""
    return {k: _hex_to_rgb(v) for k, v in resolve_palette(labels, **kwargs).items()}


def _normalize_hex(value):
    """Accept '#rrggbb', 'rrggbb', (r,g,b) or (r,g,b,a) and return '#rrggbb'."""
    if isinstance(value, str):
        v = value.strip()
        if not v.startswith('#'):
            v = '#' + v
        try:
            return _rgb_to_hex(_hex_to_rgb(v))
        except Exception:
            return '#808080'
    try:
        return _rgb_to_hex(tuple(int(c) for c in value[:3]))
    except Exception:
        return '#808080'


def scheme_from_legacy(value):
    """
    Normalise any historical scheme spelling to a canonical scheme name.

    Accepts the old integer codes (0/1/2), booleans, scheme names, and common
    variants. Anything unrecognised becomes 'Default' rather than raising, so
    a stale saved setting can never break a render.
    """
    if value is None:
        return SCHEME_DEFAULT
    if isinstance(value, bool):
        return SCHEME_ALT if value else SCHEME_DEFAULT
    if isinstance(value, int):
        return {0: SCHEME_DEFAULT,
                1: SCHEME_ALT,
                2: SCHEME_COLORBLIND,
                3: SCHEME_CUSTOM,
                4: SCHEME_PREVIOUS}.get(value, SCHEME_DEFAULT)
    key = str(value).strip().lower()
    for canonical in SCHEMES:
        if key == canonical.lower():
            return canonical
    return {
        'alt': SCHEME_ALT, 'alternative': SCHEME_ALT, 'alt color': SCHEME_ALT,
        'colorblind': SCHEME_COLORBLIND, 'color blind': SCHEME_COLORBLIND,
        'cvd': SCHEME_COLORBLIND,
        'custom': SCHEME_CUSTOM,
        'previous': SCHEME_PREVIOUS, 'last': SCHEME_PREVIOUS,
        'default': SCHEME_DEFAULT, 'standard': SCHEME_DEFAULT,
    }.get(key, SCHEME_DEFAULT)


def hex_to_rgb(hex_color):
    """'#rrggbb' -> (r, g, b)."""
    return _hex_to_rgb(_normalize_hex(hex_color))


def hex_to_rgba(hex_color, alpha=255):
    """'#rrggbb' -> (r, g, b, a). Convenience for LUT/array consumers."""
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, alpha)


def rgb_to_hex(rgb):
    """(r, g, b) -> '#rrggbb'."""
    return _normalize_hex(rgb)


def generate_scheme_rgb(n_colors, scheme=SCHEME_DEFAULT, background='w'):
    """`generate_scheme_hex` as (r, g, b) tuples, for array/LUT consumers."""
    return [_hex_to_rgb(h) for h in generate_scheme_hex(n_colors, scheme, background)]


# =============================================================================
# Custom color picker dialog
# =============================================================================
#
# Qt is imported lazily inside the functions below so that this module stays
# importable in headless contexts (community_extractor is a package module and
# may run without a QApplication).

def _require_qt():
    """Import PyQt6 on demand. Raises ImportError with a useful message."""
    try:
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QScrollArea, QWidget, QColorDialog, QLineEdit, QFrame,
            QDialogButtonBox, QGridLayout,
        )
        from PyQt6.QtGui import QColor
        from PyQt6.QtCore import Qt
        return dict(
            QDialog=QDialog, QVBoxLayout=QVBoxLayout, QHBoxLayout=QHBoxLayout,
            QLabel=QLabel, QPushButton=QPushButton, QScrollArea=QScrollArea,
            QWidget=QWidget, QColorDialog=QColorDialog, QLineEdit=QLineEdit,
            QFrame=QFrame, QDialogButtonBox=QDialogButtonBox,
            QGridLayout=QGridLayout, QColor=QColor, Qt=Qt,
        )
    except ImportError as e:
        raise ImportError(
            "The custom color picker needs PyQt6. Install PyQt6, or use one "
            "of the automatic schemes instead."
        ) from e


def _label_text(label):
    """
    Human-readable text for a swatch row.

    Labels arrive already flattened to scalars, but guard the container case
    anyway so a raw identity list never renders as "['thing']".
    """
    if isinstance(label, (list, tuple, set, frozenset)):
        parts = sorted((str(x) for x in label), key=str)
        return ', '.join(parts) if parts else 'Unassigned'
    text = str(label)
    if text.strip() == '':
        return 'Unassigned'
    if text == '0':
        return '0  (outliers)'
    return text


def _canvas_css(background):
    """CSS background color for the preview strip, from a canvas value."""
    return _rgb_to_hex(normalize_background(background))


def make_custom_color_dialog(labels, parent=None, background='w',
                             initial_map=None, category_name='Category',
                             seeded_kwargs=None):
    """
    Build (but do not show) the custom color dialog. Returns the dialog.

    Most callers want `prompt_custom_colors` instead.
    """
    Q = _require_qt()

    class _Swatch(Q['QPushButton']):
        """A clickable color chip that opens the system color picker."""

        def __init__(self, hex_color, on_change, outer_parent=None):
            super().__init__(outer_parent)
            self._hex = _normalize_hex(hex_color)
            self._on_change = on_change
            self.setFixedSize(46, 22)
            self.setCursor(Q['Qt'].CursorShape.PointingHandCursor)
            self.clicked.connect(self._pick)
            self._refresh()

        def _refresh(self):
            # Thin neutral border so a swatch matching the canvas is still
            # visible as a chip rather than vanishing into the row.
            self.setStyleSheet(
                f"background-color: {self._hex};"
                "border: 1px solid #888; border-radius: 3px;"
            )
            self.setToolTip(f"{self._hex}  (click to change)")

        def hex(self):
            return self._hex

        def set_hex(self, hex_color):
            self._hex = _normalize_hex(hex_color)
            self._refresh()

        def _pick(self):
            col = Q['QColorDialog'].getColor(
                Q['QColor'](self._hex), self,
                "Pick a color",
                Q['QColorDialog'].ColorDialogOption.DontUseNativeDialog,
            )
            if col.isValid():
                self.set_hex(f"#{col.red():02x}{col.green():02x}{col.blue():02x}")
                if self._on_change:
                    self._on_change()

    class _CustomColorDialog(Q['QDialog']):
        def __init__(self):
            super().__init__(parent)
            self.setWindowTitle(f"Custom {category_name} Colors")
            self.setModal(True)
            self._labels = list(labels)
            self._background = background
            self._seeded_kwargs = dict(seeded_kwargs or {})
            self._rows = {}      # label -> (row_widget, swatch, text)
            self._result = None
            self._build()

        # ---------------------------------------------------------- build --
        def _build(self):
            root = Q['QVBoxLayout'](self)

            blurb = Q['QLabel'](
                f"Click a color chip to change it. {len(self._labels)} "
                f"{category_name.lower()}"
                f"{'' if len(self._labels) == 1 else 's'}."
            )
            blurb.setWordWrap(True)
            root.addWidget(blurb)

            # -- batch assignment --
            batch = Q['QHBoxLayout']()
            batch.addWidget(Q['QLabel']("Fill all from:"))
            for scheme in AUTO_SCHEMES:
                btn = Q['QPushButton'](scheme.replace(' Color Scheme', '')
                                              .replace(' Scheme', ''))
                btn.setToolTip(f"Reassign every color using: {scheme}")
                btn.clicked.connect(
                    lambda _checked=False, s=scheme: self._apply_scheme(s))
                batch.addWidget(btn)
            batch.addStretch()
            root.addLayout(batch)

            # -- filter (only earns its space with a long list) --
            if len(self._labels) > 12:
                self._filter = Q['QLineEdit']()
                self._filter.setPlaceholderText("Filter…")
                self._filter.textChanged.connect(self._apply_filter)
                root.addWidget(self._filter)
            else:
                self._filter = None

            # -- scrollable rows --
            self._scroll = Q['QScrollArea']()
            self._scroll.setWidgetResizable(True)
            inner = Q['QWidget']()
            inner.setObjectName("cs_preview_inner")
            self._inner = inner
            self._inner_layout = Q['QVBoxLayout'](inner)
            self._inner_layout.setContentsMargins(4, 4, 4, 4)
            self._inner_layout.setSpacing(2)

            initial = dict(initial_map or {})
            for label in self._labels:
                self._add_row(label, initial.get(label, '#808080'))
            self._inner_layout.addStretch()
            self._scroll.setWidget(inner)

            # Preview against the real canvas color, so a color that would
            # disappear on this background is obvious here rather than later.
            #
            # Styling the QScrollArea alone is not enough: the frame is painted
            # but the scrolled widget keeps the default palette, so label text
            # picked for a dark canvas ended up near-white on a light grey
            # surface and was invisible. Paint the inner widget and the
            # viewport explicitly.
            css = _canvas_css(self._background)
            self._scroll.setStyleSheet(
                f"QScrollArea {{ background-color: {css}; border: 1px solid #888; }}"
            )
            self._scroll.viewport().setStyleSheet(f"background-color: {css};")
            inner.setAutoFillBackground(True)
            inner.setStyleSheet(f"#cs_preview_inner {{ background-color: {css}; }}")

            # Height: show up to ~12 rows, scroll beyond that.
            visible_rows = min(max(len(self._labels), 1), 12)
            self._scroll.setMinimumHeight(min(28 * visible_rows + 16, 360))
            self._scroll.setMinimumWidth(300)
            root.addWidget(self._scroll, stretch=1)

            # -- accept / cancel --
            buttons = Q['QDialogButtonBox'](
                Q['QDialogButtonBox'].StandardButton.Ok
                | Q['QDialogButtonBox'].StandardButton.Cancel
            )
            buttons.accepted.connect(self._accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

        def _add_row(self, label, hex_color):
            row = Q['QWidget']()
            h = Q['QHBoxLayout'](row)
            h.setContentsMargins(2, 1, 2, 1)
            h.setSpacing(8)

            swatch = _Swatch(hex_color, None, row)
            h.addWidget(swatch)

            # The label text is the whole point of the row -- it tells the
            # user which community/identity this swatch belongs to.
            text = Q['QLabel'](_label_text(label))
            text.setToolTip(_label_text(label))
            # Ink must contrast with the preview background, not the dialog's
            # own palette.
            dark = _rgb_to_oklab(normalize_background(self._background))[0] < 0.5
            text.setStyleSheet(
                f"color: {'#eeeeee' if dark else '#111111'}; "
                "background: transparent; padding-left: 2px;")
            h.addWidget(text, stretch=1)

            self._inner_layout.addWidget(row)
            self._rows[label] = (row, swatch, text)

        # --------------------------------------------------------- actions --
        def _apply_scheme(self, scheme):
            """Batch-fill every chip from an automatic scheme."""
            kwargs = dict(self._seeded_kwargs)
            kwargs.pop('scheme', None)
            kwargs.pop('custom_map', None)
            kwargs.pop('record', None)
            # record=False: previewing a scheme in the editor must not
            # redefine what 'Previous' means. Only an accepted, rendered
            # palette does that.
            cmap = resolve_palette(self._labels, scheme=scheme,
                                   background=self._background,
                                   record=False, **kwargs)
            for label, (_row, swatch, _t) in self._rows.items():
                if label in cmap:
                    swatch.set_hex(cmap[label])

        def _apply_filter(self, text):
            needle = (text or '').strip().lower()
            for label, (row, _s, _t) in self._rows.items():
                row.setVisible(needle in str(label).lower() if needle else True)

        def _accept(self):
            self._result = {label: swatch.hex()
                            for label, (_r, swatch, _t) in self._rows.items()}
            self.accept()

        def result_map(self):
            """The chosen {label: hex} map, or None if cancelled."""
            return self._result

    return _CustomColorDialog()


def can_prompt():
    """
    True when the custom color editor can actually be shown.

    Two separate failure modes matter here. PyQt6 may not be installed at all,
    or it may be installed with no QApplication running -- which is the normal
    case when community_extractor is driven from a script. Constructing a
    QWidget in that state aborts the process rather than raising, so this must
    be checked BEFORE building any dialog.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return False
    return QApplication.instance() is not None


def prompt_custom_colors(labels, parent=None, background='w',
                         initial_scheme=SCHEME_DEFAULT, initial_map=None,
                         category_name='Category', seeded_kwargs=None):
    """
    Show the custom color editor and return {label: '#rrggbb'}, or None if the
    user cancelled.

    The grid is seeded from `initial_map` when given, otherwise from
    `initial_scheme`, so the dialog always opens showing the colors that would
    render right now rather than an empty or arbitrary state.
    """
    labels = list(_unique_sorted(labels))
    if not labels:
        return None
    if not can_prompt():
        return None

    if not initial_map:
        kwargs = dict(seeded_kwargs or {})
        kwargs.pop('scheme', None)
        kwargs.pop('custom_map', None)
        kwargs.pop('record', None)
        initial_map = resolve_palette(labels, scheme=initial_scheme,
                                      background=background,
                                      record=False, **kwargs)

    dlg = make_custom_color_dialog(
        labels, parent=parent, background=background,
        initial_map=initial_map, category_name=category_name,
        seeded_kwargs=seeded_kwargs,
    )
    if dlg.exec():
        return dlg.result_map()
    return None
