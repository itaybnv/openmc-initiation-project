"""Geometry builders for the heterogeneous MTR plate-fuel line.

Convention: fuel plates are thin along X (stacking / plate-normal axis), wide
along Y (meat width), tall along Z (active height). Each plate is clad|meat|clad
in X; plates are separated by water channels. This mirrors the sphere line's
``geometry.py`` (explicit cell ids, ``reset_auto_ids()`` discipline) but is a
fully separate module -- nothing here touches the homogeneous sphere.

Three builders, increasing in scope:
- ``unit_cell``          : one plate + water gap, reflective BC -> k_inf lattice
- ``plate_stack``        : finite N-plate element, reflector + vacuum -> k_eff
- ``coupled_plate_packs``: two packs + central water gap + (asymmetric) graphite
                           reflector + detector cell (id 2) -> the loosely-coupled
                           Rossi-alpha system

Performance note: the interior is tiled by SIMPLE X-slab cells and the reflector
is the complement of a single active-box region (plus the graphite/detector
boxes). An earlier version defined the water as the complement of the ~180-term
union of every meat/clad slab, which made per-history tracking ~100x slower.

Nominal layer dimensions are the IAEA 10 MW benchmark standard element
(TECDOC-233/-643). The detector keeps ``DETECTOR_CELL_ID`` so the entire
collision_track / rossi_benchmark analysis core is reused unchanged. The pack
width (Y) represents several standard elements abreast (a homogenized fuel slab),
which is what brings a water-moderated HEU assembly to a near-critical k_eff.
"""

import openmc

from .geometry import DETECTOR_CELL_ID
from .plate_materials import fuel_meat, clad, water, graphite, detector_material

__all__ = [
    "T_CLAD",
    "T_MEAT",
    "T_PLATE",
    "WATER_GAP",
    "PITCH",
    "PLATE_WIDTH",
    "ACTIVE_HEIGHT",
    "MEAT_CELL_ID",
    "WATER_CELL_ID",
    "MEAT_CELL_BASE",
    "unit_cell",
    "plate_stack",
    "coupled_plate_packs",
    "stack_fuel_x_hi",
    "coupled_fuel_x_hi",
    "coupled_detector_positions",
]

# --- Nominal IAEA 10 MW benchmark standard-element layer dimensions (cm) ---
T_MEAT = 0.051  # fuel meat thickness
T_CLAD = 0.038  # aluminium clad, each side
T_PLATE = T_MEAT + 2 * T_CLAD  # 0.127
WATER_GAP = 0.223  # coolant channel between plates
PITCH = T_PLATE + WATER_GAP  # 0.350
PLATE_WIDTH = 6.65  # single standard-element width (meat 6.30); packs use wider slabs
ACTIVE_HEIGHT = 60.0  # fuelled height

# Cell ids. Detector = DETECTOR_CELL_ID (2). Per-plate meat cells = MEAT_CELL_BASE+i.
MEAT_CELL_BASE = 100
MEAT_CELL_ID = 11  # unit-cell meat/water (own tally) -- distinct from the finite builders
WATER_CELL_ID = 14
_CLAD_LO_CELL_ID = 12
_CLAD_HI_CELL_ID = 13
_STRUCT_ID_BASE = 1000  # clad / water-channel / reflector / graphite cells


def unit_cell(
    water_gap: float = WATER_GAP,
    width: float = PLATE_WIDTH,
    height: float = ACTIVE_HEIGHT,
) -> openmc.Geometry:
    """One clad|meat|clad plate + water gap, fully reflective -> infinite lattice.

    Used for k_inf, the thermal disadvantage factor (meat vs water flux), and the
    water-gap moderation scan. Lateral (Y, Z) extent is irrelevant under reflective
    BCs; the nominal plate footprint is used for concreteness.
    """
    openmc.reset_auto_ids()
    pitch = T_PLATE + water_gap
    x0 = openmc.XPlane(0.0, boundary_type="reflective")
    x1 = openmc.XPlane(T_CLAD)
    x2 = openmc.XPlane(T_CLAD + T_MEAT)
    x3 = openmc.XPlane(T_PLATE)
    x4 = openmc.XPlane(pitch, boundary_type="reflective")
    ylo = openmc.YPlane(-width / 2, boundary_type="reflective")
    yhi = openmc.YPlane(width / 2, boundary_type="reflective")
    zlo = openmc.ZPlane(-height / 2, boundary_type="reflective")
    zhi = openmc.ZPlane(height / 2, boundary_type="reflective")
    lateral = +ylo & -yhi & +zlo & -zhi
    clad_lo = openmc.Cell(cell_id=_CLAD_LO_CELL_ID, fill=clad, region=+x0 & -x1 & lateral)
    meat = openmc.Cell(cell_id=MEAT_CELL_ID, fill=fuel_meat, region=+x1 & -x2 & lateral)
    clad_hi = openmc.Cell(cell_id=_CLAD_HI_CELL_ID, fill=clad, region=+x2 & -x3 & lateral)
    chan = openmc.Cell(cell_id=WATER_CELL_ID, fill=water, region=+x3 & -x4 & lateral)
    return openmc.Geometry([clad_lo, meat, clad_hi, chan])


# --- Fuel envelope helpers (for the box source / detector placement) ---

def stack_fuel_x_hi(n_plates: int) -> float:
    return n_plates * PITCH


def coupled_fuel_x_hi(n_a: int, n_b: int, coupling_gap: float) -> float:
    return n_a * PITCH + coupling_gap + n_b * PITCH


def coupled_detector_positions(n_a=12, n_b=12, coupling_gap=3.0, refl_water=12.0):
    """Two contrasting detector centres: 'left' sits in the -X water gap backed by
    the graphite reflector (slow reflector-return mode); 'right' sits in the bare
    +X water. Their apparent prompt-alpha differs iff higher spatial modes matter.
    """
    x_hi = coupled_fuel_x_hi(n_a, n_b, coupling_gap)
    return {"left": (-refl_water / 2, 0.0, 0.0), "right": (x_hi + refl_water / 2, 0.0, 0.0)}


# --- Interval tiling (simple, fast slab cells) ---

def _pack_intervals(x_start: float, n_plates: int, meat_i0: int):
    """Return (intervals, x_end, next_meat_i) for one plate pack.

    Each interval is (x_lo, x_hi, tag); tag is 'clad', 'water', or ('meat', i).
    """
    iv = []
    for j in range(n_plates):
        x = x_start + j * PITCH
        iv.append((x, x + T_CLAD, "clad"))
        iv.append((x + T_CLAD, x + T_CLAD + T_MEAT, ("meat", meat_i0 + j)))
        iv.append((x + T_CLAD + T_MEAT, x + T_PLATE, "clad"))
        iv.append((x + T_PLATE, x + PITCH, "water"))
    return iv, x_start + n_plates * PITCH, meat_i0 + n_plates


def _geometry_from_intervals(intervals, x_hi, width, height, refl_water,
                             graphite_thickness, detector_center, detector_half):
    """Assemble slab cells + reflector/graphite/(detector) inside a vacuum box."""
    ay_lo = openmc.YPlane(-width / 2)
    ay_hi = openmc.YPlane(width / 2)
    az_lo = openmc.ZPlane(-height / 2)
    az_hi = openmc.ZPlane(height / 2)
    lateral = +ay_lo & -ay_hi & +az_lo & -az_hi

    # Shared X planes (rounded keys so coincident boundaries reuse one surface).
    planes: dict[float, openmc.XPlane] = {}

    def xp(x: float) -> openmc.XPlane:
        key = round(x, 7)
        if key not in planes:
            planes[key] = openmc.XPlane(key)
        return planes[key]

    cells = []
    cid = _STRUCT_ID_BASE
    for x_lo, x_hi_i, tag in intervals:
        reg = +xp(x_lo) & -xp(x_hi_i) & lateral
        if isinstance(tag, tuple):
            cells.append(openmc.Cell(cell_id=MEAT_CELL_BASE + tag[1], fill=fuel_meat, region=reg))
        else:
            cells.append(openmc.Cell(cell_id=cid, fill=(clad if tag == "clad" else water), region=reg))
            cid += 1

    active = +xp(0.0) & -xp(x_hi) & lateral

    # Outer box (vacuum). Graphite backs the -X face; water elsewhere.
    y_out = width / 2 + refl_water
    z_out = height / 2 + refl_water
    x_out_lo = -refl_water - (graphite_thickness if graphite_thickness > 0 else 0.0)
    x_out_hi = x_hi + refl_water
    ox_lo = openmc.XPlane(x_out_lo, boundary_type="vacuum")
    ox_hi = openmc.XPlane(x_out_hi, boundary_type="vacuum")
    oy_lo = openmc.YPlane(-y_out, boundary_type="vacuum")
    oy_hi = openmc.YPlane(y_out, boundary_type="vacuum")
    oz_lo = openmc.ZPlane(-z_out, boundary_type="vacuum")
    oz_hi = openmc.ZPlane(z_out, boundary_type="vacuum")
    outer = +ox_lo & -ox_hi & +oy_lo & -oy_hi & +oz_lo & -oz_hi

    excl = active
    if graphite_thickness > 0:
        gx_hi = openmc.XPlane(-refl_water)
        graph_region = +ox_lo & -gx_hi & +oy_lo & -oy_hi & +oz_lo & -oz_hi
        cells.append(openmc.Cell(cell_id=60, fill=graphite, region=graph_region))
        excl = excl | graph_region

    if detector_center is not None:
        cx, cy, cz = detector_center
        hx, hy, hz = detector_half
        det_region = (+openmc.XPlane(cx - hx) & -openmc.XPlane(cx + hx)
                      & +openmc.YPlane(cy - hy) & -openmc.YPlane(cy + hy)
                      & +openmc.ZPlane(cz - hz) & -openmc.ZPlane(cz + hz))
        cells.append(openmc.Cell(cell_id=DETECTOR_CELL_ID, fill=detector_material, region=det_region))
        excl = excl | det_region

    cells.append(openmc.Cell(cell_id=70, fill=water, region=outer & ~excl))
    return openmc.Geometry(cells)


def plate_stack(
    n_plates: int = 23,
    refl_water: float = 12.0,
    graphite_thickness: float = 0.0,
    width: float = PLATE_WIDTH,
    height: float = ACTIVE_HEIGHT,
) -> openmc.Geometry:
    """Finite N-plate element/slab in a water (optionally graphite-backed) reflector.

    Vacuum outer boundary => a genuine k_eff.
    """
    openmc.reset_auto_ids()
    intervals, x_hi, _ = _pack_intervals(0.0, n_plates, 0)
    return _geometry_from_intervals(intervals, x_hi, width, height, refl_water,
                                    graphite_thickness, None, None)


def coupled_plate_packs(
    n_a: int = 23,
    n_b: int = 23,
    coupling_gap: float = 6.0,
    refl_water: float = 12.0,
    graphite_thickness: float = 10.0,
    detector_center: tuple[float, float, float] | None = None,
    detector_half: tuple[float, float, float] = (0.5, 3.325, 10.0),
    width: float = 40.0,
    height: float = ACTIVE_HEIGHT,
) -> openmc.Geometry:
    """Two plate packs separated by a central water channel -- the loosely-coupled
    Rossi-alpha system.

    Asymmetry (for detector-position-dependent alpha): a graphite reflector backs
    the -X face only (slow reflector-return mode), while +X leaks into bare water.
    ``detector_center=None`` builds no detector (for k_eff / dominance-ratio runs);
    otherwise a B-10 box (cell id 2) is placed there, carved from the water so the
    collision_track / rossi_benchmark analysis core is reused unchanged.
    """
    openmc.reset_auto_ids()
    iv_a, xa_end, mi = _pack_intervals(0.0, n_a, 0)
    iv_gap = [(xa_end, xa_end + coupling_gap, "water")]
    iv_b, xb_end, _ = _pack_intervals(xa_end + coupling_gap, n_b, mi)
    intervals = iv_a + iv_gap + iv_b
    return _geometry_from_intervals(intervals, xb_end, width, height, refl_water,
                                    graphite_thickness, detector_center, detector_half)
