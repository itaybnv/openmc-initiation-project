import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Vacuum boundary - large sphere at center
vacuum_radius = 45
vacuum_sphere = patches.Circle(
    (0, 0),
    vacuum_radius,
    color="#ECF0F1",
    alpha=0.3,
    linewidth=2,
    edgecolor="#34495E",
    linestyle="--",
    zorder=0,
)
ax.add_patch(vacuum_sphere)

# Main sphere (homogeneous mixture) - 20 cm radius at center
mixture_radius = 20
mixture_diameter = 2 * mixture_radius
main_sphere = patches.Circle(
    (0, 0),
    mixture_radius,
    color="#4A90E2",
    alpha=0.8,
    linewidth=3,
    edgecolor="#2E5C8A",
    zorder=2,
)
ax.add_patch(main_sphere)

# Detector sphere - 5 cm radius, offset 10 cm from edge of mixture sphere
detector_radius = 5
detector_diameter = 2 * detector_radius
gap = 10  # 10 cm gap from mixture sphere surface
detector_center_x = (
    mixture_radius + gap + detector_radius
)  # 20 + 10 + 5 = 35 cm from origin
detector_center_y = 0
detector_sphere = patches.Circle(
    (detector_center_x, detector_center_y),
    detector_radius,
    color="#E74C3C",
    alpha=0.8,
    linewidth=3,
    edgecolor="#A93226",
    zorder=2,
)
ax.add_patch(detector_sphere)

# Add labels for regions
ax.text(
    0,
    0,
    "Homogeneous\nMixture",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
    zorder=3,
)
ax.text(
    detector_center_x,
    detector_center_y,
    "Detector",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color="black",
    zorder=3,
)
ax.text(
    0,
    32,
    "Vacuum",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
    style="italic",
    zorder=3,
)

# Mixture sphere diameter annotation (horizontal, outside sphere)
offset_y = -28  # Position below the sphere
ax.annotate(
    "",
    xy=(-mixture_radius, offset_y),
    xytext=(mixture_radius, offset_y),
    arrowprops=dict(arrowstyle="<->", color="black", lw=2),
)
# Add vertical connection lines from sphere to annotation
ax.plot(
    [-mixture_radius, -mixture_radius],
    [-mixture_radius, offset_y],
    "k--",
    lw=1,
    alpha=0.5,
)
ax.plot(
    [mixture_radius, mixture_radius],
    [-mixture_radius, offset_y],
    "k--",
    lw=1,
    alpha=0.5,
)
ax.text(
    0,
    offset_y - 3.5,
    f"D = {mixture_diameter} cm",
    fontsize=13,
    fontweight="bold",
    ha="center",
    color="black",
)

# Detector sphere diameter annotation (horizontal, outside sphere)
detector_offset_y = -15  # Position below the detector
ax.annotate(
    "",
    xy=(detector_center_x - detector_radius, detector_offset_y),
    xytext=(detector_center_x + detector_radius, detector_offset_y),
    arrowprops=dict(arrowstyle="<->", color="black", lw=2),
)
# Add vertical connection lines from sphere to annotation
ax.plot(
    [detector_center_x - detector_radius, detector_center_x - detector_radius],
    [detector_center_y - detector_radius, detector_offset_y],
    "k--",
    lw=1,
    alpha=0.5,
)
ax.plot(
    [detector_center_x + detector_radius, detector_center_x + detector_radius],
    [detector_center_y - detector_radius, detector_offset_y],
    "k--",
    lw=1,
    alpha=0.5,
)
ax.text(
    detector_center_x,
    detector_offset_y - 3.5,
    f"d = {detector_diameter} cm",
    fontsize=13,
    fontweight="bold",
    ha="center",
    color="black",
)

# Gap annotation (horizontal, outside spheres)
gap_offset_y = 28  # Position above the spheres
gap_start_x = mixture_radius
gap_end_x = detector_center_x - detector_radius
ax.annotate(
    "",
    xy=(gap_start_x, gap_offset_y),
    xytext=(gap_end_x, gap_offset_y),
    arrowprops=dict(arrowstyle="<->", color="black", lw=2),
)
# Add vertical connection lines from sphere surfaces to annotation
ax.plot(
    [gap_start_x, gap_start_x], [mixture_radius, gap_offset_y], "k--", lw=1, alpha=0.5
)
ax.plot([gap_end_x, gap_end_x], [detector_radius, gap_offset_y], "k--", lw=1, alpha=0.5)
ax.text(
    (gap_start_x + gap_end_x) / 2,
    gap_offset_y + 3.5,
    f"Gap = {gap} cm",
    fontsize=13,
    fontweight="bold",
    ha="center",
    color="black",
)

# Styling
ax.set_xlim(-55, 55)
ax.set_ylim(-55, 55)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3, linestyle=":", color="gray")
ax.axhline(y=0, color="k", linewidth=0.8, alpha=0.4)
ax.axvline(x=0, color="k", linewidth=0.8, alpha=0.4)
ax.set_xlabel("X (cm)", fontsize=14, fontweight="bold")
ax.set_ylabel("Y (cm)", fontsize=14, fontweight="bold")

# Add legend
legend_elements = [
    patches.Patch(
        facecolor="#4A90E2",
        edgecolor="#2E5C8A",
        label=f"Homogeneous Mixture (D={mixture_diameter} cm)",
    ),
    patches.Patch(
        facecolor="#E74C3C",
        edgecolor="#A93226",
        label=f"Detector Sphere (d={detector_diameter} cm)",
    ),
    patches.Patch(
        facecolor="#ECF0F1",
        edgecolor="#34495E",
        linestyle="--",
        label=f"Vacuum Boundary",
    ),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig(
    "results/figures/system_plot.png", dpi=300, bbox_inches="tight", facecolor="white"
)
plt.show()

print("Plot saved as 'system_plot.png'")
print(f"\nGeometry Summary:")
print(f"  Vacuum boundary: R = {vacuum_radius} cm (centered at origin)")
print(f"  Mixture sphere: D = {mixture_diameter} cm (centered at origin)")
print(
    f"  Detector sphere: d = {detector_diameter} cm (center at x={detector_center_x} cm, y=0)"
)
print(f"  Gap between surfaces: {gap} cm")
