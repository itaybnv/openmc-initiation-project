import openmc
import numpy as np

def create_sphere_by_h_to_u_ratio(h_to_u_ratio, u235_density=0.048, radius=50.0):
    """
    Create sphere with specified H/U atomic ratio
    
    Parameters:
    -----------
    h_to_u_ratio : float
        Atomic ratio of H to U (typical range: 10-1000 for subcritical)
    u235_density : float
        U-235 atomic density (atoms/barn-cm)
    """
    
    # Calculate water density from H/U ratio
    # Each water molecule has 2 H atoms
    water_molecules_per_u = h_to_u_ratio / 2.0
    water_density = u235_density * water_molecules_per_u
    
    # Create materials
    fuel_mix = openmc.Material(name='U235_H2O_mix')
    fuel_mix.add_nuclide('U235', u235_density, 'ao')
    fuel_mix.add_element('H', 2.0 * water_density, 'ao')
    fuel_mix.add_element('O', water_density, 'ao')
    fuel_mix.add_s_alpha_beta('c_H_in_H2O')
    
    # Set total density - this is important!
    # For mixture, you can let OpenMC calculate or set explicitly
    fuel_mix.set_density('sum')  # auto-calculate from constituents
    
    # Geometry
    sphere = openmc.Sphere(r=radius, boundary_type='vacuum')
    cell = openmc.Cell(fill=fuel_mix, region=-sphere)
    geom = openmc.Geometry([cell])
    
    return geom, fuel_mix


def find_keff_for_system(h_to_u_ratio, target_keff=0.98, 
                         u235_density=0.048, radius=50.0):
    """
    Run eigenvalue calculation for given configuration
    """
    
    geom, mat = create_sphere_by_h_to_u_ratio(h_to_u_ratio, u235_density, radius)
    
    # Settings
    settings = openmc.Settings()
    settings.batches = 150
    settings.inactive = 50
    settings.particles = 10000
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0))
    )
    
    # Run
    model = openmc.Model(geom, materials=openmc.Materials([mat]), settings=settings)
    model.run()
    
    # Get k_eff
    sp = openmc.StatePoint('statepoint.150.h5')
    keff = sp.keff
    
    return keff.nominal_value, keff.std_dev


from scipy.optimize import brentq

def keff_residual(h_to_u_ratio, target_keff, u_density, radius):
    """Residual function for root finding"""
    keff, _ = find_keff_for_system(h_to_u_ratio, target_keff, u_density, radius)
    return keff - target_keff

# Find H/U ratio that gives target k_eff
# target_keff = 0.98
# h_to_u_optimal = brentq(
#     lambda x: keff_residual(x, target_keff, u_density=0.048, radius=50.0),
#     1,    # lower bound for H/U
#     10000   # upper bound
# )

print(h_to_u_optimal)