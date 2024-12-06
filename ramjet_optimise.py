import numpy as np
from scipy.optimize import differential_evolution

# ------------------------------------------------------------
# Parameterization of the Ramjet Geometry
# ------------------------------------------------------------
# We fix total length = 100 mm.
# Let’s define a parameterization:
# 1. Spike angle (degrees)
# 2. Inlet fraction (0 < inlet_length < total_length)
# 3. Combustion chamber fraction
# 4. Nozzle fraction
# 5. Nozzle contraction ratio
#
# The sum of inlet_length + combustion_chamber_length + nozzle_length = 100 mm - spike_length
#
# Assume a tiny spike at the front: spike_length = a few mm (e.g., 5 mm)
# Then 95 mm remains to be split among inlet, combustor, and nozzle.
# We define:
# x = [spike_angle, inlet_fraction, combustor_fraction, nozzle_fraction, nozzle_contraction_ratio]
#
# Constraints:
# spike_angle: 5 to 30 degrees
# inlet_fraction: 0.1 to 0.5 (of the 95 mm)
# combustor_fraction: 0.2 to 0.6
# nozzle_fraction: remainder (but we’ll set it with constraints)
# nozzle_contraction_ratio: 0.7 to 1.0 (just a guess)
#
# The fractions must sum to ≤ 1.0: inlet_fraction + combustor_fraction + nozzle_fraction = 1
# We'll allow some slack and define nozzle_fraction = 1 - inlet_fraction - combustor_fraction internally.


def evaluate_design(params):
    """
    A surrogate performance evaluation for the ramjet design.
    In reality, this would call a CFD solver or a complex model.
    """
    spike_angle, inlet_fraction, combustor_fraction, nozzle_contraction_ratio = params
    
    # Fixed parameters
    total_length = 100.0  # mm
    spike_length = 5.0
    available_length = total_length - spike_length
    
    # Compute nozzle fraction
    nozzle_fraction = 1.0 - inlet_fraction - combustor_fraction
    
    # If nozzle_fraction < 0 or > 1, penalty:
    if nozzle_fraction < 0.05 or nozzle_fraction > 0.8:
        return 1e6  # massive penalty
    
    # Check geometric sanity:
    # Spike angle too large or too small might affect shock formation
    # We already constrained that in parameter bounds, but let’s add mild penalties:
    if spike_angle < 5 or spike_angle > 30:
        return 1e6
    
    # Compute actual lengths:
    inlet_length = inlet_fraction * available_length
    combustor_length = combustor_fraction * available_length
    nozzle_length = nozzle_fraction * available_length
    
    # Very rough surrogate for efficiency:
    # Let's say efficiency depends on:
    # - A moderate spike angle: best ~ 12 degrees
    # - A balanced division of lengths: too short combustor reduces mixing, too long reduces efficiency
    # - A nozzle contraction ratio close to a certain sweet spot (e.g., ~0.85)
    
    # Base efficiency model:
    # Start from a hypothetical baseline efficiency = 0.3
    efficiency = 0.3
    
    # Spike angle effect (quadratic penalty away from 12 degrees)
    efficiency -= 0.01 * (spike_angle - 12)**2
    
    # Ideal combustor fraction ~0.4
    efficiency -= 0.02 * (combustor_fraction - 0.4)**2
    
    # Ideal nozzle fraction ~0.3
    efficiency -= 0.015 * (nozzle_fraction - 0.3)**2
    
    # Ideal inlet fraction ~0.3 as well
    efficiency -= 0.015 * (inlet_fraction - 0.3)**2
    
    # Nozzle contraction ratio effect (ideal around 0.85)
    efficiency -= 0.02 * (nozzle_contraction_ratio - 0.85)**2
    
    # Length-based penalty: if any length is too small, mixing/burn incomplete
    if inlet_length < 5.0:  # too short inlet
        efficiency -= 0.1
    if combustor_length < 10.0:  # too short combustor
        efficiency -= 0.1
    if nozzle_length < 5.0:  # too short nozzle
        efficiency -= 0.1
    
    # Efficiency > 0 is good; if it drops below zero, it’s terrible
    # Our goal: MAXIMIZE efficiency. The optimizer we use (differential_evolution) MINIMIZES the objective, 
    # so we’ll return negative efficiency so minimizing negative efficiency is equivalent to maximizing efficiency.
    return -efficiency

# ------------------------------------------------------------
# Optimization Setup
# ------------------------------------------------------------
# Bounds for parameters:
# spike_angle: [5,30]
# inlet_fraction: [0.1,0.5]
# combustor_fraction: [0.2,0.6]
# nozzle_contraction_ratio: [0.7,1.0]
#
# We'll let the nozzle fraction be defined by these and enforced via penalty.

bounds = [
    (5, 30),    # spike_angle
    (0.1, 0.5), # inlet_fraction
    (0.2, 0.6), # combustor_fraction
    (0.7, 1.0)  # nozzle_contraction_ratio
]

# We will use differential_evolution for a global search.
result = differential_evolution(evaluate_design, bounds, maxiter=100, popsize=20, tol=1e-6, mutation=(0.5,1.0), recombination=0.7)

print("Optimization Complete!")
print("Best parameters found:")
print("Spike Angle (deg):", result.x[0])
print("Inlet Fraction:", result.x[1])
print("Combustor Fraction:", result.x[2])
print("Nozzle Contraction Ratio:", result.x[3])

best_efficiency = -result.fun
print(f"Approximate Efficiency of Best Design: {best_efficiency:.4f}")

# From here, you'd take these parameters, refine the design in CAD,
# run high-fidelity CFD, and iterate to truly optimize your micro-ramjet.
