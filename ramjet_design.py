import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Physical constants
gamma = 1.4  # Ratio of specific heats for air
R = 287.0    # Gas constant for air [J/kg·K]
T0 = 288.15  # Freestream temperature [K]
P0 = 101325  # Freestream pressure [Pa]
M0 = 2.5     # Design Mach number for ramjet
Cp = 1004.5  # Specific heat at constant pressure [J/kg·K]
Pr = 0.72    # Prandtl number
mu0 = 1.789e-5  # Dynamic viscosity at reference temperature [kg/m·s]

# Geometric constraints (all in mm)
total_length = 100.0   
radius_outer = 10.0    

def real_gas_properties(M, T):
    """Calculate real gas properties with temperature dependence."""
    # Temperature-dependent specific heat ratio
    gamma_T = gamma * (1 - 0.00002 * (T - 288.15))
    # Temperature-dependent gas constant
    R_T = R * (1 + 0.00003 * (T - 288.15))
    return gamma_T, R_T

def oblique_shock_angle(M1, theta):
    """Calculate oblique shock angle beta given M1 and deflection angle theta."""
    def shock_equation(beta):
        return np.tan(theta) - 2 * (1/np.tan(beta)) * \
               ((M1**2 * np.sin(beta)**2 - 1)/(M1**2 * (gamma + np.cos(2*beta)) + 2))
    
    # Initial guess for shock angle
    beta_guess = theta + 0.1
    beta = fsolve(shock_equation, beta_guess)[0]
    return beta

def shock_properties(M1, beta, theta):
    """Calculate flow properties after oblique shock."""
    M1n = M1 * np.sin(beta)
    M2n = np.sqrt((1 + (gamma-1)/2 * M1n**2)/(gamma*M1n**2 - (gamma-1)/2))
    M2 = M2n/np.sin(beta - theta)
    
    # Pressure and temperature ratios
    P2_P1 = 1 + 2*gamma/(gamma+1) * (M1n**2 - 1)
    T2_T1 = P2_P1 * (1 + (gamma-1)/2 * M1n**2)/(1 + (gamma-1)/2 * M2n**2)
    
    return M2, P2_P1, T2_T1

def optimize_geometry():
    """Optimize key geometric parameters with improved constraints."""
    # Define bounds for M=2.5 outside the objective function
    bounds = [
        (4.0, 9.0),     # radius_inlet (increased minimum)
        (3.5, 8.0),     # radius_throat (adjusted for higher compression)
        (4.5, 9.5),     # radius_exit (increased for higher expansion)
        (15.0, 25.0),   # spike_length (longer for better compression)
        (12.0, 18.0),   # theta1 (increased angles for stronger shocks)
        (14.0, 20.0)    # theta2 (increased for higher compression)
    ]
    
    def objective(params):
        radius_inlet, radius_throat, radius_exit, spike_length, theta1, theta2 = params
        
        try:
            # Adjust shock angles for M=2.5
            beta1 = oblique_shock_angle(M0, np.radians(theta1))
            M1, P1_P0, T1_T0 = shock_properties(M0, beta1, np.radians(theta1))
            
            # Second shock should be weaker to avoid separation
            beta2 = oblique_shock_angle(M1, np.radians(theta2-theta1))
            M2, P2_P1, T2_T1 = shock_properties(M1, beta2, np.radians(theta2-theta1))
            
            # Calculate performance with real gas effects
            T1 = T0 * (1 + (gamma-1)/2 * M0**2)
            gamma_T, R_T = real_gas_properties(M0, T1)
            
            # Calculate total pressure recovery
            P_recovery = P1_P0 * P2_P1
            
            # Calculate mass flow and thrust
            A_inlet = np.pi * radius_inlet**2
            mdot = P0/(R_T*T0) * M0 * np.sqrt(gamma_T*R_T*T0) * A_inlet
            
            # Enhanced penalties for better constraints
            penalties = 0
            if radius_inlet >= radius_outer:
                penalties += 1000 * (radius_inlet - radius_outer)
            if radius_throat >= radius_inlet:
                penalties += 1000 * (radius_throat - radius_inlet)
            if radius_exit <= radius_throat:
                penalties += 1000 * (radius_throat - radius_exit)
            if spike_length >= total_length/3:
                penalties += 1000 * (spike_length - total_length/3)
            
            # Multi-objective optimization
            thrust_term = mdot * P_recovery
            drag_term = calculate_drag(radius_inlet, spike_length, M0)
            efficiency = thrust_term / (1 + drag_term)
            
            return -efficiency + penalties
            
        except:
            return 1e10
    
    # Use differential evolution for better global optimization
    result = differential_evolution(
        objective, 
        bounds, 
        popsize=20,
        mutation=(0.5, 1.0),
        recombination=0.7,
        maxiter=100
    )
    return result.x

def calculate_drag(radius, length, mach):
    """Calculate approximate drag coefficient."""
    Re = P0 * mach * np.sqrt(gamma*R*T0) * length / (R*T0 * mu0)
    Cf = 0.074 / Re**0.2  # Turbulent flat plate skin friction
    
    # Form drag coefficient
    Cd_form = 0.1 * (radius/length)**2
    
    # Wave drag coefficient (simplified)
    Cd_wave = 0.1 * (mach - 1)**2
    
    return Cf + Cd_form + Cd_wave

def generate_spike_profile():
    """Generate optimized spike profile with improved curve fitting."""
    params = optimize_geometry()
    radius_inlet, _, _, spike_length, theta1, theta2 = params
    
    # Use cubic spline for smoother compression surface
    x_external = np.linspace(0, spike_length, 50)
    
    def improved_compression_curve(x):
        t = x/spike_length
        # Cubic spline profile
        angle = theta1 * (3*t**2 - 2*t**3) * (1 - 0.5*t)
        return np.cumsum(np.tan(np.radians(angle))) * (x[1] - x[0])
    
    y_external = improved_compression_curve(x_external)
    
    # Add internal support section (extends into diffuser)
    internal_length = spike_length * 0.5  # Support extends 50% more into engine
    x_internal = np.linspace(spike_length, spike_length + internal_length, 20)
    
    # Create support strut profile (tapered cylinder)
    y_internal = y_external[-1] * np.exp(-(x_internal - spike_length)/(internal_length/3))
    
    # Combine profiles
    x = np.concatenate([x_external, x_internal])
    y = np.concatenate([y_external, y_internal])
    
    # Calculate flow properties for curved profile
    # Use average angle for property calculations
    avg_theta = np.arctan(np.gradient(y, x)[len(x)//2])
    beta1 = oblique_shock_angle(M0, avg_theta)
    M1, P1_P0, T1_T0 = shock_properties(M0, beta1, avg_theta)
    
    # Second compression through curved section
    avg_theta2 = np.arctan(np.gradient(y, x)[3*len(x)//4]) - avg_theta
    beta2 = oblique_shock_angle(M1, avg_theta2)
    M2, P2_P1, T2_T1 = shock_properties(M1, beta2, avg_theta2)
    
    return x, y, M2, P1_P0*P2_P1, T1_T0*T2_T1, params

def generate_flow_path():
    """Generate flow path with improved compression for M=2.5."""
    _, _, M_diff, P_diff, T_diff, params = generate_spike_profile()
    radius_inlet, radius_throat, radius_exit, spike_length, _, _ = params
    
    # Start cowl slightly earlier to better capture shock
    cowl_start_x = spike_length * 0.80
    
    # Generate sections
    x_diffuser = np.linspace(cowl_start_x, cowl_start_x + 25, 30)  # Longer diffuser
    x_combustor = np.linspace(x_diffuser[-1], x_diffuser[-1] + 40, 30)
    x_nozzle = np.linspace(x_combustor[-1], total_length, 30)
    
    # Initial sharp compression followed by controlled diffusion
    t_diff = (x_diffuser - x_diffuser[0])/(x_diffuser[-1] - x_diffuser[0])
    
    def diffuser_profile(t):
        # Sharp initial turn (15 degrees)
        initial_angle = np.radians(15)
        # Blend between initial angle and final diffuser angle
        angle = initial_angle * (1 - t)**2
        # Calculate radius change
        dr = np.cumsum(np.tan(angle)) * (x_diffuser[1] - x_diffuser[0])
        return radius_outer - dr
    
    y_diffuser = diffuser_profile(t_diff)
    
    # Constant area combustor with very slight divergence
    t_comb = (x_combustor - x_combustor[0])/(x_combustor[-1] - x_combustor[0])
    combustor_radius = y_diffuser[-1]  # Match diffuser exit
    y_combustor = combustor_radius + 0.05 * t_comb  # Minimal divergence
    
    # CD nozzle with optimized contour
    t_nozzle = (x_nozzle - x_nozzle[0])/(x_nozzle[-1] - x_nozzle[0])
    
    def nozzle_profile(t):
        # Method of characteristics-inspired contour
        if t < 0.2:  # Convergent section
            return y_combustor[-1] + (radius_throat - y_combustor[-1]) * (t/0.2)**2
        else:  # Divergent section
            t_adj = (t - 0.2)/0.8
            return radius_throat + (radius_exit - radius_throat) * \
                   (1.5 * t_adj - 0.5 * t_adj**2)
    
    y_nozzle = np.array([nozzle_profile(t) for t in t_nozzle])
    
    # Combine all sections
    x = np.concatenate([x_diffuser, x_combustor, x_nozzle])
    y_upper = np.concatenate([y_diffuser, y_combustor, y_nozzle])
    y_lower = -y_upper
    
    return x, y_upper, y_lower

def plot_ramjet():
    """Plot the ramjet design with flow properties."""
    plt.figure(figsize=(12, 6))
    
    # Generate geometries - now correctly unpacking 6 values
    spike_x, spike_y, M_spike, P_ratio, T_ratio, params = generate_spike_profile()
    x_flow, y_upper, y_lower = generate_flow_path()
    
    # Plot components
    plt.plot([0, total_length], [radius_outer, radius_outer], 'k--', label='Outer Casing')
    plt.plot([0, total_length], [-radius_outer, -radius_outer], 'k--')
    
    plt.plot(x_flow, y_upper, 'b-', label='Flow Path Upper')
    plt.plot(x_flow, y_lower, 'b-')
    plt.fill_between(x_flow, y_upper, y_lower, color='lightblue', alpha=0.3)
    
    plt.plot(spike_x, spike_y, 'g-', label='Spike')
    plt.plot(spike_x, -spike_y, 'g-')
    
    # Add section labels with key properties
    plt.text(10, 0, f'Spike\nM={M_spike:.1f}\nP/P0={P_ratio:.1f}', ha='center')
    plt.text(35, 0, 'Diffuser', ha='center')
    plt.text(60, 0, 'Combustion\nChamber', ha='center')
    plt.text(85, 0, 'Nozzle\nM=2.5', ha='center')
    
    plt.xlabel('Axial Distance (mm)')
    plt.ylabel('Radius (mm)')
    plt.title('Optimized Small-Scale Ramjet Design')
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def calculate_performance():
    """Enhanced performance calculations with additional metrics."""
    _, _, M_spike, P_ratio, T_ratio, params = generate_spike_profile()
    radius_inlet, radius_throat, radius_exit = params[:3]
    
    # Convert dimensions from mm to m for calculations
    radius_inlet = radius_inlet / 1000  # mm to m
    radius_exit = radius_exit / 1000    # mm to m
    
    A_inlet = np.pi * radius_inlet**2
    A_exit = np.pi * radius_exit**2
    
    # Calculate mass flow rate with real gas effects
    T1 = T0 * (1 + (gamma-1)/2 * M0**2)
    gamma_T, R_T = real_gas_properties(M0, T1)
    
    rho0 = P0/(R_T*T0)
    V0 = M0 * np.sqrt(gamma_T*R_T*T0)
    mdot = rho0 * V0 * A_inlet
    
    # Enhanced thrust calculation
    P_exit = P0  # Assume optimal expansion
    T_exit = T0 * T_ratio
    V_exit = 2.5 * np.sqrt(gamma_T*R_T*T_exit)
    
    thrust = mdot * (V_exit - V0) + A_exit * (P_exit - P0)
    
    # Calculate additional performance metrics
    isp = thrust / (mdot * 9.81)  # Specific impulse
    
    # Improved thermal efficiency calculation using Brayton cycle approach
    T2 = T1 * P_ratio**((gamma-1)/gamma)  # Temperature after compression
    T3 = T_exit  # Temperature after combustion
    
    # Calculate enthalpies at each point
    h0 = Cp * T0
    h1 = Cp * T1
    h2 = Cp * T2
    h3 = Cp * T3
    
    # Energy input from fuel (heat addition in combustor)
    q_in = h3 - h2
    
    # Compression work (negative as work is done on the fluid)
    w_comp = h1 - h0
    
    # Expansion work (positive as work is done by the fluid)
    w_exp = h3 - h2
    
    # Net work output
    w_net = w_exp - abs(w_comp)
    
    # Calculate thermal efficiency
    thermal_efficiency = w_net / q_in if q_in > 0 else 0
    
    # Debug prints
    print("\nDebug Information:")
    print(f"T0 (ambient): {T0:.2f} K")
    print(f"T1 (after ram compression): {T1:.2f} K")
    print(f"T2 (after diffuser): {T2:.2f} K")
    print(f"T3 (after combustion): {T3:.2f} K")
    print(f"V0: {V0:.2f} m/s")
    print(f"V_exit: {V_exit:.2f} m/s")
    print(f"Compression work: {w_comp:.2f} J/kg")
    print(f"Heat added: {q_in:.2f} J/kg")
    print(f"Expansion work: {w_exp:.2f} J/kg")
    print(f"Net work: {w_net:.2f} J/kg")
    print(f"Raw thermal efficiency: {thermal_efficiency:.4f}")
    
    return {
        'thrust': thrust,
        'specific_impulse': isp,
        'thermal_efficiency': thermal_efficiency,
        'pressure_ratio': P_ratio,
        'temperature_ratio': T_ratio,
        'mass_flow': mdot
    }

if __name__ == "__main__":
    plot_ramjet()
    performance = calculate_performance()
    print("\nPerformance Metrics:")
    print(f"Thrust: {performance['thrust']/1000:.2f} kN")
    print(f"Specific Impulse: {performance['specific_impulse']:.1f} s")
    print(f"Thermal Efficiency: {performance['thermal_efficiency']*100:.1f}%")
    print(f"Pressure Recovery: {performance['pressure_ratio']:.3f}")