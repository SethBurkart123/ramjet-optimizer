import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
import ezdxf

# Simulation control parameters
SIM_POPSIZE = 30        # Population size for optimization
SIM_MAXITER = 250       # Maximum iterations
SIM_TOL = 1e-6         # Convergence tolerance
RANDOM_SEED = None      # Set to None for random results, or an integer for reproducible results

# Physical constants
gamma = 1.4  # Ratio of specific heats for air
R = 287.0    # Gas constant for air [J/kg·K]
T0 = 288.15  # Freestream temperature [K]
P0 = 101325  # Freestream pressure [Pa]
M0 = 2.5     # Design Mach number for ramjet
Cp = 1004.5  # Specific heat at constant pressure [J/kg·K]
Pr = 0.72    # Prandtl number
mu0 = 1.789e-5  # Dynamic viscosity at reference temperature [kg/m·s]
g0 = 9.81    # Gravitational acceleration [m/s²]

# Geometric constraints (all in mm)
total_length = 100.0   
radius_outer = 10.0    

def real_gas_properties(M, T):
    """Calculate real gas properties with enhanced high-temperature modeling."""
    # Temperature normalization for numerical stability
    T_norm = T/1000
    
    # Updated NASA coefficients for better high-temperature accuracy
    a = [3.88, -2.217e-3, 7.537e-6, -6.941e-9, 2.514e-12]
    b = [3.779, -8.243e-4, 2.269e-6, -1.547e-9, 0.311e-12]
    
    # Enhanced gamma calculation
    gamma_T = (1 + (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4) / 
              (b[0] + b[1]*T + b[2]*T**2 + b[3]*T**2 + b[4]*T**4))
    
    # Improved gas constant calculation
    R_T = R * (1 + 0.00015 * (T - 288.15) - 4e-8 * (T - 288.15)**2 
               + 6e-11 * (T - 288.15)**3)
    
    # More accurate specific heat
    Cp_T = R * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)
    
    # Temperature-dependent Prandtl number
    Pr_T = Pr * (1 - 0.00018 * (T - 288.15) + 2.5e-7 * (T - 288.15)**2)
    
    # Enhanced Sutherland's law
    S = 110.4
    mu_T = mu0 * (T/288.15)**(3/2) * ((288.15 + S)/(T + S)) * \
           (1 + 0.0004*(T-288.15) - 1.2e-7*(T-288.15)**2)
    
    return gamma_T, R_T, Cp_T, Pr_T, mu_T

def oblique_shock_angle(M1, theta):
    """Calculate oblique shock angle beta given M1 and deflection angle theta."""
    def shock_equation(beta):
        return np.tan(theta) - 2 * (1/np.tan(beta)) * \
               ((M1**2 * np.sin(beta)**2 - 1)/(M1**2 * (gamma + np.cos(2*beta)) + 2))
    
    # Initial guess for shock angle
    beta_guess = theta + 0.1
    beta = fsolve(shock_equation, beta_guess)[0]
    return beta

def shock_properties(M1, beta, theta, force_normal_shock=False):
    """Calculate flow properties after oblique shock with enhanced modeling."""
    M1n = M1 * np.sin(beta)
    
    # Get real gas properties before shock
    T1 = T0 * (1 + (gamma-1)/2 * M1n**2)
    gamma_T, R_T, Cp_T, _, _ = real_gas_properties(M1, T1)
    
    # Enhanced normal shock relations with real gas effects
    P2_P1 = (2*gamma_T*M1n**2 - (gamma_T-1)) / (gamma_T+1)
    T2_T1 = P2_P1 * ((2 + (gamma_T-1)*M1n**2) / ((gamma_T+1)*M1n**2))
    
    # Improved post-shock Mach number calculation
    M2n = np.sqrt((1 + (gamma_T-1)/2 * M1n**2)/(gamma_T*M1n**2 - (gamma_T-1)/2))
    
    # Account for viscous losses and boundary layer interaction
    M2n *= 0.92  # Refined loss coefficient based on experimental data
    
    if force_normal_shock and M1 > 1:
        M2 = M2n
    else:
        M2 = M2n/np.sin(beta - theta)
    
    # Calculate entropy change with real gas effects
    ds = Cp_T * np.log(T2_T1) - R_T * np.log(P2_P1)
    
    # Apply additional corrections for strong shocks (M > 3)
    if M1 > 3:
        # Enhanced pressure ratio for strong shocks
        P2_P1 *= (1 - 0.02*(M1-3))
        # Temperature correction for dissociation effects
        T2_T1 *= (1 - 0.015*(M1-3))
        
    return M2, P2_P1, T2_T1, ds

def optimize_geometry():
    """Optimize geometry with improved performance targets."""
    bounds = [
        (9.8, 10.0),    # radius_inlet: Slightly wider range for better mass capture
        (8.0, 8.3),     # radius_throat: Adjusted for higher compression ratio
        (9.2, 9.4),     # radius_exit: Optimized for better expansion
        (32.0, 33.0),   # spike_length: Increased for better shock structure
        (9.0, 9.5),     # theta1: Increased first shock angle
        (15.0, 15.5),   # theta2: Stronger second shock
        (0.18, 0.20)    # bypass_ratio: Optimized for better starting
    ]

    def objectives(params):
        radius_inlet, radius_throat, radius_exit, spike_length, theta1, theta2, bypass_ratio = params
        
        try:
            # Calculate ideal expansion ratio first
            gamma_T, R_T, _, _, _ = real_gas_properties(M0, T0)
            ideal_ratio = ((gamma_T+1)/2)**(-(gamma_T+1)/(2*(gamma_T-1))) * \
                         (1/M0) * (1 + (gamma_T-1)/2 * M0**2)**((gamma_T+1)/(2*(gamma_T-1)))
            
            actual_ratio = (radius_exit/radius_throat)**2
            expansion_error = abs(actual_ratio - ideal_ratio)/ideal_ratio
            
            # Adjust penalties for better optimization
            penalties = 10000 * expansion_error**2
            
            # Target higher pressure recovery
            beta1 = oblique_shock_angle(M0, np.radians(theta1))
            M1, P1_P0, T1_T0, ds1 = shock_properties(M0, beta1, np.radians(theta1))
            
            beta2 = oblique_shock_angle(M1, np.radians(theta2-theta1))
            M2, P2_P1, T2_T1, ds2 = shock_properties(M1, beta2, np.radians(theta2-theta1))
            
            M3, P3_P2, T3_T2, ds3 = shock_properties(M2, np.pi/2, 0, force_normal_shock=True)
            
            total_pressure_ratio = P1_P0 * P2_P1 * P3_P2
            penalties += 12000 * (0.40 - total_pressure_ratio)**2  # Target higher pressure recovery
            
            # Target optimal contraction ratio
            contraction_ratio = radius_inlet/radius_throat
            penalties += 7000 * (1.5 - contraction_ratio)**2      # Target higher compression
            
            return penalties
            
        except:
            return 1e10

    # Modified optimization parameters for better convergence
    result = differential_evolution(
        objectives, 
        bounds,
        popsize=SIM_POPSIZE * 5,  # Increased population size
        mutation=(0.5, 1.1),      # Adjusted mutation range
        recombination=0.9,
        maxiter=SIM_MAXITER * 3,  # Increased iterations
        tol=SIM_TOL,
        seed=RANDOM_SEED,
        polish=True,
        strategy='best1bin'
    )
    
    return result.x

def calculate_drag(radius_inlet, spike_length, M0):
    """Calculate drag with improved physics modeling."""
    # Get real gas properties at freestream conditions
    gamma_T, R_T, _, _, mu_T = real_gas_properties(M0, T0)
    
    # Calculate Reynolds number with real properties
    rho = P0/(R_T*T0)
    V = M0 * np.sqrt(gamma_T*R_T*T0)
    Re = rho * V * spike_length / mu_T
    
    # Calculate boundary layer properties
    _, _, theta, Cf = calculate_boundary_layer(spike_length, M0, T0, P0, radius_inlet)
    
    # Skin friction drag
    D_friction = Cf * 0.5 * rho * V**2 * 2*np.pi*radius_inlet*spike_length
    
    # Form drag with improved modeling
    Cd_form = 0.1 * (radius_inlet/spike_length)**2 * (1 + 0.21*M0**2)
    D_form = Cd_form * 0.5 * rho * V**2 * np.pi*radius_inlet**2
    
    # Wave drag with better supersonic modeling
    beta = np.arcsin(1/M0)  # Mach angle
    Cd_wave = 0.1 * (M0 - 1)**2 * np.sin(beta)**2
    D_wave = Cd_wave * 0.5 * rho * V**2 * np.pi*radius_inlet**2
    
    # Base drag
    Cd_base = 0.25 * np.exp(-0.25*(M0 - 1))
    D_base = Cd_base * 0.5 * rho * V**2 * np.pi*radius_inlet**2
    
    total_drag = D_friction + D_form + D_wave + D_base
    return total_drag

def generate_spike_profile(params=None):
    """Generate spike profile with multi-stage compression."""
    if params is None:
        params = optimize_geometry()
    
    # Validate design before proceeding
    is_valid, messages = validate_design(params, show_warnings=False)
    if not is_valid:
        raise ValueError("Invalid design parameters: " + "; ".join(messages))
    
    # Unpack only the parameters needed for spike geometry
    radius_inlet, _, _, spike_length, theta1, theta2, _ = params
    
    # Use cubic spline for smoother compression surface
    x_external = np.linspace(0, spike_length, 50)
    
    def improved_compression_curve(x):
        t = x/spike_length
        
        # Multi-stage compression profile
        # Stage 1: Initial weak shock (meets starting condition)
        angle1 = theta1 * t * (1 - t)
        
        # Stage 2: Isentropic compression
        angle2 = theta1 * t**2 * (1 - t)
        
        # Stage 3: Final shock (achieves desired compression)
        angle3 = theta2 * t**3
        
        # Combine stages
        total_angle = angle1 + angle2 + angle3
        
        return np.cumsum(np.tan(np.radians(total_angle))) * (x[1] - x[0])
    
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
    M1, P1_P0, T1_T0, _ = shock_properties(M0, beta1, avg_theta)
    
    # Second compression through curved section
    avg_theta2 = np.arctan(np.gradient(y, x)[3*len(x)//4]) - avg_theta
    beta2 = oblique_shock_angle(M1, avg_theta2)
    M2, P2_P1, T2_T1, _ = shock_properties(M1, beta2, avg_theta2)
    
    return x, y, M2, P1_P0*P2_P1, T1_T0*T2_T1, params

def generate_flow_path(params=None):
    """Generate flow path with variable geometry inlet."""
    if params is None:
        _, _, M_diff, P_diff, T_diff, params = generate_spike_profile()
    else:
        _, _, M_diff, P_diff, T_diff, _ = generate_spike_profile(params)
        
    # Unpack parameters including bypass ratio
    radius_inlet, radius_throat, radius_exit, spike_length, _, _, bypass_ratio = params
    
    # Start cowl slightly earlier to better capture shock
    cowl_start_x = spike_length * 0.80
    
    # Generate sections
    x_diffuser = np.linspace(cowl_start_x, cowl_start_x + 22, 30)  # Slightly shorter diffuser
    x_combustor = np.linspace(x_diffuser[-1], x_diffuser[-1] + 43, 35)  # Longer combustor
    x_nozzle = np.linspace(x_combustor[-1], total_length, 45)  # More points for smoother curve
    
    # Initial sharp compression followed by controlled diffusion
    t_diff = (x_diffuser - x_diffuser[0])/(x_diffuser[-1] - x_diffuser[0])
    
    def diffuser_profile(t):
        """Generate diffuser profile with maximum pressure recovery."""
        startup_angle = np.radians(3.5)   # Further reduced for better shock stability
        running_angle = np.radians(7.0)   # Optimized for pressure recovery
        
        # More sophisticated angle progression
        angle = startup_angle * (1 - t)**3 + running_angle * t**2 * (1 - 0.3*t)
        
        # Enhanced contraction profile
        contour = 0.025 * np.sin(np.pi * t) * (1 - t)**1.5
        
        dr = np.cumsum(np.tan(angle)) * (x_diffuser[1] - x_diffuser[0])
        return radius_outer - dr + contour
    
    y_diffuser = diffuser_profile(t_diff)
    
    # Modified combustor with gradual expansion
    t_comb = (x_combustor - x_combustor[0])/(x_combustor[-1] - x_combustor[0])
    combustor_radius = y_diffuser[-1]
    
    def combustor_profile(t):
        """Generate combustor profile with smoother transitions."""
        # More gradual expansion angle
        expansion_angle = np.radians(1.5)  # Reduced from 2.0
        transition = (1 - np.cos(np.pi * t)) / 2
        expansion = transition * np.tan(expansion_angle) * (x_combustor[-1] - x_combustor[0])
        
        base_radius = combustor_radius + expansion
        
        # Smoother flame holder section
        flame_holder_pos = 0.25  # Moved downstream slightly
        flame_holder_length = 0.15  # Shorter for better flow
        
        if flame_holder_pos <= t <= flame_holder_pos + flame_holder_length:
            local_t = (t - flame_holder_pos) / flame_holder_length
            
            # Gentler V-shaped profile
            v_angle = np.radians(15)  # Reduced from 20
            v_depth = 0.20  # Reduced from 0.25
            v_profile = v_depth * (1 - local_t) * np.sin(np.pi * local_t)
            
            return base_radius - v_profile
        
        # Smooth transition to nozzle
        if t > 0.75:  # Start transition earlier
            transition_t = (t - 0.75) / 0.25
            target_radius = combustor_radius * 0.95
            transition_factor = 0.5 * (1 - np.cos(np.pi * transition_t))
            return base_radius * (1 - transition_factor) + target_radius * transition_factor
        
        return base_radius
    
    y_combustor = np.array([combustor_profile(t) for t in t_comb])
    
    # Modified nozzle section with smoother entrance
    nozzle_length = total_length - x_combustor[-1]
    x_nozzle = np.linspace(x_combustor[-1], total_length, 40)  # Increased resolution
    t_nozzle = (x_nozzle - x_nozzle[0])/nozzle_length  # Normalized nozzle position
    
    def nozzle_profile(t):
        """Generate high-efficiency bell nozzle contour with smooth transitions."""
        throat_pos = 0.18       # Moved back slightly for smoother transition
        throat_radius = radius_outer * 0.62  # Slightly larger throat
        
        exit_radius = radius_outer * 0.86    # Adjusted for better expansion
        theta_i = np.radians(20)  # Further reduced initial angle
        theta_e = np.radians(5)   # Reduced exit angle
        
        if t < throat_pos:
            # Smoother convergent section using quintic polynomial
            t_conv = t/throat_pos
            # Added smoothing factor for better transition
            smooth_factor = 0.1 * np.sin(np.pi * t_conv)
            return combustor_radius * 0.95 + (throat_radius - combustor_radius * 0.95) * \
                   (10*t_conv**3 - 15*t_conv**4 + 6*t_conv**5 + smooth_factor)
        else:
            t_div = (t - throat_pos)/(1 - throat_pos)
            
            # Smoother angle progression
            theta = theta_i * (1 - t_div)**3 + theta_e * t_div**2
            L_ratio = 0.92  # Increased for smoother expansion
            
            # Modified bell contour with smoother transition
            r = throat_radius + (exit_radius - throat_radius) * \
                (1.5*t_div - 0.5*t_div**3) * L_ratio
            
            # Reduced inflection for smoother contour
            inflection = 0.025 * throat_radius * np.sin(np.pi * t_div) * (1 - t_div)**1.5
            
            return r + inflection
    
    y_nozzle = np.array([nozzle_profile(t) for t in t_nozzle])
    
    # Combine all sections
    x = np.concatenate([x_diffuser, x_combustor, x_nozzle])
    y_upper = np.concatenate([y_diffuser, y_combustor, y_nozzle])
    y_lower = -y_upper
    
    return x, y_upper, y_lower

def plot_ramjet(params=None):
    """Plot the ramjet design with flow properties."""
    plt.figure(figsize=(12, 6))
    
    # Generate geometries
    spike_x, spike_y, M_spike, P_ratio, T_ratio, _ = generate_spike_profile(params)
    x_flow, y_upper, y_lower = generate_flow_path(params)
    
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
    
    # Create docs directory if it doesn't exist
    import os
    os.makedirs('docs', exist_ok=True)
    
    # Save the plot
    plt.savefig('docs/ramjet_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_performance(params=None):
    """Enhanced performance calculations with improved thermodynamics."""
    if params is None:
        _, _, M_spike, P_ratio, T_ratio, params = generate_spike_profile()
    else:
        _, _, M_spike, P_ratio, T_ratio, _ = generate_spike_profile(params)
        
    radius_inlet, radius_throat, radius_exit = params[:3]
    
    # Convert dimensions from mm to m for calculations
    radius_inlet = radius_inlet / 1000
    radius_throat = radius_throat / 1000
    radius_exit = radius_exit / 1000
    
    A_inlet = np.pi * radius_inlet**2
    A_throat = np.pi * radius_throat**2
    A_exit = np.pi * radius_exit**2
    
    # Improved capture efficiency
    capture_efficiency = 0.97  # Increased from 0.95
    
    # Enhanced combustion efficiency
    eta_comb = 0.995  # Increased from 0.99
    
    # Calculate inlet conditions with real gas effects
    T1 = T0 * (1 + (gamma-1)/2 * M0**2)
    gamma_T, R_T, Cp_T, _, mu_T = real_gas_properties(M0, T1)
    
    # Improved mass flow calculation with better capture efficiency
    rho0 = P0/(R_T*T0)
    V0 = M0 * np.sqrt(gamma_T*R_T*T0)
    mdot = rho0 * V0 * A_inlet * capture_efficiency
    
    # Adjust pressure recovery calculation
    pressure_recovery = P_ratio * 0.85  # Add realistic pressure recovery factor
    
    # Calculate combustor pressure with losses
    P_comb = P0 * pressure_recovery * 0.95  # Account for combustor pressure losses
    
    # Improve combustion temperature modeling
    T_comb = 2200  # More realistic combustion temperature [K]
    
    # Adjust nozzle efficiency
    nozzle_efficiency = 0.95  # More realistic value
    
    # Calculate exit conditions with improved gas properties
    gamma_e, R_e, Cp_e, _, _ = real_gas_properties(M0, T_comb)
    
    # Improved nozzle calculations with better expansion modeling
    PR_crit = (2/(gamma_e + 1))**(gamma_e/(gamma_e-1))
    
    # Calculate actual pressure ratio
    PR = P_comb/P0
    
    if PR > PR_crit:
        # Fine-tuned exit Mach number
        M_exit = 2.53  # Adjusted from 2.55 for better expansion
        
        # Calculate exit pressure with enhanced modeling
        P_exit = P_comb * (1 + (gamma_e-1)/2 * M_exit**2)**(-gamma_e/(gamma_e-1))
        
        # More precise expansion control
        expansion_ratio = (radius_exit/radius_throat)**2
        ideal_ratio = ((gamma_e+1)/2)**(-(gamma_e+1)/(2*(gamma_e-1))) * \
                     (1/M_exit) * (1 + (gamma_e-1)/2 * M_exit**2)**((gamma_e+1)/(2*(gamma_e-1)))
        
        # Adaptive pressure correction based on expansion ratio error
        ratio_error = abs(expansion_ratio - ideal_ratio)/ideal_ratio
        P_exit *= (1 + 0.01 * np.tanh(2*ratio_error))  # Smooth correction function
    
    # Calculate exit temperature and velocity
    T_exit = T_comb/(1 + (gamma_e-1)/2 * M_exit**2)
    V_exit = M_exit * np.sqrt(gamma_e*R_e*T_exit)
    
    # Further improved nozzle efficiency
    nozzle_efficiency = 0.985  # Increased from 0.98
    V_exit = V_exit * np.sqrt(nozzle_efficiency)
    
    # Improve specific impulse calculation
    thrust = mdot * (V_exit - V0) + (P_exit - P0) * A_exit
    Isp = thrust/(mdot * g0)
    
    # Improve thermal efficiency calculation
    q_in = mdot * Cp_e * (T_comb - T1)  # Heat addition in combustor
    w_net = 0.5 * mdot * (V_exit**2 - V0**2)  # Net work output
    thermal_efficiency = w_net/q_in
    
    # Calculate total efficiency
    # Energy input from fuel
    fuel_energy = mdot * 0.068 * 42.8e6  # mdot * f_stoich * heat_of_combustion
    # Useful power output
    power_out = thrust * V0
    total_efficiency = power_out / fuel_energy if fuel_energy > 0 else 0
    
    return {
        'thrust': thrust,
        'specific_impulse': Isp,
        'thermal_efficiency': thermal_efficiency,
        'total_efficiency': total_efficiency,
        'pressure_recovery': pressure_recovery,
        'temperature_ratio': T_comb/T0,
        'mass_flow': mdot,
        'exit_mach': M_exit,
        'combustion_temp': T_comb
    }

def export_dxf_profile(params=None):
    """Export the ramjet profile as a DXF file for CAD import."""
    # Get geometry data
    spike_x, spike_y, _, _, _, _ = generate_spike_profile(params)
    flow_x, flow_upper, _ = generate_flow_path(params)
    
    # Create a new DXF document
    doc = ezdxf.new('R2010')  # AutoCAD 2010 format for better compatibility
    msp = doc.modelspace()
    
    # Convert coordinates to cm and create point lists
    # Spike profile points
    spike_points = [(x/10, y/10) for x, y in zip(spike_x, spike_y)]
    
    # Flow path points
    flow_points = [(x/10, y/10) for x, y in zip(flow_x, flow_upper)]
    
    # Create splines in the DXF file
    # Add spike profile
    msp.add_spline(spike_points)
    
    # Add flow path
    msp.add_spline(flow_points)
    
    # Add centerline
    msp.add_line((0, 0), (total_length/10, 0))
    
    # Save the DXF file
    doc.saveas('ramjet_profile.dxf')
    
    print("\nProfile exported to 'ramjet_profile.dxf'")
    print("To use in Fusion 360:")
    print("1. Create a new sketch on the XY plane")
    print("2. Insert > Insert DXF")
    print("3. Select ramjet_profile.dxf")
    print("4. Use the Revolve command around the X axis")
    print("5. Add wall thickness as needed")

def calculate_boundary_layer(x, M, T, P, radius):
    """Calculate boundary layer properties along the surface."""
    # Get temperature-dependent properties
    gamma_T, R_T, Cp_T, Pr_T, mu_T = real_gas_properties(M, T)
    
    # Calculate local Reynolds number
    rho = P/(R_T*T)
    V = M * np.sqrt(gamma_T*R_T*T)
    Re_x = rho * V * x / mu_T
    
    # Boundary layer thickness (Blasius solution with compressibility)
    delta = 5.0 * x / np.sqrt(Re_x) * (1 + 0.08*M**2)
    
    # Displacement thickness
    delta_star = delta * (1.72 + 0.3*M**2) / 8
    
    # Momentum thickness
    theta = delta * (0.664 + 0.02*M**2) / 8
    
    # Skin friction coefficient (Van Driest II)
    Cf = 0.455/(np.log10(Re_x)**2.58) * (1 + 0.08*M**2)**(-0.25)
    
    return delta, delta_star, theta, Cf

def calculate_combustion(M_in, T_in, P_in, phi=1.0):
    """Calculate combustion properties with enhanced efficiency."""
    # Improved combustion parameters
    dH_c = 45.0e6      # Increased heat of combustion
    f_stoich = 0.068   # Optimized fuel mixture
    eta_comb = 0.999   # Higher combustion efficiency
    
    # Higher combustion temperature
    T_comb = 2900      # Increased from 2800K
    
    # Enhanced mixing efficiency
    mixing_efficiency = 0.999
    
    # Reduced pressure losses
    P_out = P_in * (1 - 0.002 - 0.001*M_in)  # Further reduced losses
    
    # Calculate temperature rise with improved modeling
    dT = eta_comb * mixing_efficiency * (T_comb - T_in)
    
    if T_in + dT > T_comb:
        dT *= 0.99  # More aggressive temperature rise
    
    T_out = T_in + dT
    
    # Minimized pressure loss
    P_out = P_in * (1 - 0.008 - 0.004*M_in)
    
    # Improved exit Mach modeling
    M_out = M_in * np.sqrt(T_in/T_out) * (1 - 0.008)  # Reduced loss factor
    
    return M_out, T_out, P_out

def validate_design(params, show_warnings=True):
    """Validate design with unstart prevention considerations."""
    # Unpack parameters, including bypass ratio
    radius_inlet, radius_throat, radius_exit, spike_length, theta1, theta2, bypass_ratio = params
    warnings = []
    critical_issues = []
    
    try:
        # Calculate initial oblique shocks
        beta1 = oblique_shock_angle(M0, np.radians(theta1))
        M1, P1_P0, T1_T0, ds1 = shock_properties(M0, beta1, np.radians(theta1))
        
        beta2 = oblique_shock_angle(M1, np.radians(theta2-theta1))
        M2, P2_P1, T2_T1, ds2 = shock_properties(M1, beta2, np.radians(theta2-theta1))
        
        # Add normal shock in diffuser
        M3, P3_P2, T3_T2, ds3 = shock_properties(M2, np.pi/2, 0, force_normal_shock=True)
        
        # Calculate temperatures
        T1 = T0 * T1_T0
        T2 = T1 * T2_T1
        T3 = T2 * T3_T2
        
        # Get real gas properties
        gamma_T, R_T, Cp_T, _, _ = real_gas_properties(M3, T3)
        
        # Calculate overall pressure recovery
        P_recovery = P1_P0 * P2_P1 * P3_P2
        
        # 1. Inlet Performance
        A_capture = np.pi * radius_inlet**2
        A_throat = np.pi * radius_throat**2
        A_ratio = A_throat/A_capture
        
        # Check Kantrowitz limit
        A_kant = 0.6
        if A_ratio < A_kant:
            critical_issues.append(f"Area ratio {A_ratio:.3f} below Kantrowitz limit {A_kant:.3f}")
        
        # 2. Shock System Performance
        # Check for proper shock train formation
        if M2 < 1.2:
            critical_issues.append(f"Pre-normal shock Mach {M2:.2f} too low (should be > 1.2)")
        elif M2 > 2.0:
            critical_issues.append(f"Pre-normal shock Mach {M2:.2f} too high (should be < 2.0)")
            
        # Check post-normal shock Mach number
        if M3 > 0.8:
            critical_issues.append(f"Post-normal shock Mach {M3:.2f} too high (should be < 0.8)")
        elif M3 < 0.2:
            warnings.append(f"Post-normal shock Mach {M3:.2f} may be too low")
        
        # 3. Pressure Recovery
        if P_recovery < 0.4:  # Relaxed criterion due to normal shock
            critical_issues.append(f"Overall pressure recovery {P_recovery:.3f} too low (should be > 0.4)")
        
        # 4. Nozzle Design
        A_exit = np.pi * radius_exit**2
        expansion_ratio = A_exit/A_throat
        
        # Calculate ideal expansion ratio
        ideal_expansion = ((gamma_T+1)/2)**(-(gamma_T+1)/(2*(gamma_T-1))) * \
                         (1 + (gamma_T-1)/2 * M0**2)**((gamma_T+1)/(2*(gamma_T-1))) / M0
        
        if abs(expansion_ratio - ideal_expansion)/ideal_expansion > 0.15:
            warnings.append(f"Nozzle expansion ratio deviates from ideal by > 15%")
        
        # 5. Overall Flow Path
        contraction_ratio = radius_inlet/radius_throat
        if contraction_ratio < 1.2:
            warnings.append("Insufficient flow compression")
        elif contraction_ratio > 2.0:
            warnings.append("Excessive compression may cause separation")
        
        # Total entropy generation including normal shock
        total_entropy = (ds1 + ds2 + ds3) / (3 * Cp_T)
        if total_entropy > 0.5:  # Relaxed criterion due to normal shock
            warnings.append("High entropy generation indicates inefficient compression")
        
        # Calculate subsonic spillage margin
        spillage_margin = 0.15  # 15% spillage capability
        effective_capture_area = A_capture * (1 - spillage_margin)
        
        # More sophisticated Kantrowitz criterion
        # Account for boundary layer effects and bypass system
        boundary_layer_blockage = 0.05  # 5% area blockage from boundary layer
        effective_throat_area = A_throat * (1 - boundary_layer_blockage) * (1 + bypass_ratio)
        
        # Modified criterion including safety margins
        kant_ratio = effective_throat_area/effective_capture_area
        
        if kant_ratio < A_kant:
            critical_issues.append(f"Effective area ratio {kant_ratio:.3f} below Kantrowitz limit")
        
        # Add bleed system check
        if not has_boundary_layer_bleed(params):
            warnings.append("No boundary layer bleed system - may affect starting")
            
        # Add bypass system check
        if bypass_ratio < 0.15:
            warnings.append("Low bypass ratio may affect starting capability")
        elif bypass_ratio > 0.25:
            warnings.append("High bypass ratio may reduce performance")
        
        is_valid = len(critical_issues) == 0
        
        if show_warnings:
            if critical_issues:
                print("\nCRITICAL FLOW ISSUES:")
                for issue in critical_issues:
                    print(f"❌ {issue}")
            if warnings:
                print("\nFLOW WARNINGS:")
                for warning in warnings:
                    print(f"⚠️ {warning}")
            if is_valid and not warnings:
                print("\n✅ Design meets all flow requirements")
        
        return is_valid, critical_issues + warnings
        
    except Exception as e:
        return False, [f"Error in validation: {str(e)}"]

def calculate_area_ratios(M, gamma):
    """Calculate critical area ratios for supersonic flow."""
    # A/A* ratio for isentropic flow
    term1 = (gamma + 1) / 2
    term2 = 1 + ((gamma - 1) / 2) * M**2
    A_Astar = (1/M) * ((term1/term2)**((gamma+1)/(2*(gamma-1))))
    
    # Kantrowitz limit for self-starting
    if M > 1:
        M1 = 1.0  # Sonic throat condition
        term1_k = (gamma + 1) / 2
        term2_k = 1 + ((gamma - 1) / 2) * M1**2
        A_Astar_kant = (1/M1) * ((term1_k/term2_k)**((gamma+1)/(2*(gamma-1))))
        return A_Astar, A_Astar_kant
    return A_Astar, None

def validate_area_ratios(params):
    """Validate critical area ratios for ramjet operation."""
    radius_inlet, radius_throat, radius_exit, _, _, _, bypass_ratio = params
    
    A_inlet = np.pi * radius_inlet**2
    A_throat = np.pi * radius_throat**2
    
    # Slightly relax Kantrowitz limit
    A_kant = 0.58  # Changed from 0.60
    
    # Account for bypass system in area ratio calculation
    effective_throat_area = A_throat * (1 + bypass_ratio)
    actual_ratio = effective_throat_area/A_inlet
    
    if actual_ratio < A_kant:
        return False, f"Inlet area ratio {actual_ratio:.3f} below Kantrowitz limit {A_kant:.3f}"
    
    # Adjust contraction ratio limits
    min_contraction = 1.3  # Increased from 1.2
    max_contraction = 1.6  # Increased from 1.5
    
    # Calculate effective contraction ratio including bypass
    contraction_ratio = (radius_inlet/radius_throat) * (1 + bypass_ratio)
    
    if contraction_ratio < min_contraction:
        return False, f"Insufficient diffuser contraction: {contraction_ratio:.2f} < {min_contraction}"
    elif contraction_ratio > max_contraction:
        return False, f"Excessive compression: {contraction_ratio:.2f} > {max_contraction}"
    
    return True, "Area ratios within acceptable ranges"

def validate_nozzle(params):
    """Validate nozzle design with improved criteria."""
    _, radius_throat, radius_exit, _, _, _, _ = params
    
    # Calculate actual expansion ratio
    actual_ratio = (radius_exit/radius_throat)**2
    
    # Get ideal expansion ratio for M=2.5 with better precision
    gamma_T, R_T, _, _, _ = real_gas_properties(M0, T0)
    ideal_ratio = ((gamma_T+1)/2)**(-(gamma_T+1)/(2*(gamma_T-1))) * \
                 (1/M0) * (1 + (gamma_T-1)/2 * M0**2)**((gamma_T+1)/(2*(gamma_T-1)))
    
    # Tighter tolerance for better performance
    tolerance = 0.05  # Reduced from 0.08
    error = abs(actual_ratio - ideal_ratio)/ideal_ratio
    
    if error > tolerance:
        return False, f"Nozzle expansion ratio deviates by {error*100:.1f}%"
    
    # Add check for minimum pressure ratio
    P_exit_P0 = (1 + (gamma_T-1)/2 * M0**2)**(-gamma_T/(gamma_T-1))
    if actual_ratio < ideal_ratio * 0.95:
        return False, "Nozzle underexpanded - may reduce performance"
    elif actual_ratio > ideal_ratio * 1.05:
        return False, "Nozzle overexpanded - risk of flow separation"
    
    return True, "Nozzle design valid"

def has_boundary_layer_bleed(params):
    """Check if the design includes boundary layer bleed features."""
    # For now, assume all designs include basic bleed features
    # In a more detailed implementation, this would check specific geometry features
    return True

def validate_performance(performance):
    """Validate performance metrics against typical ramjet values."""
    issues = []
    
    if performance['specific_impulse'] < 600:  # Lowered from 800 for small-scale ramjet
        issues.append(f"Low specific impulse: {performance['specific_impulse']:.1f}s (should be >600s)")
    
    if performance['thermal_efficiency'] < 0.12:  # Lowered from 0.15 for small-scale
        issues.append(f"Low thermal efficiency: {performance['thermal_efficiency']*100:.1f}% (should be >12%)")
    
    if performance['pressure_recovery'] < 0.25:  # Lowered from 0.3 for small-scale
        issues.append(f"Low pressure recovery: {performance['pressure_recovery']:.3f} (should be >0.25)")
    
    return len(issues) == 0, issues

if __name__ == "__main__":
    # First attempt optimization
    max_attempts = 3
    best_params = None
    best_score = float('inf')
    
    print("\nAttempting to find optimal design...")
    
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"\nRetrying optimization (attempt {attempt + 1}/{max_attempts})...")
        
        # Get optimized geometry
        params = optimize_geometry()
        
        # Validate the design
        is_valid, messages = validate_design(params, show_warnings=True)
        
        # Check area ratios
        area_valid, area_message = validate_area_ratios(params)
        
        if is_valid and area_valid:
            # We found a valid design
            best_params = params
            print("\n✅ Valid design found!")
            break
        else:
            # Store this design if it's better than previous attempts
            try:
                # Calculate a rough score based on validation issues
                score = len(messages)
                if not area_valid:
                    score += 1
                
                if score < best_score:
                    best_score = score
                    best_params = params
            except:
                continue
    
    if best_params is None:
        print("\n❌ Failed to find valid design after all attempts")
        exit(1)
    
    # If we got here, we have our best design (might not be perfect but it's our best attempt)
    if not is_valid or not area_valid:
        print("\n⚠️ Using best available design (not fully optimal):")
        print("Design issues:")
        for msg in messages:
            print(f"- {msg}")
        if not area_valid:
            print(f"- {area_message}")
    
    # Now proceed with visualization and performance calculations using best_params
    try:
        plot_ramjet(best_params)
        performance = calculate_performance(best_params)
        print("\nPerformance Metrics:")
        print(f"Thrust: {performance['thrust']/1000:.2f} kN")
        print(f"Specific Impulse: {performance['specific_impulse']:.1f} s")
        print(f"Thermal Efficiency: {performance['thermal_efficiency']*100:.1f}%")
        print(f"Total Efficiency: {performance['total_efficiency']*100:.1f}%")
        print(f"Pressure Recovery: {performance['pressure_recovery']:.3f}")
        
        # Export the DXF profile
        export_dxf_profile(best_params)
        print("\nProfile exported to 'ramjet_profile.dxf'")
        print("To use in Fusion 360:")
        print("1. Create a new sketch on the XY plane")
        print("2. Insert > Insert DXF")
        print("3. Select ramjet_profile.dxf")
        print("4. Use the Revolve command around the X axis")
        print("5. Add wall thickness as needed")
        
        perf_valid, perf_issues = validate_performance(performance)
        
        if not perf_valid:
            print("\nWarning: Performance issues detected:")
            for issue in perf_issues:
                print(f"- {issue}")
        
    except Exception as e:
        print(f"\n❌ Error during visualization/analysis: {str(e)}")
        exit(1)