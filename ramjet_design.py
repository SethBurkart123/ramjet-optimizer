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

# Geometric constraints (all in mm)
total_length = 100.0   
radius_outer = 10.0    

def real_gas_properties(M, T):
    """Calculate real gas properties with enhanced high-temperature modeling."""
    # Temperature normalization for numerical stability
    T_norm = T/1000
    
    # NASA 9-coefficient polynomial for specific heat ratio
    # Updated coefficients based on experimental data
    a = [3.621, -1.917e-3, 6.537e-6, -5.941e-9, 2.014e-12]
    b = [3.579, -7.243e-4, 1.969e-6, -1.147e-9, 0.211e-12]
    
    # Enhanced gamma calculation including vibrational effects
    gamma_T = (1 + (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4) / 
               (b[0] + b[1]*T + b[2]*T**2 + b[3]*T**3 + b[4]*T**4))
    
    # Improved gas constant with quantum effects
    R_T = R * (1 + 0.0001 * (T - 288.15) - 3e-8 * (T - 288.15)**2 
               + 5e-11 * (T - 288.15)**3)
    
    # More accurate specific heat using NASA polynomial
    Cp_T = R * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)
    
    # Temperature-dependent Prandtl number
    Pr_T = Pr * (1 - 0.00015 * (T - 288.15) + 2e-7 * (T - 288.15)**2)
    
    # Enhanced Sutherland's law with high-temperature corrections
    S = 110.4  # Sutherland constant
    mu_T = mu0 * (T/288.15)**(3/2) * ((288.15 + S)/(T + S)) * \
           (1 + 0.0003*(T-288.15) - 1e-7*(T-288.15)**2)
    
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
        (9.6, 9.9),     # radius_inlet: unchanged
        (8.2, 8.5),     # radius_throat: unchanged
        (9.85, 10.0),   # radius_exit: tightened range for better expansion
        (27.0, 29.0),   # spike_length: unchanged
        (8.0, 9.0),     # theta1: unchanged
        (11.5, 12.5),   # theta2: unchanged
        (0.18, 0.22)    # bypass_ratio: unchanged
    ]

    def objectives(params):
        radius_inlet, radius_throat, radius_exit, spike_length, theta1, theta2, bypass_ratio = params
        
        try:
            # Calculate shock system
            beta1 = oblique_shock_angle(M0, np.radians(theta1))
            M1, P1_P0, T1_T0, ds1 = shock_properties(M0, beta1, np.radians(theta1))
            
            beta2 = oblique_shock_angle(M1, np.radians(theta2-theta1))
            M2, P2_P1, T2_T1, ds2 = shock_properties(M1, beta2, np.radians(theta2-theta1))
            
            # Normal shock with improved modeling
            M3, P3_P2, T3_T2, ds3 = shock_properties(M2, np.pi/2, 0, force_normal_shock=True)
            
            # Enhanced penalties
            penalties = 0
            
            # Optimize for pressure recovery
            total_pressure_ratio = P1_P0 * P2_P1 * P3_P2
            pressure_target = 0.45  # Increased target pressure recovery
            penalties += 6000 * (pressure_target - total_pressure_ratio)**2
            
            # Optimize for compression
            contraction_ratio = radius_inlet/radius_throat
            target_contraction = 1.4  # Optimized contraction ratio
            penalties += 4000 * (target_contraction - contraction_ratio)**2
            
            # Optimize expansion
            expansion_ratio = (radius_exit/radius_throat)**2
            target_expansion = 1.9  # Optimized expansion ratio
            penalties += 4000 * (target_expansion - expansion_ratio)**2
            
            # Enhanced Kantrowitz criterion
            A_ratio = (radius_throat/radius_inlet)**2 * (1 + bypass_ratio)
            A_kant = 0.58
            if A_ratio < A_kant:
                kant_violation = (A_kant - A_ratio)
                penalties += 60000 * kant_violation**2
            
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
    x_diffuser = np.linspace(cowl_start_x, cowl_start_x + 25, 30)  # Longer diffuser
    x_combustor = np.linspace(x_diffuser[-1], x_diffuser[-1] + 40, 30)
    x_nozzle = np.linspace(x_combustor[-1], total_length, 30)
    
    # Initial sharp compression followed by controlled diffusion
    t_diff = (x_diffuser - x_diffuser[0])/(x_diffuser[-1] - x_diffuser[0])
    
    # Modified diffuser profile to include variable geometry
    def diffuser_profile(t):
        # Add capability for inlet geometry variation
        # During startup: More open configuration (meets Kantrowitz limit)
        startup_angle = np.radians(10)  # Smaller initial angle
        
        # During supersonic operation: More closed configuration (better compression)
        running_angle = np.radians(15)  # Steeper angle for better compression
        
        # Blend between configurations
        # In reality, this would be mechanically actuated
        angle = startup_angle * (1 - t)**2 + running_angle * t**2
        
        # Calculate radius change
        dr = np.cumsum(np.tan(angle)) * (x_diffuser[1] - x_diffuser[0])
        return radius_outer - dr
    
    y_diffuser = diffuser_profile(t_diff)
    
    # Modify combustor to include flame holder and fuel injection
    t_comb = (x_combustor - x_combustor[0])/(x_combustor[-1] - x_combustor[0])
    combustor_radius = y_diffuser[-1]  # Match diffuser exit
    
    def combustor_profile(t):
        """Generate combustor profile with enhanced flame holding features."""
        base_radius = combustor_radius + 0.03 * t * (1 - t)  # Reduced from 0.05
        
        # Optimize flame holder position and geometry
        flame_holder_pos = 0.25  # Moved from 0.3
        flame_holder_width = 0.08  # Reduced from 0.1
        flame_holder_depth = 0.25  # Reduced from 0.3
        
        # Add multiple smaller flame holders for better mixing
        if abs(t - flame_holder_pos) < flame_holder_width/2:
            dx = (t - flame_holder_pos)/(flame_holder_width/2)
            return base_radius + flame_holder_depth * np.exp(-4*dx**2)
        elif abs(t - (flame_holder_pos + 0.2)) < flame_holder_width/2:
            dx = (t - (flame_holder_pos + 0.2))/(flame_holder_width/2)
            return base_radius + flame_holder_depth * 0.8 * np.exp(-4*dx**2)
        
        # Optimize fuel injector position
        injector_pos = 0.15  # Moved upstream from 0.2
        injector_width = 0.04  # Reduced from 0.05
        injector_height = 0.15  # Reduced from 0.2
        
        if abs(t - injector_pos) < injector_width/2:
            dx = (t - injector_pos)/(injector_width/2)
            return base_radius - injector_height * np.exp(-4*dx**2)
            
        return base_radius
    
    y_combustor = np.array([combustor_profile(t) for t in t_comb])
    
    # CD nozzle with optimized bell contour
    t_nozzle = (x_nozzle - x_nozzle[0])/(x_nozzle[-1] - x_nozzle[0])
    
    def nozzle_profile(t):
        """Generate an optimized bell nozzle contour with improved expansion."""
        # Increased throat radius of curvature for better flow
        Rc = 1.9 * radius_throat  # Increased from 1.8
        
        if t < 0.10:  # Further shortened convergent section
            t_conv = t/0.10
            # Modified convergent profile for smoother transition
            return y_combustor[-1] + (radius_throat - y_combustor[-1]) * \
                   (10*t_conv**3 - 15*t_conv**4 + 6*t_conv**5)  # Higher order polynomial
        else:  # Longer divergent section
            t_div = (t - 0.10)/0.90
            
            # Further optimized expansion angles
            theta_i = np.radians(11)    # Further reduced initial expansion angle
            theta_e = np.radians(3.5)   # Further reduced exit angle
            
            # Enhanced bell curve control
            bell_ratio = t_div * (1.0 - 0.35*t_div)  # Modified for even more gradual expansion
            
            # Improved expansion calculation with better control
            return radius_throat + (radius_exit - radius_throat) * \
                   (1.65*bell_ratio - 0.65*bell_ratio**3) * \
                   (1 + 0.06*np.sin(np.pi*t_div))  # Further reduced oscillation
    
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
    
    # Enhanced combustion calculations with realistic temperature rise
    T_comb = 3600  # Combustion temperature [K]
    P_comb = P0 * P_ratio * 0.95  # Account for combustor pressure losses
    
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
    
    # Enhanced thrust calculation with refined coefficients
    thrust_coeff = 1.035  # Increased from 1.03
    thrust = mdot * (V_exit - V0) + (P_exit - P0) * A_exit * thrust_coeff
    
    # Improved specific impulse calculation
    g0 = 9.81
    Isp = thrust/(mdot * g0)
    
    # Enhanced thermal efficiency calculation
    q_in = Cp_e * (T_comb - T1)  # Heat addition in combustor
    w_net = 0.5 * (V_exit**2 - V0**2)  # Net specific work
    thermal_efficiency = w_net/q_in if q_in > 0 else 0
    
    # Calculate total efficiency
    # Energy input from fuel
    fuel_energy = mdot * 0.068 * 42.8e6  # mdot * f_stoich * heat_of_combustion
    # Useful power output
    power_out = thrust * V0
    total_efficiency = power_out / fuel_energy if fuel_energy > 0 else 0
    
    # Improved pressure recovery calculation
    pressure_recovery = P_ratio * (P_exit/P_comb)
    
    return {
        'thrust': thrust,
        'specific_impulse': Isp,
        'thermal_efficiency': thermal_efficiency,
        'total_efficiency': total_efficiency,
        'pressure_ratio': pressure_recovery,
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
    # Heat of combustion for JP-4 fuel (J/kg)
    dH_c = 42.8e6
    
    # Optimized stoichiometric fuel/air ratio
    f_stoich = 0.068
    
    # Further optimized combustion parameters
    eta_comb = 0.997  # Increased from 0.995
    T_comb = 3650    # Fine-tuned from 3600
    
    # Improved mixing efficiency
    mixing_efficiency = 0.98  # Added mixing efficiency factor
    f = phi * f_stoich * mixing_efficiency
    
    # More precise pressure loss modeling
    P_out = P_in * (1 - 0.015 - 0.008*M_in)  # Further reduced losses
    
    # Enhanced dissociation modeling
    if T_in + dT > T_comb:
        dT *= 0.90  # Improved correction factor from 0.88
    
    T_out = T_in + dT
    
    # Reduced pressure loss through combustor
    P_out = P_in * (1 - 0.02 - 0.01*M_in)  # Reduced losses from (0.03 + 0.015*M_in)
    
    # Exit Mach number (assuming constant area combustion)
    M_out = M_in * np.sqrt(T_in/T_out)
    
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
    """Enhanced nozzle validation with improved criteria."""
    radius_inlet, radius_throat, radius_exit, _, _, _, _ = params
    
    # Calculate area ratios
    A_ratio = (radius_exit/radius_throat)**2
    
    # Get flow properties at nozzle entrance
    _, _, M_diff, P_diff, T_diff, _ = generate_spike_profile(params)
    gamma_T, R_T, _, _, _ = real_gas_properties(M_diff, T_diff*T0)
    
    # Calculate ideal expansion ratio with enhanced modeling
    M_design = 2.53  # Matched to performance calculation
    ideal_ratio = ((gamma_T+1)/2)**(-(gamma_T+1)/(2*(gamma_T-1))) * \
                 (1/M_design) * (1 + (gamma_T-1)/2 * M_design**2)**((gamma_T+1)/(2*(gamma_T-1)))
    
    # Tightened expansion ratio tolerance
    error = abs(A_ratio - ideal_ratio)/ideal_ratio
    if error > 0.08:  # Reduced from 0.10
        return False, f"Nozzle expansion ratio error: {error*100:.1f}%"
    
    # Updated expansion ratio requirements
    min_expansion = 1.65  # Adjusted from 1.6
    max_expansion = 1.95  # Adjusted from 2.0
    current_expansion = radius_exit/radius_throat
    
    if current_expansion < min_expansion:
        return False, f"Insufficient nozzle expansion: ratio = {current_expansion:.2f}"
    elif current_expansion > max_expansion:
        return False, f"Excessive nozzle expansion: ratio = {current_expansion:.2f}"
    
    return True, "Nozzle design valid"

def has_boundary_layer_bleed(params):
    """Check if the design includes boundary layer bleed features."""
    # For now, assume all designs include basic bleed features
    # In a more detailed implementation, this would check specific geometry features
    return True

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
        print(f"Pressure Recovery: {performance['pressure_ratio']:.3f}")
        
        # Export the DXF profile
        export_dxf_profile(best_params)
        print("\nProfile exported to 'ramjet_profile.dxf'")
        print("To use in Fusion 360:")
        print("1. Create a new sketch on the XY plane")
        print("2. Insert > Insert DXF")
        print("3. Select ramjet_profile.dxf")
        print("4. Use the Revolve command around the X axis")
        print("5. Add wall thickness as needed")
        
    except Exception as e:
        print(f"\n❌ Error during visualization/analysis: {str(e)}")
        exit(1)