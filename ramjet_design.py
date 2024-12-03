import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Physical constants
gamma = 1.4  # Ratio of specific heats for air
R = 287.0    # Gas constant for air [J/kg·K]
T0 = 288.15  # Freestream temperature [K]
P0 = 101325  # Freestream pressure [Pa]
M0 = 2.5     # Design Mach number for ramjet

# Geometric constraints (all in mm)
total_length = 100.0   
radius_outer = 10.0    

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
    """Optimize key geometric parameters for maximum performance."""
    def objective(params):
        # Unpack parameters
        radius_inlet, radius_throat, radius_exit, spike_length, theta1, theta2 = params
        
        try:
            # Calculate performance
            beta1 = oblique_shock_angle(M0, np.radians(theta1))
            M1, P1_P0, _ = shock_properties(M0, beta1, np.radians(theta1))
            
            beta2 = oblique_shock_angle(M1, np.radians(theta2-theta1))
            M2, P2_P1, _ = shock_properties(M1, beta2, np.radians(theta2-theta1))
            
            # Calculate total pressure recovery
            P_recovery = P1_P0 * P2_P1
            
            # Calculate mass flow and thrust
            A_inlet = np.pi * radius_inlet**2
            mdot = P0/(R*T0) * M0 * np.sqrt(gamma*R*T0) * A_inlet
            
            # Penalties for constraints
            penalties = 0
            if radius_inlet >= radius_outer:
                penalties += 1000
            if radius_throat >= radius_inlet:
                penalties += 1000
            if radius_exit <= radius_throat:
                penalties += 1000
            if spike_length >= total_length/3:
                penalties += 1000
            
            # Objective is thrust-to-drag ratio
            performance = (mdot * P_recovery) / (1 + penalties)
            return -performance  # Negative because we're minimizing
            
        except:
            return 1e10
    
    # Initial guess and bounds
    x0 = [6.0, 5.0, 8.5, 25.0, 12.0, 14.0]  # Initial values
    bounds = [
        (3.0, 9.0),    # radius_inlet
        (3.0, 8.0),    # radius_throat
        (4.0, 9.5),    # radius_exit
        (15.0, 35.0),  # spike_length
        (8.0, 16.0),   # theta1
        (10.0, 18.0)   # theta2
    ]
    
    from scipy.optimize import minimize
    result = minimize(objective, x0, bounds=bounds, method='SLSQP')
    return result.x

def generate_spike_profile():
    """Generate optimized spike profile with proper back design."""
    # Get optimized parameters
    params = optimize_geometry()
    radius_inlet, _, _, spike_length, theta1, theta2 = params
    
    # Front cone sections
    x1 = np.linspace(0, spike_length*0.6, 30)
    y1 = x1 * np.tan(np.radians(theta1))
    
    x2 = np.linspace(spike_length*0.6, spike_length, 20)
    y2 = y1[-1] + (x2 - x1[-1]) * np.tan(np.radians(theta2))
    
    # Add spike back design (curved profile)
    x_back = np.linspace(spike_length, spike_length*1.2, 15)
    
    # Bezier curve for smooth transition
    t = (x_back - x_back[0])/(x_back[-1] - x_back[0])
    y_back = y2[-1] * (1 - t)**2  # Quadratic bezier for smooth closure
    
    # Combine all sections
    x = np.concatenate([x1, x2, x_back])
    y = np.concatenate([y1, y2, y_back])
    
    # Calculate flow properties
    beta1 = oblique_shock_angle(M0, np.radians(theta1))
    M1, P1_P0, T1_T0 = shock_properties(M0, beta1, np.radians(theta1))
    
    beta2 = oblique_shock_angle(M1, np.radians(theta2-theta1))
    M2, P2_P1, T2_T1 = shock_properties(M1, beta2, np.radians(theta2-theta1))
    
    return x, y, M2, P1_P0*P2_P1, T1_T0*T2_T1, params

def generate_flow_path():
    """Generate flow path using optimized parameters."""
    # Get optimized geometry
    _, _, M_diff, P_diff, T_diff, params = generate_spike_profile()
    radius_inlet, radius_throat, radius_exit, spike_length, _, _ = params
    
    cowl_start_x = spike_length * 0.8  # Start cowl before spike tip for shock capture
    
    # Normal shock at diffuser entrance
    M_after_normal = np.sqrt((1 + (gamma-1)/2 * M_diff**2)/(gamma*M_diff**2 - (gamma-1)/2))
    P_normal = P_diff * (1 + 2*gamma/(gamma+1) * (M_diff**2 - 1))
    
    # Subsonic diffuser
    area_ratio = radius_inlet**2/radius_throat**2
    
    # Generate geometry sections
    x_diffuser = np.linspace(cowl_start_x, cowl_start_x + 15, 20)
    x_combustor = np.linspace(x_diffuser[-1], x_diffuser[-1] + 45, 20)
    x_nozzle = np.linspace(x_combustor[-1], total_length, 20)
    
    # Diffuser profile with smooth compression
    t_diff = (x_diffuser - x_diffuser[0])/(x_diffuser[-1] - x_diffuser[0])
    y_diffuser = radius_outer - (radius_outer - radius_inlet)*(1 - np.sin(np.pi*t_diff/2))
    
    # Constant area combustor
    y_combustor = radius_outer * np.ones_like(x_combustor)
    
    # Optimized CD nozzle profile
    t_nozzle = (x_nozzle - x_nozzle[0])/(x_nozzle[-1] - x_nozzle[0])
    
    # Converging section
    y_conv = radius_outer - (radius_outer - radius_throat)*np.sin(np.pi*t_nozzle/2)
    
    # Diverging section optimized for M=2.5 exit
    A_Astar = lambda M: ((gamma+1)/2)**(-((gamma+1)/(2*(gamma-1)))) * \
              (1/M) * (1 + ((gamma-1)/2)*M**2)**((gamma+1)/(2*(gamma-1)))
    
    exit_area_ratio = A_Astar(2.5)
    y_nozzle = np.where(t_nozzle <= 0.5, y_conv,
                        radius_throat + (radius_exit - radius_throat)*(2*(t_nozzle-0.5)))
    
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
    """Calculate key performance parameters."""
    # Get flow properties through engine - now correctly unpacking 6 values
    _, _, M_spike, P_ratio, T_ratio, params = generate_spike_profile()
    radius_inlet, radius_throat, radius_exit = params[:3]  # Get geometric parameters
    
    # Calculate mass flow rate
    A_inlet = np.pi * radius_inlet**2
    rho0 = P0/(R*T0)
    V0 = M0 * np.sqrt(gamma*R*T0)
    mdot = rho0 * V0 * A_inlet
    
    # Estimate thrust (simplified)
    P_exit = P0  # Assume optimal expansion
    V_exit = 2.5 * np.sqrt(gamma*R*T0*T_ratio)  # Exit velocity
    A_exit = np.pi * radius_exit**2
    
    thrust = mdot * (V_exit - V0) + A_exit * (P_exit - P0)
    return thrust

if __name__ == "__main__":
    plot_ramjet()
    thrust = calculate_performance()
    print(f"Estimated thrust: {thrust/1000:.2f} kN")
