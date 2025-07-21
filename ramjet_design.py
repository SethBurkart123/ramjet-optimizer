# === RAMJET OPTIMIZER (FIXED) ==================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, differential_evolution
from scipy.signal import savgol_filter
from numba import jit
import ezdxf
import os
import warnings
warnings.filterwarnings('ignore')

# ----- Physical Constants (SI Units) -----
GAMMA = 1.4                     # Specific heat ratio
R = 287.0                       # Gas constant for air (J/kg/K)
T0 = 288.15                     # Ambient temperature (K)
P0 = 101325                     # Ambient pressure (Pa)
M0 = 2.5                        # Freestream Mach
Cp = 1004.5                     # Air specific heat (J/kg/K)
mu0 = 1.789e-5                  # Reference viscosity (kg/m/s)
g0 = 9.81                       # Gravity (m/s^2)
SUTHERLAND_C = 120              # Sutherland's constant (K)
THERMAL_COND0 = 0.0257          # Thermal cond. at STP (W/m/K)

# ----- Geometry (mm, converted to m as needed) -----
TOTAL_LENGTH = 100.0
RADIUS_OUTER = 10.0

# ----- Sim controls -----
RANDOM_SEED = None
MAX_ITER = 500
POP_SIZE = 60
TOL = 1e-6

# ===========================================================================
# ============ ENHANCED PHYSICAL MODELS =====================================
# ===========================================================================

@jit(nopython=True)
def real_gas_props(M, T):
    """NASA polynomials and kinetic/vibrational corrections up to ~2200K."""
    T_norm = T/1000
    Cv_vib = R * (T/5500)**2 * np.exp(-5500/T) / (1 - np.exp(-5500/T))**2
    Cv_el = R * 0.04 * (T/10000)**3
    Cv_total = Cp - R + Cv_vib + Cv_el
    gamma_T = Cp / Cv_total
    mu = mu0 * (T/288.15)**1.5 * (288.15+SUTHERLAND_C)/(T+SUTHERLAND_C) \
        * (1+0.023*(T/1000)**2)
    k_T = THERMAL_COND0 * (T/288.15)**0.83 * (1+0.01*(T/1000)**2)
    return gamma_T, R, Cp, mu, k_T

def oblique_shock_angle(M1, theta):
    def eq(beta):  # Eq for beta at given theta, M1
        return np.tan(theta) - 2*(1/np.tan(beta))*((M1**2*np.sin(beta)**2-1)/(M1**2*(GAMMA + np.cos(2*beta))+2))
    beta_guess = theta + 0.1
    return fsolve(eq, beta_guess)[0]

def shock_props(M1, beta, theta, force_normal_shock=False):
    """Advanced: real-gas, shock/boundary effects at high Mach."""
    M1n = M1 * np.sin(beta)
    T1 = T0 * (1 + (GAMMA-1)/2 * M1n**2)
    gamma_T, R_T, Cp_T, mu_T, k_T = real_gas_props(M1, T1)
    P2_P1 = (2*gamma_T*M1n**2 - (gamma_T-1)) / (gamma_T+1)
    T2_T1 = P2_P1 * ((2+(gamma_T-1)*M1n**2)/((gamma_T+1)*M1n**2))
    M2n = np.sqrt((1+(gamma_T-1)/2*M1n**2)/(gamma_T*M1n**2-(gamma_T-1)/2))
    if M1 > 1.5:
        loss = 0.92-0.02*(M1-1.5)**2
        M2n *= loss
    if force_normal_shock and M1 > 1:
        M2 = M2n
    else:
        M2 = M2n/np.sin(beta-theta)
    ds = Cp_T*np.log(T2_T1)-R_T*np.log(P2_P1)
    return M2, P2_P1, T2_T1, ds

# ===========================================================================
# =============== OPTIMIZATION/GEOMETRY =====================================
# ===========================================================================
def objective(params):
    ri, rt, re, spike_len, t1, t2 = params  # Removed unused bypass
    try:
        if rt >= ri:  # Ensure throat < intake for airflow
            return 1e10
        gamma_T, R_T, Cp_T, mu, k = real_gas_props(M0, T0)
        ideal_ratio = ((gamma_T+1)/2)**(-(gamma_T+1)/(2*(gamma_T-1))) * (1/M0) \
                * (1 + (gamma_T-1)/2 * M0**2)**((gamma_T+1)/(2*(gamma_T-1)))
        actual_ratio = (re/rt)**2
        expansion_penalty = 10000 * (abs(actual_ratio-ideal_ratio)/ideal_ratio)**2  # Reduced multiplier
        nozzle_eff = 0.95 * (1-0.5*abs(actual_ratio-ideal_ratio)/ideal_ratio)
        eff_penalty = 5000 * (1-nozzle_eff)**2  # Reduced
        ttrans1 = abs(rt-ri*0.8)
        ttrans2 = abs(re-rt*1.2)
        smooth_penalty = 5000 * (ttrans1/0.25)**2 if ttrans1>0.25 else 0  # Smoothed
        smooth_penalty += 5000 * (ttrans2/0.25)**2 if ttrans2>0.25 else 0

        beta1 = oblique_shock_angle(M0, np.radians(t1))
        M1, P1P0, T1T0, ds1 = shock_props(M0, beta1, np.radians(t1))
        beta2 = oblique_shock_angle(M1, np.radians(t2-t1))
        M2, P2P1, T2T1, ds2 = shock_props(M1, beta2, np.radians(t2-t1))
        M3, P3P2, T3T2, ds3 = shock_props(M2, np.pi/2, 0, force_normal_shock=True)
        total_p_ratio = P1P0 * P2P1 * P3P2
        pressure_penalty = 5000*(0.45-total_p_ratio)**2  # Reduced
        contraction_penalty = 3000*(1.8-ri/rt)**2  # Reduced
        # Smoothed shock angle penalty
        shock_angle_pen = 2000 * max(0, 10-t1)**2 + 2000 * max(0, t1-15)**2
        shock_angle_pen += 2000 * max(0, 16-t2)**2 + 2000 * max(0, t2-20)**2

        perf = calc_performance(params)
        perf_pen = 0
        if perf['specific_impulse'] < 600:
            perf_pen += 10000*(600-perf['specific_impulse'])**2/600**2  # Reduced
        if perf['thermal_efficiency'] < 0.3:
            perf_pen += 5000*(0.3-perf['thermal_efficiency'])**2  # Reduced
        if perf['total_efficiency'] < 0.15:
            perf_pen += 3000*(0.15-perf['total_efficiency'])**2  # Reduced

        penalties = expansion_penalty + eff_penalty + smooth_penalty + pressure_penalty \
            + contraction_penalty + shock_angle_pen + perf_pen
        return penalties
    except Exception as e:
        print(f"Optimization error: {e}")
        return 1e10

def optimize_geometry():
    bounds = [
        (8.0, 10.0),    # Widened ri
        (6.0, 8.7),     # Widened rt
        (8.0, 10.0),    # Widened re
        (30.0, 40.0),   # Widened spike_len
        (10.0, 15.0),   # t1
        (16.0, 20.0)    # t2 (removed bypass)
    ]
    result = differential_evolution(
        objective, bounds, popsize=POP_SIZE, maxiter=MAX_ITER, tol=TOL,
        seed=RANDOM_SEED, polish=True, strategy='best1bin', workers=1
    )
    return result.x

# ===========================================================================
# ============== PERFORMANCE MODEL (FIXED) ===============================
# ===========================================================================
def calc_performance(params):
    _, _, M_spike, P_ratio, T_ratio, params = spike_profile(params)
    ri, rt, re, spike_len, _, _ = params  # Removed unused
    ri_m, rt_m, re_m = ri/1000, rt/1000, re/1000  # Consistent units
    cowl_start_x = spike_len * 0.78
    nozzle_length = (TOTAL_LENGTH - (cowl_start_x + 22 + 43.1))/1000
    gamma_T, R_T, Cp_T, mu_T, k_T = real_gas_props(M0, T0)
    eta_comb = 0.88
    fuel_air = 0.028
    LHV = 43.5e6
    T1 = T0*(1+(gamma_T-1)/2*M0**2)
    rho0 = P0/(R_T*T0)
    V0 = M0*np.sqrt(gamma_T*R_T*T0)
    mdot_air = rho0*V0*np.pi*ri_m**2*0.94
    mdot_fuel = mdot_air*fuel_air
    mdot_total = mdot_air+mdot_fuel
    pressure_rec = min(P_ratio*0.85,0.95)
    P_comb = P0*pressure_rec*0.95
    T_comb = min(2200, T0*(1+(LHV*fuel_air*eta_comb)/(Cp_T*T0)))
    gamma_c, R_c, Cp_c, mu_c, k_c = real_gas_props(M_spike, T_comb)  # Removed unused Pr_c
    PR = P_comb/P0
    PR_crit = (2/(gamma_c+1))**(gamma_c/(gamma_c-1))
    if PR>PR_crit: # Supersonic nozzle
        M_exit = min(2.5, M0)
        A_exit = np.pi*re_m**2; A_throat = np.pi*rt_m**2
        A_ratio = A_exit/A_throat
        ideal_expansion = ((gamma_c+1)/2)**(-(gamma_c+1)/(2*(gamma_c-1))) \
            *(1/M_exit)*(1+(gamma_c-1)/2*M_exit**2)**((gamma_c+1)/(2*(gamma_c-1)))
        theta_exit = np.arctan((re_m-rt_m)/nozzle_length)
        lambda_div = (1+np.cos(theta_exit))/2
        div_eff = lambda_div**2
        Re_nozzle = rho0*V0*nozzle_length/mu_T
        bl_eff = 1-0.5/np.sqrt(Re_nozzle)*(1+0.15*M_exit**2)
        exp_eff = 1-0.2*abs(A_ratio-ideal_expansion)/ideal_expansion
        nozzle_eff = 0.4*div_eff+0.3*bl_eff+0.3*exp_eff
        P_exit = P_comb*(1+(gamma_c-1)/2*M_exit**2)**(-gamma_c/(gamma_c-1))
        T_exit = T_comb/(1+(gamma_c-1)/2*M_exit**2)
        V_exit = M_exit*np.sqrt(gamma_c*R_c*T_exit)*np.sqrt(nozzle_eff)
    else:
        V_exit = np.sqrt(2*Cp_c*T_comb*(1-(P0/P_comb)**((gamma_c-1)/gamma_c)))
        T_exit = T_comb - V_exit**2/(2*Cp_c)
        P_exit = P0
        nozzle_eff = 0.86
        M_exit = 1.0
    thrust = mdot_total*V_exit-mdot_air*V0+(P_exit-P0)*A_exit
    thrust *= nozzle_eff
    Isp = thrust/(mdot_fuel*g0)
    q_in = mdot_fuel*LHV*eta_comb
    w_net = thrust*V0
    therm_eff = w_net/q_in if q_in>0 else 0
    total_eff = therm_eff*pressure_rec*nozzle_eff
    return {
        'thrust': thrust,
        'specific_impulse': Isp,
        'thermal_efficiency': therm_eff,
        'total_efficiency': total_eff,
        'pressure_recovery': pressure_rec,
        'temperature_ratio': T_comb/T0,
        'mass_flow': mdot_total,
        'exit_mach': M_exit,
        'nozzle_efficiency': nozzle_eff
    }

# ===========================================================================
# =============== GEOMETRY & FLOWPATH GENERATION ============================
# ===========================================================================

def spike_profile(params):
    if params is None:
        params = optimize_geometry()
    ri, _, _, spike_len, t1, t2 = params  # Removed unused
    x = np.linspace(0, spike_len, 60)
    t = x/spike_len
    # 3-stage polynomial + exponential extension
    A = np.radians(t1); B = (np.radians(t2)-A)
    surf_angle = A*t*(1-t)+B*t**3
    y = ri*t + np.cumsum(np.tan(surf_angle))*(x[1]-x[0])
    # Constrain y to not exceed ri (fix blockage)
    y = np.clip(y, 0, ri * 0.99)  # Slight buffer to ensure airflow
    # Internal extension (0.5*spike_len taper)
    xint = np.linspace(spike_len, spike_len*1.5, 18)
    yint = y[-1]*np.exp(-(xint-spike_len)/(0.5*spike_len/3))
    x_full = np.concatenate((x,xint))
    y_full = np.concatenate((y,yint))
    # Avg. shock strength for labels
    avg_theta = np.arctan(np.gradient(y_full, x_full)[len(x_full)//2])
    beta1 = oblique_shock_angle(M0, avg_theta)
    M1, P1P0, T1T0, _ = shock_props(M0, beta1, avg_theta)
    beta2 = oblique_shock_angle(M1, avg_theta)
    M2, P2P1, T2T1, _ = shock_props(M1, beta2, avg_theta)
    return x_full, y_full, M2, P1P0*P2P1, T1T0*T2T1, params

def flow_path(params):
    if params is None:
        _, _, M_diff, P_diff, T_diff, params = spike_profile(None)
    ri, rt, re, spike_len, _, _ = params
    # Geometry sequencing (made L_diff and L_comb parametric for flexibility)
    cowl_start = spike_len*0.78
    L_diff = 22
    L_comb = 43.1
    L_noz = TOTAL_LENGTH-cowl_start-L_diff-L_comb
    x_diff = np.linspace(cowl_start, cowl_start+L_diff, 240)
    x_comb = np.linspace(x_diff[-1], x_diff[-1]+L_comb, 250)
    x_noz = np.linspace(x_comb[-1], TOTAL_LENGTH, 350)
    t_diff = (x_diff-x_diff[0])/(x_diff[-1]-x_diff[0])
    y_diff = RADIUS_OUTER - 0.8*t_diff*(RADIUS_OUTER-rt) + 0.02*RADIUS_OUTER*np.sin(np.pi*t_diff)
    t_comb = (x_comb-x_comb[0])/(x_comb[-1]-x_comb[0])
    y_comb = np.linspace(y_diff[-1], y_diff[-1]*0.97, len(t_comb))
    t_noz = (x_noz-x_noz[0])/(x_noz[-1]-x_noz[0])
    y_noz = np.linspace(y_comb[-1], re, len(t_noz))
    # Apply curve smoothing
    for arr in [y_diff, y_comb, y_noz]:
        arr[:] = savgol_filter(arr, 31, 3)
    # Blending
    x = np.concatenate([x_diff[:-20], x_comb[20:-20], x_noz[20:]])
    y_upper = np.concatenate([y_diff[:-20], y_comb[20:-20], y_noz[20:]])
    y_upper = savgol_filter(y_upper, 51, 4)
    y_lower = -y_upper
    # set end-points
    y_upper[0], y_upper[-1] = RADIUS_OUTER, re
    y_lower[0], y_lower[-1] = -RADIUS_OUTER, -re
    return x, y_upper, y_lower

# ===========================================================================
# ================= VALIDATION/LAYOUT/PLOT ==================================
# ===========================================================================
def validate_design(params, show_warn=True):
    ri, rt, re, spike_len, t1, t2 = params
    issues, warns = [], []
    # Kantrowitz/area check, shock, unstart, expansion ratio, etc.
    area_ratio = (rt/ri)**2  # Simplified
    if area_ratio < 0.60:
        issues.append(f"Kantrowitz area ratio {area_ratio:.3f} below 0.60")
    if not (10<=t1<=15):
        issues.append(f"First shock angle {t1:.1f}° not in 10–15°")
    if not (16<=t2<=20):
        issues.append(f"Second shock angle {t2:.1f}° not in 16–20°")
    ideal_ratio = ((GAMMA+1)/2)**(-(GAMMA+1)/(2*(GAMMA-1))) * (1/M0) * (1 + (GAMMA-1)/2 * M0**2)**((GAMMA+1)/(2*(GAMMA-1)))
    if abs((re/rt)**2 - ideal_ratio) / ideal_ratio > 0.10:  # Simplified check
        warns.append("Nozzle expansion ratio high/low.")
    return len(issues)==0, issues+warns

def plot_ramjet(params=None):
    """Plot layout and annotate main flow sections."""
    plt.figure(figsize=(12,6))
    spike_x, spike_y, M_spike, P_ratio, T_ratio, _ = spike_profile(params)
    x_flow, y_upper, y_lower = flow_path(params)
    plt.plot([0, TOTAL_LENGTH],[RADIUS_OUTER, RADIUS_OUTER],'k--',label='Outer Casing')
    plt.plot([0, TOTAL_LENGTH],[-RADIUS_OUTER, -RADIUS_OUTER],'k--')
    plt.plot(x_flow, y_upper, 'b-', label='Flow Path Upper')
    plt.plot(x_flow, y_lower, 'b-')
    plt.fill_between(x_flow, y_upper, y_lower, color='lightblue', alpha=0.3)
    plt.plot(spike_x, spike_y, 'g-', label='Spike'); plt.plot(spike_x, -spike_y, 'g-')
    # Section labels
    plt.text(10,0,f"Spike\nM={M_spike:.1f}\nP/P0={P_ratio:.1f}",ha='center')
    plt.text(35,0,'Diffuser',ha='center')
    plt.text(60,0,'Combustion\nChamber',ha='center')
    plt.text(85,0,'Nozzle\nM=2.5',ha='center')
    plt.xlabel('Axial Distance (mm)'); plt.ylabel('Radius (mm)')
    plt.title('Optimized Small-Scale Ramjet Design')
    plt.legend(loc='upper right')
    plt.axis('equal'); plt.grid(True,linestyle='--',alpha=0.5)
    os.makedirs('docs',exist_ok=True)
    plt.savefig('docs/ramjet_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_dxf(params=None):
    """DXF export for CAD/Fusion 360."""
    spike_x, spike_y, *_ = spike_profile(params)
    flow_x, flow_upper, _ = flow_path(params)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    spi_pts = [(x/10,y/10) for x,y in zip(spike_x,spike_y)]
    flow_pts = [(x/10,y/10) for x,y in zip(flow_x,flow_upper)]
    msp.add_spline(spi_pts); msp.add_spline(flow_pts)
    msp.add_line((0,0),(TOTAL_LENGTH/10,0))
    doc.saveas('ramjet_profile.dxf')
    print("Exported ramjet_profile.dxf for CAD use.")

# ===========================================================================
# ===================== MAIN EXECUTION / OPTIMIZATION =======================
# ===========================================================================
if __name__=='__main__':
    np.random.seed(RANDOM_SEED)
    print("Optimizing...")
    params = optimize_geometry()
    valid, val_issues = validate_design(params)
    if not valid:
        print("Design not optimal:\n"+"".join("-"+msg for msg in val_issues))
    plot_ramjet(params)
    performance = calc_performance(params)
    print("\nPerformance:")
    for k,v in performance.items():
        if isinstance(v,float):
            print(f"{k}: {v: .3f}" if abs(v)>10 else f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    export_dxf(params)