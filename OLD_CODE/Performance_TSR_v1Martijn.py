import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- 1. Load Airfoil Data ---
# Loading the Excel file directly as specified

# Adjust skiprows if the header 'Alfa, Cl, Cd, Cm' is on a different line
polar_df = pd.read_excel('polar DU95W180 (3).xlsx', skiprows=3)
cl_interp = interp1d(polar_df['Alfa'], polar_df['Cl'], kind='linear', fill_value='extrapolate')
cd_interp = interp1d(polar_df['Alfa'], polar_df['Cd'], kind='linear', fill_value='extrapolate')


# --- 2. Rotor and Operational Specifications ---
R = 50.0                # Radius [m]
N_blades = 3            # Number of blades
R_root = 0.2 * R        # Aerodynamic blade start (0.2 r/R)
U_inf = 10.0            # Freestream velocity [m/s]
rho = 1.225             # Air density [kg/m^3]
theta_pitch = -2.0      # Collective pitch [deg]
TSRs = [6, 8, 10]       # Tip Speed Ratios (lambda)
CT1 = 1.816             # Glauert constant from theory slides

# Discretization
N_annuli = 50
r_stations = np.linspace(R_root, R, N_annuli)
dr = r_stations[1] - r_stations[0]

# --- 3. Geometry Functions ---
def get_chord(r):
    return 3 * (1 - r/R) + 1

def get_twist(r):
    return 14 * (1 - r/R)

# --- 4. BEM Solver ---
def solve_rotor(TSR):
    Omega = TSR * U_inf / R
    total_thrust = 0
    total_torque = 0
    
    for r in r_stations:
        chord = get_chord(r)
        theta_twist = get_twist(r)
        # Total local pitch = twist + collective pitch
        theta_local = np.radians(theta_twist + theta_pitch)
        sigma_r = (N_blades * chord) / (2 * np.pi * r)
        
        # Initial guesses for induction factors
        a, adash = 0.3, 0
        relaxation = 0.1
        conv_limit = 1e-6
        
        for i in range(800):
            # 4.1 Kinematics
            # Inflow angle phi
            phi = np.arctan2(U_inf * (1 - a), Omega * r * (1 + adash))
            # Angle of attack alpha
            alpha_deg = np.degrees(phi) - np.degrees(theta_local)
            
            # 4.2 Airfoil coefficients from polar
            Cl = cl_interp(alpha_deg)
            Cd = cd_interp(alpha_deg)
            
            # 4.3 Force decomposition (Cx = Normal, Cy = Tangential)
            Cx = Cl * np.cos(phi) + Cd * np.sin(phi)
            Cy = Cl * np.sin(phi) - Cd * np.cos(phi)
            
            # 4.4 Prandtl Tip and Root Correction (F)
            # Tip factor
            f_tip = (N_blades / 2) * (R - r) / (r * np.abs(np.sin(phi)))
            F_tip = (2 / np.pi) * np.arccos(np.exp(-f_tip)) if f_tip < 50 else 1.0
            
            # Root factor
            f_root = (N_blades / 2) * (r - R_root) / (R_root * np.abs(np.sin(phi)))
            F_root = (2 / np.pi) * np.arccos(np.exp(-f_root)) if f_root < 50 else 1.0
            
            F = max(F_tip * F_root, 0.0001)
            
            # 4.5 Induction update with Glauert Correction
            # Local CT from Blade Element Theory
            CT_local = (sigma_r * (1 - a)**2 * Cx) / (np.sin(phi)**2)
            
            # Glauert transition check (a_critical ~ 0.35)
            a_transition = 1 - np.sqrt(CT1) / 2
            
            if a <= a_transition:
                # Standard momentum theory induction
                a_new = 1 / ((4 * F * (np.sin(phi)**2)) / (sigma_r * Cx) + 1)
            else:
                # Glauert empirical correction for high loading
                a_new = (1/F) * (0.143 + np.sqrt(max(0, 0.0203 - 0.6427 * (0.889 - CT_local))))
            
            # Tangential induction update
            adash_new = 1 / ((4 * F * np.sin(phi) * np.cos(phi)) / (sigma_r * Cy) - 1)
            
            # Check convergence
            if abs(a_new - a) < conv_limit:
                break
            
            # Apply relaxation to induction factors
            a = a * (1 - relaxation) + a_new * relaxation
            adash = adash * (1 - relaxation) + adash_new * relaxation

        # 4.6 Integrate local loads into global performance
        # Local resultant velocity W
        W = np.sqrt((U_inf * (1 - a))**2 + (Omega * r * (1 + adash))**2)
        
        # Incremental Thrust (dT) and Torque (dQ)
        dT = 0.5 * rho * (W**2) * N_blades * chord * Cx * dr
        dQ = 0.5 * rho * (W**2) * N_blades * chord * Cy * r * dr
        
        total_thrust += dT
        total_torque += dQ
        
    # 4.7 Performance Coefficients
    Area = np.pi * R**2
    CT = total_thrust / (0.5 * rho * U_inf**2 * Area)
    CP = (total_torque * Omega) / (0.5 * rho * U_inf**3 * Area)
    
    return CT, CP

# --- 5. Main Execution ---
print(f"{'TSR (lambda)':<15} | {'CT':<10} | {'CP':<10}")
print("-" * 40)

for tsr in TSRs:
    ct, cp = solve_rotor(tsr)
    print(f"{tsr:<15.1f} | {ct:<10.4f} | {cp:<10.4f}")