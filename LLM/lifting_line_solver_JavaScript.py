import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. AIRFOIL AND BLADE ELEMENT CALCULATIONS
# =============================================================================

def polarAirfoil(alpha):
    polar_alpha = [-180, -16.062, -15.506, -15.064, -14.589, -14.109, -13.698,
        -13.237, -12.745, -12.268, -11.748, -11.183, -10.768, -10.231, -9.743,
        -9.223, -8.209, -7.187, -6.162, -5.143, -4.127, -3.106, -2.073, -1.04,
        .017, 1.025, 2.042, 3.096, 4.114, 5.126, 6.163, 7.189, 7.713,
        8.216, 8.734, 9.251, 9.558, 9.771, 10.269, 10.757, 11.257, 11.761,
        12.239, 13.224, 14.234, 15.227, 16.208, 17.201, 18.2, 19.201, 20.189,
        21.179, 22.162, 23.144, 23.547, 24.062, 25.056, 26.06, 27.059,
        28.062, 29.062, 30.056, 180]
    polar_cl = [-0, -.425, -.43, -.461, -.496, -.567, -.682, -.719, -.755,
        -.77, -.774, -.77, -.756, -.736, -.711, -.68, -.612, -.529, -.434,
        -.337, -.237, -.127, -.014, .102, .217, .33, .445, .571, .683,
        .792, .902, 1.014, 1.068, 1.12, 1.168, 1.207, 1.227, 1.229,
        1.215, 1.192, 1.173, 1.148, 1.126, 1.103, 1.093, 1.077,
        1.06, 1.038, 1.047, 1.043, 1.04, 1.029, 1.023, 1.007, .829,
        .854, .877, .89, .953, .976, 1.016, 1.041, 0]
    polar_cd = [0.1, .225, .217, .211, .208, .201, .077, .05, .039,
        .033, .029, .025, .023, .02, .018, .017, .014, .013, .011, .01, .008,
        .008, .008, .007, .007, .008, .008, .008, .008, .009, .009, .009, .009,
        .01, .01, .01, .012, .013, .019, .025, .036, .045, .053, .066, .076, .089, .104,
        .117, .132, .152, .172, .198, .228, .261, .416, .436, .466, .491, .541, .574, .616, .652, 0.1]
    
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    return [cl, cd]

def geoBlade(r_R):
    pitch = 2.0
    chord = 3.0 * (1.0 - r_R) + 1.0
    twist = -14.0 * (1.0 - r_R)
    return [chord, twist + pitch]

def loadBladeElement(Vnorm, Vtan, r_R):
    Vmag2 = Vnorm**2 + Vtan**2
    InflowAngle = math.atan2(Vnorm, Vtan)
    
    chord, twist = geoBlade(r_R)
    alpha = twist + InflowAngle * 180.0 / math.pi
    cl, cd = polarAirfoil(alpha)
    
    Lift = 0.5 * Vmag2 * cl * chord
    Drag = 0.5 * Vmag2 * cd * chord
    
    Fnorm = Lift * math.cos(InflowAngle) + Drag * math.sin(InflowAngle)
    Ftan = Lift * math.sin(InflowAngle) - Drag * math.cos(InflowAngle)
    Gamma = 0.5 * math.sqrt(Vmag2) * cl * chord
    
    # Returning extended data for plots
    return [Fnorm, Ftan, Gamma, InflowAngle * 180.0 / math.pi, alpha]

# =============================================================================
# 2. ROTOR AND WAKE GEOMETRY
# =============================================================================

def create_rotor_geometry(span_array, radius, tipspeedratio, theta_array, nblades):
    rings = []
    controlpoints = []
    
    for krot in range(nblades):
        angle_rotation = 2 * math.pi / nblades * krot
        cosrot = math.cos(angle_rotation)
        sinrot = math.sin(angle_rotation)

        for i in range(len(span_array) - 1):
            r = (span_array[i] + span_array[i+1]) / 2.0
            chord, twist = geoBlade(r / radius)
            
            cp = {
                'coordinates': [0.0, r, 0.0],
                'chord': chord,
                'dr': span_array[i+1] - span_array[i]
            }
            
            coords = cp['coordinates']
            cp['coordinates'] = [0.0, coords[1]*cosrot - coords[2]*sinrot, coords[1]*sinrot + coords[2]*cosrot]
            controlpoints.append(cp)
            
            filaments = []
            filaments.append({'x1': 0.0, 'y1': span_array[i], 'z1': 0.0, 'x2': 0.0, 'y2': span_array[i+1], 'z2': 0.0, 'Gamma': 0.0})
            
            c_in, tw_in = geoBlade(span_array[i]/radius)
            ang_in = tw_in * math.pi / 180.0
            filaments.append({'x1': c_in*math.sin(-ang_in), 'y1': span_array[i], 'z1': -c_in*math.cos(ang_in), 'x2': 0.0, 'y2': span_array[i], 'z2': 0.0, 'Gamma': 0.0})
            
            for j in range(len(theta_array) - 1):
                xt, yt, zt = filaments[-1]['x1'], filaments[-1]['y1'], filaments[-1]['z1']
                dy = (math.cos(-theta_array[j+1]) - math.cos(-theta_array[j])) * span_array[i]
                dz = (math.sin(-theta_array[j+1]) - math.sin(-theta_array[j])) * span_array[i]
                dx = (theta_array[j+1] - theta_array[j]) / tipspeedratio * radius
                filaments.append({'x1': xt+dx, 'y1': yt+dy, 'z1': zt+dz, 'x2': xt, 'y2': yt, 'z2': zt, 'Gamma': 0.0})

            c_out, tw_out = geoBlade(span_array[i+1]/radius)
            ang_out = tw_out * math.pi / 180.0
            filaments.append({'x1': 0.0, 'y1': span_array[i+1], 'z1': 0.0, 'x2': c_out*math.sin(-ang_out), 'y2': span_array[i+1], 'z2': -c_out*math.cos(ang_out), 'Gamma': 0.0})
            
            for j in range(len(theta_array) - 1):
                xt, yt, zt = filaments[-1]['x2'], filaments[-1]['y2'], filaments[-1]['z2']
                dy = (math.cos(-theta_array[j+1]) - math.cos(-theta_array[j])) * span_array[i+1]
                dz = (math.sin(-theta_array[j+1]) - math.sin(-theta_array[j])) * span_array[i+1]
                dx = (theta_array[j+1] - theta_array[j]) / tipspeedratio * radius
                filaments.append({'x1': xt, 'y1': yt, 'z1': zt, 'x2': xt+dx, 'y2': yt+dy, 'z2': zt+dz, 'Gamma': 0.0})

            for fil in filaments:
                fil['y1'], fil['z1'] = fil['y1']*cosrot - fil['z1']*sinrot, fil['y1']*sinrot + fil['z1']*cosrot
                fil['y2'], fil['z2'] = fil['y2']*cosrot - fil['z2']*sinrot, fil['y2']*sinrot + fil['z2']*cosrot

            rings.append({'filaments': filaments})

    return {'controlpoints': controlpoints, 'rings': rings}

# =============================================================================
# 3. FAST VECTORIZED BIOT-SAVART INDUCTION 
# =============================================================================

def velocity_induced_single_ring(ring, cp):
    velind = np.zeros(3)
    CORE = 1e-3
    for fil in ring['filaments']:
        X1 = np.array([fil['x1'], fil['y1'], fil['z1']])
        X2 = np.array([fil['x2'], fil['y2'], fil['z2']])
        XP = np.array(cp)
        
        R1 = XP - X1
        R2 = XP - X2
        R1xR2 = np.cross(R1, R2)
        R1xR2_sqr = np.dot(R1xR2, R1xR2)
        
        if R1xR2_sqr < CORE**2 or np.linalg.norm(R1) < CORE or np.linalg.norm(R2) < CORE:
            continue
            
        R0R1 = np.dot(X2 - X1, R1)
        R0R2 = np.dot(X2 - X1, R2)
        
        K = 1.0 / (4.0 * math.pi * R1xR2_sqr) * (R0R1/np.linalg.norm(R1) - R0R2/np.linalg.norm(R2))
        velind += K * R1xR2
        
    return velind

# =============================================================================
# 4. SOLVER
# =============================================================================

def solve_lifting_line(TSR, U_inf, R, N_blades, N_elements, N_rotations, a_wake_guess):
    Omega = TSR * U_inf / R
    
    # Cosine spacing recommended for LLM
    theta_dist = np.linspace(0, math.pi, N_elements + 1)
    span_array = R * (-0.5 * (np.cos(theta_dist) - 1) * 0.8 + 0.2)
    theta_array = np.linspace(0, N_rotations * 2 * math.pi, int(N_rotations * 18))
    
    tsr_wake = TSR / (1 - a_wake_guess)
    rotor_wake = create_rotor_geometry(span_array, R, tsr_wake, theta_array, N_blades)
    
    cps = rotor_wake['controlpoints']
    rings = rotor_wake['rings']
    N_total = len(cps)
    
    MatrixU = np.zeros((N_total, N_total))
    MatrixV = np.zeros((N_total, N_total))
    MatrixW = np.zeros((N_total, N_total))
    
    for icp in range(N_total):
        for jring in range(N_total):
            vind = velocity_induced_single_ring(rings[jring], cps[icp]['coordinates'])
            MatrixU[icp, jring] = vind[0]
            MatrixV[icp, jring] = vind[1]
            MatrixW[icp, jring] = vind[2]
            
    Gamma = np.zeros(N_total)
    GammaNew = np.zeros(N_total)
    
    a_res, aline_res, r_R_res = np.zeros(N_total), np.zeros(N_total), np.zeros(N_total)
    Fnorm_res, Ftan_res, phi_res, alpha_res = np.zeros(N_total), np.zeros(N_total), np.zeros(N_total), np.zeros(N_total)
    
    for kiter in range(300):
        for icp in range(N_total):
            r_cp = np.linalg.norm(cps[icp]['coordinates'])
            
            u = np.dot(MatrixU[icp, :], Gamma)
            v = np.dot(MatrixV[icp, :], Gamma)
            w = np.dot(MatrixW[icp, :], Gamma)
            
            vrot = np.cross([-Omega, 0.0, 0.0], cps[icp]['coordinates'])
            vel_tot = np.array([U_inf + u + vrot[0], v + vrot[1], w + vrot[2]])
            
            azimdir = np.cross([-1.0/r_cp, 0.0, 0.0], cps[icp]['coordinates'])
            vazim = np.dot(azimdir, vel_tot)
            vaxial = vel_tot[0]
            
            fnorm, ftan, gamma_val, phi, alpha = loadBladeElement(vaxial, vazim, r_cp/R)
            GammaNew[icp] = gamma_val
            
            a_res[icp] = -(u + vrot[0]) / U_inf
            aline_res[icp] = vazim / (r_cp * Omega) - 1.0
            r_R_res[icp] = r_cp / R
            Fnorm_res[icp] = fnorm
            Ftan_res[icp] = ftan
            phi_res[icp] = phi
            alpha_res[icp] = alpha
            
        err = np.max(np.abs(GammaNew - Gamma)) / max(np.max(np.abs(GammaNew)), 1e-3)
        if err < 0.02:
            print(f"  > Converged in {kiter} iterations.")
            break
            
        Gamma = (1 - 0.2) * Gamma + 0.2 * GammaNew

    # We only need the results for one blade (the first N_elements)
    blade_idx = slice(0, N_elements)
    
    # Calculate Integral Coefficients
    CT = np.sum(Fnorm_res[blade_idx] * [cp['dr'] for cp in cps[blade_idx]] * N_blades) / (0.5 * 1.0 * U_inf**2 * math.pi * R**2)
    CP = np.sum(Ftan_res[blade_idx] * r_R_res[blade_idx]*R * Omega * [cp['dr'] for cp in cps[blade_idx]] * N_blades) / (0.5 * 1.0 * U_inf**3 * math.pi * R**2)
    
    return {
        'r_R': r_R_res[blade_idx],
        'a': a_res[blade_idx],
        'aline': aline_res[blade_idx],
        'Fnorm': Fnorm_res[blade_idx],
        'Ftan': Ftan_res[blade_idx],
        'Gamma': Gamma[blade_idx],
        'phi': phi_res[blade_idx],
        'alpha': alpha_res[blade_idx],
        'CT': CT,
        'CP': CP
    }

# =============================================================================
# 5. EXECUTION & SEPARATE PDF PLOTTING
# =============================================================================
R = 50.0
U_inf = 10.0
N_blades = 3
TSRs = [6, 8, 10]
results = {}

for tsr in TSRs:
    print(f"Running LLM for TSR = {tsr}...")
    results[tsr] = solve_lifting_line(tsr, U_inf, R, N_blades, N_elements=15, N_rotations=3, a_wake_guess=0.25)

# --- Define Subfolder Path ---
# Get the exact folder where this Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the new subfolder name
output_dir = os.path.join(script_dir, "LLM_Plots_JavaScript")

# Create the folder if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving plots to: {output_dir}\n")

# 1. Angles
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for tsr in TSRs:
    axes[0].plot(results[tsr]['r_R'], results[tsr]['phi'], marker='o', label=f'TSR {tsr}')
    axes[1].plot(results[tsr]['r_R'], results[tsr]['alpha'], marker='s', label=f'TSR {tsr}')
axes[0].set(xlabel='r/R', ylabel='Inflow Angle $\Phi$ (deg)', title='Radial Distribution of Inflow Angle')
axes[1].set(xlabel='r/R', ylabel='Angle of Attack $\\alpha$ (deg)', title='Radial Distribution of Angle of Attack')
axes[0].grid(True); axes[0].legend()
axes[1].grid(True); axes[1].legend()
fig.savefig(os.path.join(output_dir, "LLM_1_Angles.pdf"), bbox_inches='tight')
plt.close()

# 2. Induction
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for tsr in TSRs:
    axes[0].plot(results[tsr]['r_R'], results[tsr]['a'], marker='o', label=f'TSR {tsr}')
    axes[1].plot(results[tsr]['r_R'], results[tsr]['aline'], marker='s', label=f'TSR {tsr}')
axes[0].set(xlabel='r/R', ylabel='Axial Induction $a$', title='Radial Distribution of Axial Induction')
axes[1].set(xlabel='r/R', ylabel='Tangential Induction $a\'$', title='Radial Distribution of Tangential Induction')
axes[0].grid(True); axes[0].legend()
axes[1].grid(True); axes[1].legend()
fig.savefig(os.path.join(output_dir, "LLM_2_Induction.pdf"), bbox_inches='tight')
plt.close()

# 3. Loading
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
norm_F = 0.5 * 1.0 * U_inf**2 * R
for tsr in TSRs:
    axes[0].plot(results[tsr]['r_R'], results[tsr]['Fnorm'] / norm_F, marker='o', label=f'TSR {tsr}')
    axes[1].plot(results[tsr]['r_R'], results[tsr]['Ftan'] / norm_F, marker='s', label=f'TSR {tsr}')
axes[0].set(xlabel='r/R', ylabel='Axial Loading $\\hat{F}_{axial}$', title='Non-dimensional Axial Loading')
axes[1].set(xlabel='r/R', ylabel='Tangential Loading $\\hat{F}_{azim}$', title='Non-dimensional Tangential Loading')
axes[0].grid(True); axes[0].legend()
axes[1].grid(True); axes[1].legend()
fig.savefig(os.path.join(output_dir, "LLM_3_Loading.pdf"), bbox_inches='tight')
plt.close()

# 4. Circulation
fig, ax = plt.subplots(figsize=(7, 5))
for tsr in TSRs:
    Omega = tsr * U_inf / R
    norm_Gamma = (N_blades * Omega) / (math.pi * U_inf**2)
    ax.plot(results[tsr]['r_R'], results[tsr]['Gamma'] * norm_Gamma, marker='o', label=f'TSR {tsr}')
ax.set(xlabel='r/R', ylabel='Circulation $\\hat{\\Gamma}$', title='Non-dimensional Radial Circulation')
ax.grid(True); ax.legend()
fig.savefig(os.path.join(output_dir, "LLM_4_Circulation.pdf"), bbox_inches='tight')
plt.close()

# 5. CT and CP
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
CTs = [results[tsr]['CT'] for tsr in TSRs]
CPs = [results[tsr]['CP'] for tsr in TSRs]
axes[0].plot(TSRs, CTs, 'b-o')
axes[1].plot(TSRs, CPs, 'r-s')
axes[0].set(xlabel='TSR', ylabel='$C_T$', title='Total Thrust Coefficient')
axes[1].set(xlabel='TSR', ylabel='$C_P$', title='Total Power Coefficient')
axes[0].grid(True); axes[1].grid(True)
fig.savefig(os.path.join(output_dir, "LLM_5_Performance.pdf"), bbox_inches='tight')
plt.close()

print(f"Finished generating plots.")