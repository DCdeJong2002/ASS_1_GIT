"""
assignment.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 1
================================================================
Self-contained script producing all required plots and saving
results to full_bem_results.npz for use with plot_results.py.

Run:  python assignment.py
"""

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

USE_HELICAL_PRANDTL = False   # True -> Prandtl 1919; False -> Glauert 1935

N_STARTS   = 12

# ── Two separate TSR sweeps ───────────────────────────────────────────────────
# TSR_SWEEP_SPAN   : used for all spanwise plots (4.1-4.3, 5, 6, 7, 8)
#                    3 values -> black, green, red line colors
# TSR_SWEEP_PERF   : used for CT/CQ/CP vs TSR performance curves (4.4)
#                    can be as wide as desired
TSR_SWEEP_SPAN = [6, 8, 10]
TSR_SWEEP_PERF = [4, 5, 6, 7, 8, 9, 10, 11, 12]

TSR_DESIGN = 8.0
CT_TARGET  = 0.75

Radius         = 50.0
NBlades        = 3
U0             = 10.0
rho            = 1.0              # air density [kg/m³]
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0             # baseline pitch [deg]

# Unified chord constraints — identical for ALL optimisation methods
CHORD_ROOT    = 3.4
CHORD_MIN     = 0.3
CHORD_MAX_REG = 6.0

# Single unified grid — constant spacing, 160 annuli
DELTA_R_R = 0.005

# =============================================================================
# MENU
# =============================================================================

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_PLOTS = False   # False -> save and close immediately (non-interactive)
                     # True  -> plt.show() after each save

# ── Computations ─────────────────────────────────────────────────────────────
RUN_TSR_SWEEP_SPAN  = True   # spanwise BEM for TSR_SWEEP_SPAN
                              #   required by: PLOT_4_1..4_3, PLOT_5, PLOT_6, PLOT_7
RUN_TSR_SWEEP_PERF  = True   # performance sweep for TSR_SWEEP_PERF
                              #   required by: PLOT_4_4
RUN_NO_CORRECTION   = True   # F=1 run at TSR=8
                              #   required by: PLOT_5
RUN_ANALYTICAL      = True   # analytical optimum via Brent root-finder
                              #   required by: PLOT_8, PLOT_9, PLOT_10
RUN_CUBIC           = True   # cubic polynomial optimiser
                              #   required by: PLOT_8
RUN_QUARTIC         = True   # quartic polynomial optimiser
                              #   required by: PLOT_8

# ── Plots ─────────────────────────────────────────────────────────────────────
PLOT_4_1 = True   # alpha and inflow angle vs r/R        (requires RUN_TSR_SWEEP_SPAN)
PLOT_4_2 = True   # axial and tangential induction vs r/R (requires RUN_TSR_SWEEP_SPAN)
PLOT_4_3 = True   # Cn and Ct loading vs r/R              (requires RUN_TSR_SWEEP_SPAN)
PLOT_4_4 = True   # CT, CQ, CP vs TSR  (broad sweep)      (requires RUN_TSR_SWEEP_PERF)
PLOT_5   = True   # tip correction influence               (requires RUN_TSR_SWEEP_SPAN
                  #                                         and RUN_NO_CORRECTION)
PLOT_6   = True   # annuli count, spacing, convergence     (requires RUN_TSR_SWEEP_SPAN)
PLOT_7   = True   # stagnation pressure                    (requires RUN_TSR_SWEEP_SPAN)
PLOT_8   = True   # design comparison                      (requires relevant RUN_* flags)
PLOT_9   = True   # Cl and chord relation                  (requires RUN_ANALYTICAL)
PLOT_10  = True   # Cl/Cd polar with operating points      (requires RUN_ANALYTICAL)

# ── Save ─────────────────────────────────────────────────────────────────────
SAVE_RESULTS = True   # write full_bem_results.npz after all computations

# =============================================================================
# 1.  POLAR DATA
# =============================================================================

_df         = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl    = _df["Cl"].to_numpy()
polar_cd    = _df["Cd"].to_numpy()

# =============================================================================
# 2.  COLOR SCHEME
# =============================================================================
#
# Spanwise sweep lines (3 values: 6, 8, 10): black, green, red.
# More values: extended colorblind-safe palette.
# Design comparison: fixed color per label (3 designs: green/blue/red,
#                                           4 designs: blue/green/red/orange).

_TSR_3_COLORS   = ["#0000ff", "#2ca02c", "#d62728"]
_TSR_MANY_COLORS = ["#000000","#2ca02c","#d62728","#0000ff",
                    "#ff7f0e","#9467bd","#8c564b","#e377c2"]

def _tsr_color(idx, n):
    return _TSR_3_COLORS[idx % 3] if n <= 3 else _TSR_MANY_COLORS[idx % len(_TSR_MANY_COLORS)]

_DESIGN_COLORS_3 = {"Baseline":"#2ca02c","Analytical":"#1f77b4",
                    "Cubic poly":"#d62728","Quartic poly":"#ff7f0e"}
_DESIGN_COLORS_4 = {"Baseline":"#1f77b4","Analytical":"#2ca02c",
                    "Cubic poly":"#d62728","Quartic poly":"#ff7f0e"}

def _design_color(label, n):
    d = _DESIGN_COLORS_3 if n <= 3 else _DESIGN_COLORS_4
    return d.get(label, "#888888")

# =============================================================================
# 3.  BEM CORE FUNCTIONS
# =============================================================================

def ainduction(CT):
    CT1 = 1.816; CT2 = 2.0*np.sqrt(CT1)-CT1
    if np.isscalar(CT):
        if CT >= CT2: return 1.0+(CT-CT1)/(4.0*(np.sqrt(CT1)-1.0))
        return 0.5-0.5*np.sqrt(max(0.0,1.0-CT))
    a = np.zeros_like(np.asarray(CT,dtype=float))
    a[CT>=CT2] = 1.0+(CT[CT>=CT2]-CT1)/(4.0*(np.sqrt(CT1)-1.0))
    a[CT<CT2]  = 0.5-0.5*np.sqrt(np.maximum(0.0,1.0-CT[CT<CT2]))
    return a

def _prandtl_simplified(r_R,rootR,tipR,TSR,NB,a_in):
    a = float(np.clip(a_in,-0.9,0.99))
    sq = np.sqrt(1.0+(TSR*r_R)**2/(1.0-a)**2)
    t1 = -NB/2.0*(tipR-r_R)/r_R*sq;  t2 = NB/2.0*(rootR-r_R)/r_R*sq
    return float(2.0/np.pi*np.arccos(float(np.clip(np.exp(t2),0,1)))
                *2.0/np.pi*np.arccos(float(np.clip(np.exp(t1),0,1))))

def _prandtl_helical(r_R,rootR,tipR,TSR,NB,a_in):
    a = float(np.clip(a_in,-0.9,0.99))
    d = max(2.0*np.pi/NB*(1.0-a)/np.sqrt(TSR**2+(1.0-a)**2),1e-8)
    Ft = 2.0/np.pi*np.arccos(np.exp(max(-np.pi*(tipR-r_R)/d,-500.0)))
    Fr = 2.0/np.pi*np.arccos(np.exp(max(-np.pi*(r_R-rootR)/d,-500.0)))
    return float((0.0 if np.isnan(Fr) else Fr)*(0.0 if np.isnan(Ft) else Ft))

def prandtl(r_R,rootR,tipR,TSR,NB,a):
    return (_prandtl_helical(r_R,rootR,tipR,TSR,NB,a) if USE_HELICAL_PRANDTL
            else _prandtl_simplified(r_R,rootR,tipR,TSR,NB,a))

def load_blade_element(vnorm,vtan,chord,twist):
    vmag2 = vnorm**2+vtan**2; phi = np.arctan2(vnorm,vtan)
    alpha = twist+np.degrees(phi)
    cl = float(np.interp(alpha,polar_alpha,polar_cl))
    cd = float(np.interp(alpha,polar_alpha,polar_cd))
    lift = 0.5*vmag2*cl*chord; drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(phi)+drag*np.sin(phi)
    ftan  = lift*np.sin(phi)-drag*np.cos(phi)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord
    return fnorm,ftan,gamma,alpha,float(np.degrees(phi)),cl,cd

def solve_streamtube(r1_R,r2_R,Omega,chord,twist,max_iter=300,tol=1e-5):
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2)
    r_mid = 0.5*(r1_R+r2_R); r_local = r_mid*Radius; TSR_now = Omega*Radius/U0
    a=0.0; aline=0.0; hist=[]; cl_f=cd_f=0.0
    for _ in range(max_iter):
        Vax=U0*(1.0-a); Vtan=(1.0+aline)*Omega*r_local
        fnorm,ftan,gamma,alpha,phi,cl,cd = load_blade_element(Vax,Vtan,chord,twist)
        cl_f=cl; cd_f=cd; hist.append(fnorm)
        CT_loc = fnorm*Radius*(r2_R-r1_R)*NBlades/(0.5*Area*U0**2)
        anew = ainduction(CT_loc)
        F = max(prandtl(r_mid,RootLocation_R,TipLocation_R,TSR_now,NBlades,anew),1e-4)
        anew/=F; a_old=a; a=0.75*a+0.25*anew
        aline = ftan*NBlades/(2.0*np.pi*U0*(1.0-a)*Omega*2.0*r_local**2)/F
        if abs(a-a_old)<tol: break
    return (np.array([a,aline,r_mid,fnorm,ftan,gamma,alpha,phi,cl_f,cd_f],dtype=float),
            np.array(hist))

def solve_streamtube_nocorr(r1_R,r2_R,Omega,chord,twist,max_iter=300,tol=1e-5):
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2)
    r_mid=0.5*(r1_R+r2_R); r_local=r_mid*Radius
    a=0.0; aline=0.0
    for _ in range(max_iter):
        Vax=U0*(1.0-a); Vtan=(1.0+aline)*Omega*r_local
        fnorm,ftan,*_ = load_blade_element(Vax,Vtan,chord,twist)
        CT_loc=fnorm*Radius*(r2_R-r1_R)*NBlades/(0.5*Area*U0**2)
        anew=ainduction(CT_loc); a=0.75*a+0.25*anew
        aline=ftan*NBlades/(2.0*np.pi*U0*max(1.0-a,1e-4)*Omega*2.0*r_local**2)
        if abs(a-anew)<tol: break
    return np.array([a,aline,r_mid,fnorm,ftan],dtype=float)

# =============================================================================
# 4.  ROTOR EVALUATOR
# =============================================================================

def make_bins(delta=DELTA_R_R):
    return np.arange(RootLocation_R, TipLocation_R+delta/2.0, delta)

def _run_sweep(tsr_list, bins, rmid):
    """Run BEM sweep over tsr_list. Returns perf dict and sweep_data dict."""
    perf={}; data={}
    for TSR in tsr_list:
        Omega=U0*TSR/Radius; rows=[]; hists=[]
        for i in range(len(bins)-1):
            rm=rmid[i]
            row,hist=solve_streamtube(bins[i],bins[i+1],Omega,
                                       3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows.append(row); hists.append(hist)
        res=np.vstack(rows); dr=np.diff(bins)*Radius
        CT=float(np.sum(dr*res[:,3]*NBlades/(0.5*U0**2*np.pi*Radius**2)))
        CP=float(np.sum(dr*res[:,4]*res[:,2]*NBlades*Radius*Omega/(0.5*U0**3*np.pi*Radius**2)))
        perf[TSR]={"CT":CT,"CP":CP}; data[TSR]=res
        print(f"  TSR={TSR}  CT={CT:.4f}  CP={CP:.4f}")
    return perf, data

def evaluate_rotor(r_nodes,c_nodes,tw_nodes,tsr=TSR_DESIGN):
    Omega=U0*tsr/Radius; rows=[]
    for i in range(len(r_nodes)-1):
        r1=r_nodes[i]/Radius; r2=r_nodes[i+1]/Radius
        chord=0.5*(c_nodes[i]+c_nodes[i+1]); twist=0.5*(tw_nodes[i]+tw_nodes[i+1])
        row,_=solve_streamtube(r1,r2,Omega,chord,twist)
        rows.append(row)
    res=np.vstack(rows); dr=np.diff(r_nodes)
    CT=float(np.sum(dr*res[:,3]*NBlades/(0.5*U0**2*np.pi*Radius**2)))
    CP=float(np.sum(dr*res[:,4]*res[:,2]*NBlades*Radius*Omega/(0.5*U0**3*np.pi*Radius**2)))
    return CT,CP,res

# =============================================================================
# 5.  BASELINE GEOMETRY
# =============================================================================

def baseline_geometry():
    bins=make_bins()
    return bins*Radius, 3.0*(1.0-bins)+1.0, -(14.0*(1.0-bins)+Pitch)

# =============================================================================
# 6.  TSR SWEEPS
# =============================================================================

bins_main = make_bins()
rmid_R    = 0.5*(bins_main[:-1]+bins_main[1:])
Omega8    = U0*8.0/Radius

# Spanwise sweep (6,8,10)
sweep_span={}; sweep_data_span={}
results_tsr8=None; ct_hist_tsr8=None; F_tsr8=None

if RUN_TSR_SWEEP_SPAN:
    print("Running spanwise TSR sweep", TSR_SWEEP_SPAN, "...")
    sweep_span, sweep_data_span = _run_sweep(TSR_SWEEP_SPAN, bins_main, rmid_R)

    # Extract TSR=8 extras (convergence history, F)
    TSR=8; Omega=U0*TSR/Radius; rows=[]; hists=[]
    for i in range(len(bins_main)-1):
        rm=rmid_R[i]
        row,hist=solve_streamtube(bins_main[i],bins_main[i+1],Omega,
                                   3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
        rows.append(row); hists.append(hist)
    results_tsr8=np.vstack(rows)
    dr_m=np.diff(bins_main)*Radius
    max_len=max(len(h) for h in hists)
    padded=np.array([np.concatenate([h,np.full(max_len-len(h),h[-1])]) for h in hists])
    ct_hist_tsr8=np.sum(padded*dr_m[:,None]*NBlades/(0.5*U0**2*np.pi*Radius**2),axis=0)
    F_tsr8=np.array([max(prandtl(rmid_R[i],RootLocation_R,TipLocation_R,
                                  Omega*Radius/U0,NBlades,results_tsr8[i,0]),1e-4)
                     for i in range(len(rmid_R))])
    # Store TSR=8 result in sweep_data_span too (it may already be there)
    sweep_data_span[8] = results_tsr8

# Performance sweep (wide range)
sweep_perf={}; sweep_data_perf={}

if RUN_TSR_SWEEP_PERF:
    print("Running performance TSR sweep", TSR_SWEEP_PERF, "...")
    sweep_perf, sweep_data_perf = _run_sweep(TSR_SWEEP_PERF, bins_main, rmid_R)

# No-correction run
res_nc=None
if RUN_NO_CORRECTION:
    print("Running no-correction BEM (TSR=8) ...")
    rows_nc=[]
    for i in range(len(bins_main)-1):
        rm=rmid_R[i]
        rows_nc.append(solve_streamtube_nocorr(bins_main[i],bins_main[i+1],Omega8,
                                                3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch)))
    res_nc=np.vstack(rows_nc)

# =============================================================================
# 7.  ANALYTICAL OPTIMUM
# =============================================================================

def find_optimal_alpha():
    alphas=np.linspace(polar_alpha[0],polar_alpha[-1],2000)
    cl_v=np.interp(alphas,polar_alpha,polar_cl)
    cd_v=np.interp(alphas,polar_alpha,polar_cd)
    with np.errstate(divide="ignore",invalid="ignore"):
        ratio=np.where(cd_v>1e-6,cl_v/cd_v,0.0)
    idx=int(np.argmax(ratio))
    return float(alphas[idx]),float(cl_v[idx]),float(cd_v[idx])

def generate_ideal_rotor(target_a,n_nodes=101,alpha_opt_deg=None,cl_opt=None,cd_opt=None,ap_in=None):
    if alpha_opt_deg is None: alpha_opt_deg,cl_opt,cd_opt=find_optimal_alpha()
    r_start=RootLocation_R*Radius+0.005*Radius; r_end=TipLocation_R*Radius-0.005*Radius
    r_nodes=r_start+(r_end-r_start)*(1.0-np.cos(np.linspace(0.0,np.pi,n_nodes)))/2.0
    c_nodes=np.zeros(n_nodes); tw_nodes=np.zeros(n_nodes)
    ap_nodes=(np.zeros(n_nodes) if ap_in is None
              else np.interp(r_nodes,0.5*(r_nodes[:-1]+r_nodes[1:]),
                             ap_in,left=ap_in[0],right=ap_in[-1]))
    for i,r in enumerate(r_nodes):
        r_R=r/Radius
        F=float(max(prandtl(r_R,RootLocation_R,TipLocation_R,TSR_DESIGN,NBlades,target_a),1e-4))
        phi=np.arctan2(1.0-target_a,TSR_DESIGN*r_R*(1.0+float(ap_nodes[i])))
        tw_nodes[i]=alpha_opt_deg-np.degrees(phi)
        Cn=cl_opt*np.cos(phi)+cd_opt*np.sin(phi)
        num=8.0*np.pi*r*target_a*F*(1.0-target_a*F)*np.sin(phi)**2
        den=NBlades*(1.0-target_a)**2*max(Cn,1e-8)
        c_nodes[i]=num/den
    c_nodes=np.clip(c_nodes,0.0,CHORD_ROOT)
    mx=int(np.argmax(c_nodes)); c_nodes[:mx+1]=c_nodes[mx]
    c_nodes=np.clip(c_nodes,CHORD_MIN,None)
    return r_nodes,c_nodes,tw_nodes

def design_for_exact_ct():
    alpha_opt,cl_opt,cd_opt=find_optimal_alpha()
    last_ap=[None]; last_geom=[None]
    def residual(ta):
        r,c,tw=generate_ideal_rotor(ta,alpha_opt_deg=alpha_opt,cl_opt=cl_opt,cd_opt=cd_opt,ap_in=last_ap[0])
        last_geom[0]=(r,c,tw)
        _out,sys.stdout=sys.stdout,io.StringIO()
        try: CT,CP,res_arr=evaluate_rotor(r,c,tw)
        finally: sys.stdout=_out
        last_ap[0]=res_arr[:,1]
        print(f"  target_a={ta:.5f}  CT={CT:.5f}  CP={CP:.5f}")
        return CT-CT_TARGET
    print("\nAnalytical: root-finding for CT =",CT_TARGET)
    sol=root_scalar(residual,bracket=[0.20,0.35],method="brentq",xtol=1e-5)
    if not sol.converged: raise RuntimeError("Brent did not converge.")
    print(f"  Converged: a={sol.root:.6f}")
    r,c,tw=last_geom[0]; CT,CP,res_arr=evaluate_rotor(r,c,tw)
    print(f"  CT={CT:.6f}  CP={CP:.6f}")
    return r,c,tw,CT,CP,res_arr

# =============================================================================
# 8.  POLYNOMIAL OPTIMISERS
# =============================================================================

def _span_x(r_R): return (r_R-RootLocation_R)/(TipLocation_R-RootLocation_R)

def chord_poly(r_R,c_tip,c2,c3,c4=0.0):
    x=_span_x(r_R); b1=c_tip-CHORD_ROOT-c2-c3-c4
    return CHORD_ROOT+b1*x+c2*x**2+c3*x**3+c4*x**4

def twist_poly(r_R,pitch,t_root,t_tip,t_curve):
    x=_span_x(r_R)
    return pitch+t_root*(1-x)+t_tip*x+t_curve*x*(1-x)

def build_poly_geometry(params):
    n=len(params)
    if n==7: pitch,t_root,t_tip,t_curve,c_tip,c2,c3=params; c4=0.0
    else:    pitch,t_root,t_tip,t_curve,c_tip,c2,c3,c4=params
    bins=make_bins(); r=bins*Radius
    c=np.array([chord_poly(rr,c_tip,c2,c3,c4) for rr in bins])
    tw=np.array([twist_poly(rr,pitch,t_root,t_tip,t_curve) for rr in bins])
    return r,c,tw

def _chord_minmax(params,n=300):
    n_p=len(params)
    if n_p==7: _,_,_,_,c_tip,c2,c3=params; c4=0.0
    else:      _,_,_,_,c_tip,c2,c3,c4=params
    rr=np.linspace(RootLocation_R,TipLocation_R,n)
    cv=np.array([chord_poly(r,c_tip,c2,c3,c4) for r in rr])
    return float(cv.min()),float(cv.max())

def _make_objective(quartic):
    def obj(params):
        c_min,c_max=_chord_minmax(params); pen=0.0
        if c_min<CHORD_MIN:    pen+=1e4*(CHORD_MIN-c_min)**2
        if c_max>CHORD_MAX_REG: pen+=1e3*(c_max-CHORD_MAX_REG)**2
        if c_min<CHORD_MIN*0.5: return 1e9
        r,c,tw=build_poly_geometry(params); CT,CP,_=evaluate_rotor(r,c,tw)
        pen+=5e3*(CT-CT_TARGET)**2
        if quartic: _,_,_,t_curve,_,c2,c3,c4=params; pen+=0.05*t_curve**2+0.01*(c2**2+c3**2+c4**2)
        else:       _,_,_,t_curve,_,c2,c3=params;    pen+=0.05*t_curve**2+0.01*(c2**2+c3**2)
        return -CP+pen
    return obj

def run_poly_optimizer(quartic=False,n_starts=N_STARTS,seed=42):
    rng=np.random.default_rng(seed); obj=_make_objective(quartic); label="quartic" if quartic else "cubic"
    if quartic:
        bounds=[(-8,8),(-25,5),(-10,15),(-20,20),(0.3,2),(-5,5),(-5,5),(-5,5)]
        x0_nom=np.array([-2,-7,2,0,1,0,0,0],dtype=float)
        def _rand(): return np.array([rng.uniform(-6,6),rng.uniform(-20,0),rng.uniform(-5,10),
                                       rng.uniform(-10,10),rng.uniform(0.3,1.5),
                                       rng.uniform(-3,3),rng.uniform(-3,3),rng.uniform(-3,3)])
    else:
        bounds=[(-8,8),(-25,5),(-10,15),(-20,20),(0.3,2),(-5,5),(-5,5)]
        x0_nom=np.array([-2,-7,2,0,1,0,0],dtype=float)
        def _rand(): return np.array([rng.uniform(-6,6),rng.uniform(-20,0),rng.uniform(-5,10),
                                       rng.uniform(-10,10),rng.uniform(0.3,1.5),
                                       rng.uniform(-3,3),rng.uniform(-3,3)])
    starts=[x0_nom]+[_rand() for _ in range(n_starts-1)]
    best_obj=np.inf; best_p=x0_nom.copy()
    for k,x0 in enumerate(starts,1):
        res=minimize(obj,x0,method="L-BFGS-B",bounds=bounds,
                     options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
        try:
            r_,c_,tw_=build_poly_geometry(res.x); CT_,_,_=evaluate_rotor(r_,c_,tw_)
        except Exception: CT_=float("nan")
        print(f"  [{label}] {k:02d}/{n_starts}  obj={res.fun:+.6f}  CT={CT_:.4f}")
        if res.fun<best_obj: best_obj=res.fun; best_p=res.x.copy()
    r,c,tw=build_poly_geometry(best_p); CT,CP,res_a=evaluate_rotor(r,c,tw)
    return best_p,r,c,tw,CT,CP,res_a

# =============================================================================
# 9.  RUN ALL METHODS
# =============================================================================

r_base=c_base=tw_base=res_base=None; CT_base=CP_base=None
r_anal=c_anal=tw_anal=res_anal=None; CT_anal=CP_anal=None
p_cubic=r_cubic=c_cubic=tw_cubic=res_cubic=None; CT_cubic=CP_cubic=None
p_qrt=r_qrt=c_qrt=tw_qrt=res_qrt=None; CT_qrt=CP_qrt=None
a_ad=0.5*(1.0-np.sqrt(1.0-CT_TARGET)); CP_ad=4.0*a_ad*(1.0-a_ad)**2

print("\n"+"="*60); print("BASELINE")
r_base,c_base,tw_base=baseline_geometry()
CT_base,CP_base,res_base=evaluate_rotor(r_base,c_base,tw_base)
print(f"  CT={CT_base:.6f}  CP={CP_base:.6f}")

if RUN_ANALYTICAL:
    print("\n"+"="*60); print("ANALYTICAL OPTIMUM")
    r_anal,c_anal,tw_anal,CT_anal,CP_anal,res_anal=design_for_exact_ct()

if RUN_CUBIC:
    print("\n"+"="*60); print("CUBIC POLYNOMIAL OPTIMISER")
    p_cubic,r_cubic,c_cubic,tw_cubic,CT_cubic,CP_cubic,res_cubic=run_poly_optimizer(quartic=False)
    print(f"  Best: CT={CT_cubic:.6f}  CP={CP_cubic:.6f}")

if RUN_QUARTIC:
    print("\n"+"="*60); print("QUARTIC POLYNOMIAL OPTIMISER")
    p_qrt,r_qrt,c_qrt,tw_qrt,CT_qrt,CP_qrt,res_qrt=run_poly_optimizer(quartic=True)
    print(f"  Best: CT={CT_qrt:.6f}  CP={CP_qrt:.6f}")

print("\n"+"="*60); print("PERFORMANCE SUMMARY")
print(f"  Actuator disk  CT={CT_TARGET}  a={a_ad:.4f}  CP={CP_ad:.6f}")
print(f"  Baseline       CT={CT_base:.6f}  CP={CP_base:.6f}  CP/CP_AD={CP_base/CP_ad:.4f}")
if RUN_ANALYTICAL: print(f"  Analytical     CT={CT_anal:.6f}  CP={CP_anal:.6f}  CP/CP_AD={CP_anal/CP_ad:.4f}")
if RUN_CUBIC:      print(f"  Cubic poly     CT={CT_cubic:.6f}  CP={CP_cubic:.6f}  CP/CP_AD={CP_cubic/CP_ad:.4f}")
if RUN_QUARTIC:    print(f"  Quartic poly   CT={CT_qrt:.6f}  CP={CP_qrt:.6f}  CP/CP_AD={CP_qrt/CP_ad:.4f}")

# =============================================================================
# 10.  PLOTS
# =============================================================================

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_assignment")
os.makedirs(save_folder, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(save_folder, name), dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}")
    if SHOW_PLOTS: plt.show()
    else:          plt.close()

norm_val = 0.5*U0**2*Radius
n_span   = len(TSR_SWEEP_SPAN)

# ── 4.1  Alpha and inflow angle ───────────────────────────────────────────────
if PLOT_4_1 and sweep_data_span:
    for col_idx, (qty_col, ylabel, fname) in enumerate([
            (6, r"$\alpha$ [deg]", "4_1a_angle_of_attack_vs_rR.png"),
            (7, r"$\phi$ [deg]",   "4_1b_inflow_angle_vs_rR.png")]):
        fig, ax = plt.subplots(figsize=(8,5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col], color=_tsr_color(k,n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

# ── 4.2  Induction factors ────────────────────────────────────────────────────
if PLOT_4_2 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",   "4_2a_axial_induction_vs_rR.png"),
            (1, r"$a'$ [-]",  "4_2b_tangential_induction_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8,5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col], color=_tsr_color(k,n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

# ── 4.3  Loading ──────────────────────────────────────────────────────────────
if PLOT_4_3 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "4_3a_normal_loading_Cn_vs_rR.png"),
            (4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$", "4_3b_azimuthal_loading_Ct_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8,5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col]/norm_val, color=_tsr_color(k,n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

# ── 4.4  CT, CQ, CP vs TSR (broad sweep) ─────────────────────────────────────
if PLOT_4_4 and sweep_perf:
    tsr_p = sorted(sweep_perf)
    CT_p  = [sweep_perf[t]["CT"] for t in tsr_p]
    CP_p  = [sweep_perf[t]["CP"] for t in tsr_p]
    CQ_p  = [sweep_perf[t]["CP"]/t for t in tsr_p]

    for vals, ylabel, fname in [
            (CT_p, r"$C_T$ [-]", "4_4a_thrust_coefficient_CT_vs_TSR.png"),
            (CQ_p, r"$C_Q$ [-]", "4_4b_torque_coefficient_CQ_vs_TSR.png"),
            (CP_p, r"$C_P$ [-]", "4_4c_power_coefficient_CP_vs_TSR.png")]:
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(tsr_p, vals, "o-", color="#1f77b4", lw=2)
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]"); ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.tight_layout(); save_fig(fname)

# ── 5  Tip correction ─────────────────────────────────────────────────────────
if PLOT_5 and results_tsr8 is not None and res_nc is not None:
    r_R8 = results_tsr8[:,2]
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",    "5a_axial_induction_tip_correction_comparison.png"),
            (3, r"$C_n$ [-]",  "5b_normal_loading_tip_correction_comparison.png")]:
        fig, ax = plt.subplots(figsize=(8,5))
        yc = results_tsr8[:,qty_col] if qty_col==0 else results_tsr8[:,qty_col]/norm_val
        ync= res_nc[:,0]             if qty_col==0 else res_nc[:,3]/norm_val
        ax.plot(r_R8,       yc,  "b-",  lw=2, label="With Prandtl correction")
        ax.plot(res_nc[:,2],ync, "r--", lw=2, label="No correction (F=1)")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_5:
    print("  [SKIP] PLOT_5 — requires RUN_TSR_SWEEP_SPAN and RUN_NO_CORRECTION")

# ── 6a  Number of annuli ──────────────────────────────────────────────────────
if PLOT_6 and results_tsr8 is not None:
    annuli_cols=["#000000","#2ca02c","#d62728"]
    fig, ax = plt.subplots(figsize=(8,5))
    for idx, N in enumerate([8, 20, 100]):
        b=np.linspace(RootLocation_R,TipLocation_R,N+1); rows_N=[]
        for i in range(N):
            rm=0.5*(b[i]+b[i+1])
            row,_=solve_streamtube(b[i],b[i+1],Omega8,3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows_N.append(row)
        res_N=np.vstack(rows_N)
        ax.plot(res_N[:,2],res_N[:,3]/norm_val,"-o",markersize=4,
                color=annuli_cols[idx],lw=2,label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a_normal_loading_influence_of_number_of_annuli.png")

    # ── 6b  Spacing method ────────────────────────────────────────────────────
    spacing_cols={"Constant":"#000000","Cosine":"#2ca02c"}
    N_sp=40; fig, ax = plt.subplots(figsize=(8,5))
    for lbl, b in {"Constant":np.linspace(RootLocation_R,TipLocation_R,N_sp+1),
                   "Cosine":RootLocation_R+(TipLocation_R-RootLocation_R)
                            *0.5*(1-np.cos(np.linspace(0,np.pi,N_sp+1)))}.items():
        rows_sp=[]
        for i in range(N_sp):
            rm=0.5*(b[i]+b[i+1])
            row,_=solve_streamtube(b[i],b[i+1],Omega8,3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows_sp.append(row)
        res_sp=np.vstack(rows_sp)
        ax.plot(res_sp[:,2],res_sp[:,3]/norm_val,"-o",markersize=5,
                color=spacing_cols[lbl],lw=2,label=lbl)
    ax.set_xlim(0.85,1.01); ax.set_ylim(0.5,1.5)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b_normal_loading_spacing_method_comparison.png")

    # ── 6c  Convergence ───────────────────────────────────────────────────────
    n_show=min(60,len(ct_hist_tsr8))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1,len(ct_hist_tsr8)+1),ct_hist_tsr8,"b-",lw=2)
    ax.set_xlim(1,n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$C_T$ [-]")
    ax.grid(True); fig.tight_layout(); save_fig("6c_CT_convergence_history.png")

    resid=np.abs(np.diff(ct_hist_tsr8))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.semilogy(range(2,len(ct_hist_tsr8)+1),resid,"r-",lw=2,
                label=r"$|C_{T,i}-C_{T,i-1}|$")
    ax.axhline(1e-5,color="k",ls="--",lw=0.8,label="Tolerance = 1e-5")
    ax.set_xlim(1,n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$|\Delta C_T|$")
    ax.legend(); ax.grid(True,which="both")
    fig.tight_layout(); save_fig("6d_CT_convergence_residuals_log_scale.png")

elif PLOT_6:
    print("  [SKIP] PLOT_6 — requires RUN_TSR_SWEEP_SPAN")

# ── 7  Stagnation pressure ────────────────────────────────────────────────────
if PLOT_7 and results_tsr8 is not None and F_tsr8 is not None:
    r_R8=results_tsr8[:,2]
    P0_up=0.5*rho*U0**2*np.ones(len(r_R8))
    dP0=2.0*rho*U0**2*results_tsr8[:,0]*F_tsr8*(1.0-results_tsr8[:,0]*F_tsr8)
    P0_down=P0_up-dP0
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(r_R8,P0_up,  "b-",  lw=2,   label=r"$P_0^{\infty,\mathrm{up}}$  (stat. 1)")
    ax.plot(r_R8,P0_up,  "b--", lw=1.5, alpha=0.6, label=r"$P_0^+$  rotor upwind  (stat. 2)")
    ax.plot(r_R8,P0_down,"r--", lw=1.5, alpha=0.6, label=r"$P_0^-$  rotor downwind  (stat. 3)")
    ax.plot(r_R8,P0_down,"r-",  lw=2,   label=r"$P_0^{\infty,\mathrm{down}}$  (stat. 4)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"Stagnation pressure $P_0$ [Pa]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("7_stagnation_pressure_four_stations.png")
elif PLOT_7:
    print("  [SKIP] PLOT_7 — requires RUN_TSR_SWEEP_SPAN")

# ── 8  Design comparison ──────────────────────────────────────────────────────
if PLOT_8 and res_base is not None:
    r_R_d=np.linspace(RootLocation_R,TipLocation_R,400)

    # Build designs list from whatever was run
    designs=[("Baseline",
              np.interp(r_R_d,r_base/Radius,c_base),
              np.interp(r_R_d,r_base/Radius,tw_base),
              res_base,CT_base,CP_base)]
    if res_anal is not None:
        designs.append(("Analytical",
                         np.interp(r_R_d,r_anal/Radius,c_anal),
                         np.interp(r_R_d,r_anal/Radius,tw_anal),
                         res_anal,CT_anal,CP_anal))
    if res_cubic is not None:
        designs.append(("Cubic poly",
                         np.interp(r_R_d,r_cubic/Radius,c_cubic),
                         np.interp(r_R_d,r_cubic/Radius,tw_cubic),
                         res_cubic,CT_cubic,CP_cubic))
    if res_qrt is not None:
        designs.append(("Quartic poly",
                         np.interp(r_R_d,r_qrt/Radius,c_qrt),
                         np.interp(r_R_d,r_qrt/Radius,tw_qrt),
                         res_qrt,CT_qrt,CP_qrt))
    n_d=len(designs)

    # 8a — chord
    fig, ax = plt.subplots(figsize=(9,5))
    for lbl,c_d,*_ in designs:
        ax.plot(r_R_d,c_d,color=_design_color(lbl,n_d),lw=2,label=lbl)
    ax.axhline(CHORD_MIN,color="grey",ls=":",lw=1,label=f"Min chord  {CHORD_MIN} m")
    ax.axhline(CHORD_ROOT,color="k",ls="--",lw=0.8,label=f"Root chord  {CHORD_ROOT} m")
    ax.set_xlabel("r/R"); ax.set_ylabel("Chord [m]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8a_chord_distribution_design_comparison.png")

    # 8b — twist
    fig, ax = plt.subplots(figsize=(9,5))
    for lbl,_,tw_d,*_ in designs:
        ax.plot(r_R_d,tw_d,color=_design_color(lbl,n_d),lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel("Twist [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8b_twist_distribution_design_comparison.png")

    # 8ab — chord and twist combined
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    for lbl,c_d,tw_d,*_ in designs:
        col=_design_color(lbl,n_d)
        axes[0].plot(r_R_d,c_d,color=col,lw=2,label=lbl)
        axes[1].plot(r_R_d,tw_d,color=col,lw=2,label=lbl)
    axes[0].axhline(CHORD_MIN,color="grey",ls=":",lw=1,label=f"Min  {CHORD_MIN} m")
    axes[0].axhline(CHORD_ROOT,color="k",ls="--",lw=0.8,label=f"Root  {CHORD_ROOT} m")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
    axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); save_fig("8ab_chord_and_twist_design_comparison.png")

    # 8c — axial induction
    fig, ax = plt.subplots(figsize=(9,5))
    for lbl,_,_,res,*_ in designs:
        ax.plot(res[:,2],res[:,0],color=_design_color(lbl,n_d),lw=2,label=lbl)
    ax.axhline(1/3,color="grey",ls=":",lw=0.8,label="a = 1/3  (Betz)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8c_axial_induction_design_comparison.png")

    # 8d — normal loading
    fig, ax = plt.subplots(figsize=(9,5))
    for lbl,_,_,res,*_ in designs:
        ax.plot(res[:,2],res[:,3]/norm_val,color=_design_color(lbl,n_d),lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8d_normal_loading_design_comparison.png")

    # 8e — angle of attack
    fig, ax = plt.subplots(figsize=(9,5))
    for lbl,_,_,res,*_ in designs:
        ax.plot(res[:,2],res[:,6],color=_design_color(lbl,n_d),lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$\alpha$ [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8e_angle_of_attack_design_comparison.png")

    # 8f — performance bar chart
    a_ad_=0.5*(1.0-np.sqrt(1.0-CT_TARGET))
    cp_ad_=CP_ad if CP_ad is not None else 4.0*a_ad_*(1.0-a_ad_)**2
    labels_b=[d[0] for d in designs]+["Actuator disk"]
    cp_b=[d[5] for d in designs]+[cp_ad_]
    ct_b=[d[4] for d in designs]+[CT_TARGET]
    eff_b=[v/cp_ad_ for v in cp_b]
    bar_cols=[_design_color(d[0],n_d) for d in designs]+["#aaaaaa"]
    x=np.arange(len(labels_b)); w=0.25
    fig, ax = plt.subplots(figsize=(10,5))
    b1=ax.bar(x-w,cp_b,w,label=r"$C_P$",color=[c+"cc" for c in bar_cols])
    b2=ax.bar(x,  ct_b,w,label=r"$C_T$",color=bar_cols,alpha=0.6)
    b3=ax.bar(x+w,eff_b,w,label=r"$C_P/C_{P,\mathrm{AD}}$",color=bar_cols,hatch="//",alpha=0.85)
    for bars in [b1,b2,b3]:
        for bar in bars: bar.set_edgecolor("white"); bar.set_linewidth(0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels_b,rotation=15,ha="right")
    ax.set_ylabel("Coefficient [-]"); ax.legend()
    ax.bar_label(b1,fmt="%.4f",padding=3,fontsize=7.5)
    ax.bar_label(b2,fmt="%.4f",padding=3,fontsize=7.5)
    ax.bar_label(b3,fmt="%.3f",padding=3,fontsize=7.5)
    ax.grid(True,axis="y"); fig.tight_layout()
    save_fig("8f_performance_comparison_all_designs.png")

elif PLOT_8:
    print("  [SKIP] PLOT_8 — requires at least baseline to be run")

# ── 9  Cl and chord ──────────────────────────────────────────────────────────
if PLOT_9 and res_anal is not None and r_anal is not None:
    r_R_am=res_anal[:,2]; cl_am=res_anal[:,8]
    c_am=np.interp(r_R_am,r_anal/Radius,c_anal)

    # 9a — Cl with chord twin axis
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(r_R_am,cl_am,"b-",lw=2,label=r"$C_l$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
    ax2=ax.twinx()
    ax2.plot(r_R_am,c_am,"r--",lw=2,label="Chord [m]")
    ax2.set_ylabel("Chord [m]",color="red"); ax2.tick_params(axis="y",labelcolor="red")
    ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2)
    fig.tight_layout(); save_fig("9a_lift_coefficient_and_chord_vs_rR.png")

    # 9b — circulation proxy
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(r_R_am,cl_am*c_am,color="#2ca02c",lw=2)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l \cdot c$  [m]"); ax.grid(True)
    fig.tight_layout(); save_fig("9b_circulation_proxy_Cl_times_chord_vs_rR.png")

elif PLOT_9:
    print("  [SKIP] PLOT_9 — requires RUN_ANALYTICAL")

# ── 10  Polar ─────────────────────────────────────────────────────────────────
if PLOT_10 and res_anal is not None:
    alpha_opt,cl_opt,cd_opt=find_optimal_alpha()
    alphas_d=np.linspace(polar_alpha[0],polar_alpha[-1],500)
    cl_d=np.interp(alphas_d,polar_alpha,polar_cl)
    cd_d=np.interp(alphas_d,polar_alpha,polar_cd)
    ld_d=cl_d/np.maximum(cd_d,1e-8)
    r_R_am=res_anal[:,2]

    # 10a — Cl/Cd polar
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(polar_cd,polar_cl,"k-",lw=1.5,label="DU95W180 polar")
    sc=ax.scatter(res_anal[:,9],res_anal[:,8],c=r_R_am,cmap="viridis",s=30,zorder=5,
                  label="Analytical opt — operating points")
    plt.colorbar(sc,ax=ax,label="r/R")
    ax.set_xlabel(r"$C_d$ [-]"); ax.set_ylabel(r"$C_l$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("10a_Cl_Cd_polar_with_operating_points.png")

    # 10b — Cl/Cd vs alpha
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(alphas_d,ld_d,"k-",lw=1.5,label=r"$C_l/C_d$")
    ax.axvline(alpha_opt,color="r",ls="--",
               label=rf"$\alpha_{{opt}}={alpha_opt:.1f}°$,  $(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
    ld_ops=(np.interp(res_anal[:,6],polar_alpha,polar_cl)
            /np.maximum(np.interp(res_anal[:,6],polar_alpha,polar_cd),1e-8))
    sc2=ax.scatter(res_anal[:,6],ld_ops,c=r_R_am,cmap="viridis",s=30,zorder=5,label="Operating points")
    plt.colorbar(sc2,ax=ax,label="r/R")
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_l/C_d$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("10b_glide_ratio_vs_alpha_with_operating_points.png")

elif PLOT_10:
    print("  [SKIP] PLOT_10 — requires RUN_ANALYTICAL")

print("\n"+"="*60); print("ALL PLOTS SAVED TO:", save_folder); print("="*60)
print("\nFile list:")
for f in sorted(os.listdir(save_folder)):
    if f.endswith(".png"): print(f"  {f}")

# =============================================================================
# 11.  SAVE RESULTS
# =============================================================================

def save_results(path="full_bem_results.npz"):
    annuli_res={}
    for N in [8,20,100]:
        b=np.linspace(RootLocation_R,TipLocation_R,N+1); rows=[]
        for i in range(N):
            rm=0.5*(b[i]+b[i+1])
            row,_=solve_streamtube(b[i],b[i+1],Omega8,3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows.append(row)
        annuli_res[N]=np.vstack(rows)
    spacing_res={}
    for lbl,b in {"constant":np.linspace(RootLocation_R,TipLocation_R,41),
                  "cosine":RootLocation_R+(TipLocation_R-RootLocation_R)
                           *0.5*(1-np.cos(np.linspace(0,np.pi,41)))}.items():
        rows=[]
        for i in range(40):
            rm=0.5*(b[i]+b[i+1])
            row,_=solve_streamtube(b[i],b[i+1],Omega8,3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows.append(row)
        spacing_res[lbl]=np.vstack(rows)

    # Combine both sweeps for saving — perf sweep takes priority for TSR overlap
    all_tsrs = sorted(set(list(sweep_perf.keys())+list(sweep_span.keys())))
    all_perf = {**sweep_span, **sweep_perf}   # perf sweep overwrites span for same TSR
    all_data = {**sweep_data_span, **sweep_data_perf}

    kw=dict(
        polar_alpha=polar_alpha, polar_cl=polar_cl, polar_cd=polar_cd,
        cfg_Radius=Radius, cfg_NBlades=NBlades, cfg_U0=U0, cfg_rho=rho,
        cfg_RootLocation_R=RootLocation_R, cfg_TipLocation_R=TipLocation_R,
        cfg_Pitch=Pitch, cfg_CHORD_ROOT=CHORD_ROOT, cfg_CHORD_MIN=CHORD_MIN,
        cfg_CT_TARGET=CT_TARGET, cfg_TSR_DESIGN=TSR_DESIGN, cfg_DELTA_R_R=DELTA_R_R,
        # Span sweep
        sweep_tsrs=np.array(TSR_SWEEP_SPAN,dtype=float),
        tsr_CT=np.array([sweep_span[t]["CT"] for t in TSR_SWEEP_SPAN]) if sweep_span else np.array([]),
        tsr_CP=np.array([sweep_span[t]["CP"] for t in TSR_SWEEP_SPAN]) if sweep_span else np.array([]),
        # Performance sweep
        sweep_tsrs_perf=np.array(sorted(sweep_perf.keys()),dtype=float) if sweep_perf else np.array([]),
        tsr_CT_perf=np.array([sweep_perf[t]["CT"] for t in sorted(sweep_perf)]) if sweep_perf else np.array([]),
        tsr_CP_perf=np.array([sweep_perf[t]["CP"] for t in sorted(sweep_perf)]) if sweep_perf else np.array([]),
        # TSR=8 specific
        results_tsr8=results_tsr8 if results_tsr8 is not None else np.array([]),
        res_nc=res_nc if res_nc is not None else np.array([]),
        ct_hist_tsr8=ct_hist_tsr8 if ct_hist_tsr8 is not None else np.array([]),
        F_tsr8=F_tsr8 if F_tsr8 is not None else np.array([]),
        # Section-6
        annuli_N8=annuli_res[8], annuli_N20=annuli_res[20], annuli_N100=annuli_res[100],
        spacing_constant=spacing_res["constant"], spacing_cosine=spacing_res["cosine"],
        # Geometry nodes
        r_base=r_base if r_base is not None else np.array([]),
        c_base=c_base if c_base is not None else np.array([]),
        tw_base=tw_base if tw_base is not None else np.array([]),
        r_anal=r_anal if r_anal is not None else np.array([]),
        c_anal=c_anal if c_anal is not None else np.array([]),
        tw_anal=tw_anal if tw_anal is not None else np.array([]),
        r_cubic=r_cubic if r_cubic is not None else np.array([]),
        c_cubic=c_cubic if c_cubic is not None else np.array([]),
        tw_cubic=tw_cubic if tw_cubic is not None else np.array([]),
        r_qrt=r_qrt if r_qrt is not None else np.array([]),
        c_qrt=c_qrt if c_qrt is not None else np.array([]),
        tw_qrt=tw_qrt if tw_qrt is not None else np.array([]),
        # BEM results
        res_base=res_base if res_base is not None else np.array([]),
        res_anal=res_anal if res_anal is not None else np.array([]),
        res_cubic=res_cubic if res_cubic is not None else np.array([]),
        res_qrt=res_qrt if res_qrt is not None else np.array([]),
        # Scalar performance
        CT_base=CT_base if CT_base is not None else np.nan,
        CP_base=CP_base if CP_base is not None else np.nan,
        CT_anal=CT_anal if CT_anal is not None else np.nan,
        CP_anal=CP_anal if CP_anal is not None else np.nan,
        CT_cubic=CT_cubic if CT_cubic is not None else np.nan,
        CP_cubic=CP_cubic if CP_cubic is not None else np.nan,
        CT_qrt=CT_qrt if CT_qrt is not None else np.nan,
        CP_qrt=CP_qrt if CP_qrt is not None else np.nan,
        CP_ad=CP_ad,
        p_cubic=np.array(p_cubic,dtype=float) if p_cubic is not None else np.array([]),
        p_qrt=np.array(p_qrt,dtype=float) if p_qrt is not None else np.array([]),
    )
    # Per-TSR sweep results
    for TSR in TSR_SWEEP_SPAN:
        if TSR in sweep_data_span: kw[f"sweep_res_{int(TSR)}"]=sweep_data_span[TSR]
    for TSR in sorted(sweep_perf.keys()):
        if TSR in sweep_data_perf: kw[f"sweep_res_perf_{int(TSR)}"]=sweep_data_perf[TSR]

    np.savez(path,**kw)
    sz=sum(v.nbytes for v in kw.values() if hasattr(v,"nbytes"))
    print(f"\nResults saved to: {path}  ({len(kw)} arrays,  {sz//1024} KB)")

if SAVE_RESULTS:
    save_results("full_bem_results.npz")
else:
    print("\nSAVE_RESULTS=False — full_bem_results.npz not written.")