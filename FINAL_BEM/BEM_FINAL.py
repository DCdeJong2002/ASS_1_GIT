"""
BEM_FINAL.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 1

Authors: Douwe de Jong(5313899), Martijn van Leeuwen(5614422)
================================================================
Self-contained script producing all required plots and saving
results to full_bem_results.npz for use with PLOTTING_BEM_FINAL.py.

When running the optimization the computation takes quite long
therefore all results from this file are already saved in:

    1.bem_results.npz (includes BEM results)
    2.opt_results.npz (includes optimized designs)

These can then be plotted with PLOTTING_BEM_FINAL.py without 
re-running the BEM or optimizations.
    
Run:  python BEM_FINAL.py
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
TSR_SWEEP_PERF = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]

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
SAVE_BEM_RESULTS = True   # write bem_results.npz  (sweeps, tip correction, convergence)
SAVE_OPT_RESULTS = True   # write opt_results.npz  (baseline + all optimiser designs)

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

_TSR_3_COLORS   = ["#0000FF", "#2ca02c", "#d62728"]
_TSR_MANY_COLORS = ["#000000","#2ca02c","#d62728","#1f77b4",
                    "#ff7f0e","#9467bd","#8c564b","#e377c2"]

def _tsr_color(idx, n):
    return _TSR_3_COLORS[idx % 3] if n <= 3 else _TSR_MANY_COLORS[idx % len(_TSR_MANY_COLORS)]

_DESIGN_COLORS_3 = {
    "Baseline":     "#000000",   # black
    "Analytical":   "#2ca02c",   # green
    "Cubic poly":   "#FF0000",   # red
    "Quartic poly": "#0000FF",   # blue
}
_DESIGN_COLORS_4 = _DESIGN_COLORS_3.copy()

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
        pen+=3e3*(CT-CT_TARGET)**2
        if quartic: _,_,_,t_curve,_,c2,c3,c4=params; pen+=0.001*t_curve**2+0.0005*(c2**2+c3**2+c4**2) #objective function quartic
        else:       _,_,_,t_curve,_,c2,c3=params;    pen+=0.05*t_curve**2+0.01*(c2**2+c3**2) #objective function cubic
        return -CP+pen
    return obj

def run_poly_optimizer(quartic=False,n_starts=N_STARTS,seed=42):
    rng=np.random.default_rng(seed); obj=_make_objective(quartic); label="quartic" if quartic else "cubic"
    if quartic:
        bounds=[(-5,5),(-25,5),(-10,15),(-20,20),(0.3,2),(-3,3),(-3,3),(-3,3)] #parameter bounds quartic
        x0_nom=np.array([-2,-7,2,0,1,0,0,0],dtype=float)
        def _rand(): return np.array([rng.uniform(-6,6),rng.uniform(-20,0),rng.uniform(-5,10),
                                       rng.uniform(-10,10),rng.uniform(0.3,1.5),
                                       rng.uniform(-3,3),rng.uniform(-3,3),rng.uniform(-3,3)])
    else:
        bounds=[(-5,5),(-25,5),(-10,15),(-20,20),(0.3,2),(-3,3),(-3,3)] #parameter bounds cubic
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

    for vals, ylabel, fname, color in [
            (CT_p, r"$C_T$ [-]", "4_4a_thrust_coefficient_CT_vs_TSR.png", "blue"),
            (CQ_p, r"$C_Q$ [-]", "4_4b_torque_coefficient_CQ_vs_TSR.png", "red"),
            (CP_p, r"$C_P$ [-]", "4_4c_power_coefficient_CP_vs_TSR.png", "green")]:
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(tsr_p, vals, "o-", color=color, lw=2)
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

# ── 6  Annuli sensitivity + spacing study ────────────────────────────────────
#
#  Annuli study: N = [4, 8, 16, 32, 64, 160]  (log-spaced, TSR=8)
#    160 is the production grid (DELTA_R_R=0.005) used as reference.
#    Plots: Cn, Ct, a, alpha vs r/R; CT and CP vs N; tip-region Cn zoom.
#
#  Spacing study: N = 20  
#    Plots: Cn, Ct, a, alpha vs r/R (full span + tip zoom for Cn).
#
if PLOT_6 and results_tsr8 is not None:

    # ── palette ──────────────────────────────────────────────────────────────
    # 6 annuli counts: 
    _ANNULI_N   = [4, 8, 16, 32, 64, 160]
    _ANNULI_COLS = ["#08306b","#2171b5","#6baed6","#bdd7e7","#fd8d3c","#d62728"]

    #_ANNULI_N   = [4, 8, 16, 32, 64, 80, 100, 120, 140, 160]
    #_ANNULI_COLS = ["#08306b","#2171b5","#6baed6","#bdd7e7","#fd8d3c","#d62728", "#9467bd","#8c564b","#e377c2","#7f7f7f"]

    # spacing:
    _SPACING_COLS = {"Constant":"#000000","Cosine":"#2ca02c"}
    N_SPACING = 20   # annuli for spacing comparison

    # ── pre-compute all annuli results ────────────────────────────────────────
    annuli_data = {}   # N -> res array
    for N in _ANNULI_N:
        b=np.linspace(RootLocation_R,TipLocation_R,N+1); rows=[]
        for i in range(N):
            rm=0.5*(b[i]+b[i+1])
            row,_=solve_streamtube(b[i],b[i+1],Omega8,3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows.append(row)
        annuli_data[N]=np.vstack(rows)

    # ── pre-compute CT and CP scalars vs N ────────────────────────────────────
    annuli_CT={}; annuli_CP={}
    for N, res_N in annuli_data.items():
        b=np.linspace(RootLocation_R,TipLocation_R,N+1)
        dr=np.diff(b)*Radius
        annuli_CT[N]=float(np.sum(dr*res_N[:,3]*NBlades/(0.5*U0**2*np.pi*Radius**2)))
        annuli_CP[N]=float(np.sum(dr*res_N[:,4]*res_N[:,2]*NBlades*Radius*Omega8
                                  /(0.5*U0**3*np.pi*Radius**2)))

    # ── pre-compute spacing results ───────────────────────────────────────────
    spacing_data={}
    for lbl, b in {"Constant":np.linspace(RootLocation_R,TipLocation_R,N_SPACING+1),
                   "Cosine":RootLocation_R+(TipLocation_R-RootLocation_R)
                            *0.5*(1-np.cos(np.linspace(0,np.pi,N_SPACING+1)))}.items():
        rows=[]
        for i in range(N_SPACING):
            rm=0.5*(b[i]+b[i+1])
            row,_=solve_streamtube(b[i],b[i+1],Omega8,3.0*(1-rm)+1.0,-(14.0*(1-rm)+Pitch))
            rows.append(row)
        spacing_data[lbl]=np.vstack(rows)

    # ══ ANNULI SENSITIVITY PLOTS ══════════════════════════════════════════════

    # 6a1 — Cn vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for idx,(N,res_N) in enumerate(annuli_data.items()):
        mk="o" if N<=16 else None; ms=4 if N<=16 else None
        ax.plot(res_N[:,2],res_N[:,3]/norm_val,
                color=_ANNULI_COLS[idx],lw=2,marker=mk,markersize=ms,label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a1_Cn_vs_rR_annuli_sensitivity.png")

    # 6a2 — Ct vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for idx,(N,res_N) in enumerate(annuli_data.items()):
        mk="o" if N<=16 else None; ms=4 if N<=16 else None
        ax.plot(res_N[:,2],res_N[:,4]/norm_val,
                color=_ANNULI_COLS[idx],lw=2,marker=mk,markersize=ms,label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a2_Ct_vs_rR_annuli_sensitivity.png")

    # 6a3 — axial induction a vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for idx,(N,res_N) in enumerate(annuli_data.items()):
        mk="o" if N<=16 else None; ms=4 if N<=16 else None
        ax.plot(res_N[:,2],res_N[:,0],
                color=_ANNULI_COLS[idx],lw=2,marker=mk,markersize=ms,label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a3_axial_induction_vs_rR_annuli_sensitivity.png")

    # 6a4 — angle of attack alpha vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for idx,(N,res_N) in enumerate(annuli_data.items()):
        mk="o" if N<=16 else None; ms=4 if N<=16 else None
        ax.plot(res_N[:,2],res_N[:,6],
                color=_ANNULI_COLS[idx],lw=2,marker=mk,markersize=ms,label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$\alpha$ [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a4_alpha_vs_rR_annuli_sensitivity.png")

    # 6a5 — CT vs N (global convergence)
    _N_list = list(annuli_CT.keys())
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(_N_list,[annuli_CT[n] for n in _N_list],"o-",color="#d62728",lw=2)
    ax.axhline(annuli_CT[160],color="k",ls="--",lw=0.8,label=f"N=160 reference ({annuli_CT[160]:.4f})")
    ax.set_xlabel("Number of annuli N"); ax.set_ylabel(r"$C_T$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a5_CT_vs_N_annuli_convergence.png")

    # 6a6 — CP vs N (global convergence)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(_N_list,[annuli_CP[n] for n in _N_list],"o-",color="#2ca02c",lw=2)
    ax.axhline(annuli_CP[160],color="k",ls="--",lw=0.8,label=f"N=160 reference ({annuli_CP[160]:.4f})")
    ax.set_xlabel("Number of annuli N"); ax.set_ylabel(r"$C_P$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a6_CP_vs_N_annuli_convergence.png")

    # 6a7 — Cn tip zoom (r/R > 0.85) showing resolution effect near tip
    fig, ax = plt.subplots(figsize=(8,5))
    for idx,(N,res_N) in enumerate(annuli_data.items()):
        mk="o" if N<=32 else None; ms=4 if N<=32 else None
        ax.plot(res_N[:,2],res_N[:,3]/norm_val,
                color=_ANNULI_COLS[idx],lw=2,marker=mk,markersize=ms,label=f"N={N}")
    ax.set_xlim(0.85,1.01)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a7_Cn_tip_zoom_annuli_sensitivity.png")

    # ══ SPACING COMPARISON PLOTS (N=20) ══════════════════════════════════════

    # 6b1 — Cn vs r/R (full span)
    fig, ax = plt.subplots(figsize=(8,5))
    for lbl,res_sp in spacing_data.items():
        ax.plot(res_sp[:,2],res_sp[:,3]/norm_val,"-o",markersize=5,
                color=_SPACING_COLS[lbl],lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b1_Cn_vs_rR_spacing_comparison.png")

    # 6b2 — Cn tip zoom
    fig, ax = plt.subplots(figsize=(8,5))
    for lbl,res_sp in spacing_data.items():
        ax.plot(res_sp[:,2],res_sp[:,3]/norm_val,"-o",markersize=5,
                color=_SPACING_COLS[lbl],lw=2,label=lbl)
    ax.set_xlim(0.85,1.01)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b2_Cn_tip_zoom_spacing_comparison.png")

    # 6b3 — Ct vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for lbl,res_sp in spacing_data.items():
        ax.plot(res_sp[:,2],res_sp[:,4]/norm_val,"-o",markersize=5,
                color=_SPACING_COLS[lbl],lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b3_Ct_vs_rR_spacing_comparison.png")

    # 6b4 — axial induction a vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for lbl,res_sp in spacing_data.items():
        ax.plot(res_sp[:,2],res_sp[:,0],"-o",markersize=5,
                color=_SPACING_COLS[lbl],lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b4_axial_induction_vs_rR_spacing_comparison.png")

    # 6b5 — angle of attack alpha vs r/R
    fig, ax = plt.subplots(figsize=(8,5))
    for lbl,res_sp in spacing_data.items():
        ax.plot(res_sp[:,2],res_sp[:,6],"-o",markersize=5,
                color=_SPACING_COLS[lbl],lw=2,label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$\alpha$ [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b5_alpha_vs_rR_spacing_comparison.png")

    # ══ ITERATION CONVERGENCE PLOTS ══════════════════════════════════════════

    # 6c — CT convergence history
    n_show=min(60,len(ct_hist_tsr8))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1,len(ct_hist_tsr8)+1),ct_hist_tsr8,"b-",lw=2)
    ax.set_xlim(1,n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$C_T$ [-]")
    ax.grid(True); fig.tight_layout(); save_fig("6c_CT_convergence_history.png")

    # 6d — residuals log scale
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


# ── 7  Stagnation pressure ──────────────────────────────────────────────────────
if PLOT_7 and results_tsr8 is not None:
    r_R8 = results_tsr8[:,2]
    a_R8 = results_tsr8[:,0]

    # Freestream dynamic pressure
    q_inf = 0.5 * rho * U0**2

    # Normalised stagnation pressures
    P0_12 = np.ones(len(r_R8))
    P0_34 = (1.0 - 2.0*a_R8)**2

    # Dimensional
    P0_up = q_inf * P0_12
    P0_down = q_inf * P0_34

    # Small offset to visually separate overlapping curves
    eps = 0.003 * q_inf

    # -------------------------------------------------------
    # LEFT FIGURE: four stations with small offset 
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(r_R8, P0_up, color="#0000FF", lw=2.5,
        label=r"$P_0^{\infty,\uparrow}$ (infinity upwind)")

    ax.plot(r_R8, P0_down, color="#FF0000", lw=2.5,
            label=r"$P_0^{\infty,\downarrow}$ (infinity downwind)")

    ax.plot(r_R8, P0_up+eps, color="#000000", lw=1.8, linestyle="--", alpha=0.95,
            label=r"$P_0^{+}$ (rotor upwind)")

    ax.plot(r_R8, P0_down+eps, color="#00AA00", lw=1.8, linestyle="--", alpha=0.95,
            label=r"$P_0^{-}$ (rotor downwind)")

    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$P_0$ [Pa]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig("7_stagnation_pressure_four_stations.png")

    # -------------------------------------------------------
    # RIGHT FIGURE: 
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(r_R8, P0_12, color="#0000FF", lw=2.5,
            label=r"Upstream $P_0/q_\infty = 1$")

    ax.plot(r_R8, P0_34, color="#FF0000", lw=2.5,
            label=r"Downstream $P_0/q_\infty = (1-2a)^2$")

    # Shade stagnation pressure drop (energy extracted)
    ax.fill_between(
        r_R8,
        P0_34,
        P0_12,
        color="#B0B0B0",
        alpha=0.35,
        label=r"$\Delta P_0$"
    )

    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$P_0/q_\infty$ [-]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig("7_stagnation_pressure_drop.png")

elif PLOT_7:
    print("  [SKIP] PLOT_7 — requires results_tsr8 to be run")


# ── 8  Design comparison ──────────────────────────────────────────────────────
if PLOT_8 and res_base is not None:
    r_R_d=np.linspace(RootLocation_R,TipLocation_R,400)

    # Build designs list 
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

# ── 9  Cl and chord — optimised designs ──────────────────────────────────────
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def _lighten(hex_color, amount=0.45):
    r=int(hex_color[1:3],16)/255; g=int(hex_color[3:5],16)/255; b=int(hex_color[5:7],16)/255
    return f"#{int((r+(1-r)*amount)*255):02x}{int((g+(1-g)*amount)*255):02x}{int((b+(1-b)*amount)*255):02x}"

def _make_9a_axes(ax, r_mid, cl, chord, col, lbl):
    col_chord=_lighten(col,0.45)
    ax.plot(r_mid,cl,    color=col,       lw=2,ls="-", label=rf"{lbl} — $C_l$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
    ax2=ax.twinx()
    ax2.plot(r_mid,chord,color=col_chord,lw=2,ls="--",label=f"{lbl} — chord")
    ax2.set_ylabel("Chord [m]")
    ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=8)

if PLOT_9:
    _opt_data=[]
    for lbl,r_arr,c_arr,res_arr in [
            ("Baseline",    r_base,  c_base,  res_base),
            ("Analytical",  r_anal, c_anal,  res_anal),
            ("Cubic poly",  r_cubic,c_cubic, res_cubic),
            ("Quartic poly",r_qrt,  c_qrt,   res_qrt)]:
        if res_arr is not None and r_arr is not None and res_arr.shape[1]>=9:
            r_mid=res_arr[:,2]; cl=res_arr[:,8]
            chord=np.interp(r_mid,r_arr/Radius,c_arr)
            _opt_data.append((lbl,r_mid,cl,chord))

    if _opt_data:
        # 9a combined — all three optimised designs on one figure
        fig, ax = plt.subplots(figsize=(9,5))
        ax2_comb=ax.twinx()
        for lbl,r_mid,cl,chord in _opt_data:
            col=_design_color(lbl,1); col_chord=_lighten(col,0.45)
            ax.plot(r_mid,cl,    color=col,      lw=2,ls="-", label=rf"{lbl} — $C_l$")
            ax2_comb.plot(r_mid,chord,color=col_chord,lw=2,ls="--",label=f"{lbl} — chord")
        ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
        ax2_comb.set_ylabel("Chord [m]")
        ax.set_zorder(ax2_comb.get_zorder()+1); ax.patch.set_visible(False)
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2_comb.get_legend_handles_labels()
        ax.legend(h1+h2,l1+l2,fontsize=8)
        fig.tight_layout(); save_fig("9a_combined_Cl_and_chord_optimised_designs.png")

        # 9a individual — one file per optimised design
        for lbl,r_mid,cl,chord in _opt_data:
            fig, ax = plt.subplots(figsize=(9,5))
            _make_9a_axes(ax,r_mid,cl,chord,_design_color(lbl,1),lbl)
            fig.tight_layout()
            save_fig("9a_"+lbl.lower().replace(" ","_")+"_Cl_and_chord.png")

        # 9b circulation proxy — all optimised designs overlaid
        fig, ax = plt.subplots(figsize=(9,5))
        for lbl,r_mid,cl,chord in _opt_data:
            ax.plot(r_mid,cl*chord,color=_design_color(lbl,1),lw=2,label=lbl)
        ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l \cdot c$  [m]"); ax.grid(True)
        ax.legend(); fig.tight_layout()
        save_fig("9b_circulation_proxy_Cl_times_chord_optimised_designs.png")
    else:
        print("  [SKIP] PLOT_9 — no optimised design results with Cl data available")

# ── 10  Polar with operating points — all designs + individual optimised ───────
if PLOT_10:
    _pol_all=[]; _pol_opt=[]
    for lbl,res_arr in [("Baseline",  res_base),("Analytical", res_anal),
                         ("Cubic poly",res_cubic),("Quartic poly",res_qrt)]:
        if res_arr is not None and res_arr.shape[1]>=10:
            _pol_all.append((lbl,res_arr))
            if lbl!="Baseline": _pol_opt.append((lbl,res_arr))

    if _pol_all:
        alpha_opt,cl_opt,cd_opt=find_optimal_alpha()
        alphas_d=np.linspace(polar_alpha[0],polar_alpha[-1],500)
        cl_d=np.interp(alphas_d,polar_alpha,polar_cl)
        cd_d=np.interp(alphas_d,polar_alpha,polar_cd)
        ld_d=cl_d/np.maximum(cd_d,1e-8)
        _norm=mcolors.Normalize(vmin=RootLocation_R,vmax=TipLocation_R)
        _cmap=cm.viridis
        _MARKERS={"Baseline":"o","Analytical":"s","Cubic poly":"^","Quartic poly":"D"}

        def _draw_polar(ax,designs):
            ax.plot(polar_cd,polar_cl,"k-",lw=1.5,label="DU95W180 polar",zorder=1)
            for lbl,res_arr in designs:
                sc=ax.scatter(res_arr[:,9],res_arr[:,8],c=res_arr[:,2],
                              cmap=_cmap,norm=_norm,s=30,
                              marker=_MARKERS.get(lbl,"o"),zorder=5,label=lbl)
            plt.colorbar(sc,ax=ax,label="r/R")
            ax.set_xlabel(r"$C_d$ [-]"); ax.set_ylabel(r"$C_l$ [-]")
            ax.legend(fontsize=8); ax.grid(True)

        def _draw_glide(ax,designs):
            ax.plot(alphas_d,ld_d,"k-",lw=1.5,label=r"$C_l/C_d$  DU95W180",zorder=1)
            ax.axvline(alpha_opt,color="k",ls="--",lw=0.8,
                       label=rf"$\alpha_{{opt}}={alpha_opt:.1f}°$,  $(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
            for lbl,res_arr in designs:
                ld_ops=(np.interp(res_arr[:,6],polar_alpha,polar_cl)
                        /np.maximum(np.interp(res_arr[:,6],polar_alpha,polar_cd),1e-8))
                ax.scatter(res_arr[:,6],ld_ops,c=res_arr[:,2],
                           cmap=_cmap,norm=_norm,s=30,
                           marker=_MARKERS.get(lbl,"o"),zorder=5,label=lbl)
            plt.colorbar(ax.collections[-1],ax=ax,label="r/R")
            ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_l/C_d$ [-]")
            ax.legend(fontsize=8); ax.grid(True)

        # 10a combined
        fig,ax=plt.subplots(figsize=(9,5)); _draw_polar(ax,_pol_all)
        fig.tight_layout(); save_fig("10a_combined_Cl_Cd_polar_all_designs.png")
        # 10a individual
        for lbl,res_arr in _pol_opt:
            fig,ax=plt.subplots(figsize=(9,5)); _draw_polar(ax,[(lbl,res_arr)])
            fig.tight_layout()
            save_fig("10a_"+lbl.lower().replace(" ","_")+"_Cl_Cd_polar.png")
        # 10b combined
        fig,ax=plt.subplots(figsize=(9,5)); _draw_glide(ax,_pol_all)
        fig.tight_layout(); save_fig("10b_combined_glide_ratio_vs_alpha_all_designs.png")
        # 10b individual
        for lbl,res_arr in _pol_opt:
            fig,ax=plt.subplots(figsize=(9,5)); _draw_glide(ax,[(lbl,res_arr)])
            fig.tight_layout()
            save_fig("10b_"+lbl.lower().replace(" ","_")+"_glide_ratio_vs_alpha.png")
    else:
        print("  [SKIP] PLOT_10 — no design BEM results with Cd data available")

print("\n"+"="*60); print("ALL PLOTS SAVED TO:", save_folder); print("="*60)
print("\nFile list:")
for f in sorted(os.listdir(save_folder)):
    if f.endswith(".png"): print(f"  {f}")

# =============================================================================
# 11.  SAVE RESULTS
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
BEM_RESULTS_PATH = os.path.join(_HERE, "bem_results.npz")
OPT_RESULTS_PATH = os.path.join(_HERE, "opt_results.npz")


def save_bem_results(path=BEM_RESULTS_PATH):
    """
    Save BEM sweep and tip-correction results.

    Contents
    --------
    Polar data, configuration scalars,
    spanwise TSR sweep (TSR_SWEEP_SPAN),
    performance TSR sweep (TSR_SWEEP_PERF),
    TSR=8 specific arrays (results_tsr8, res_nc, ct_hist_tsr8, F_tsr8),
    section-6 pre-computed annuli/spacing arrays,
    baseline geometry and BEM result.
    """
    # Section-6: annuli sensitivity (N = 4, 8, 16, 32, 64, 160) and
    #            spacing comparison (N = 20, constant vs cosine)
    _ANNULI_N_SAVE = [4, 8, 16, 32, 64, 160]
    annuli_res = {}
    annuli_CT_save = {}
    annuli_CP_save = {}
    for N in _ANNULI_N_SAVE:
        b = np.linspace(RootLocation_R, TipLocation_R, N+1); rows = []
        for i in range(N):
            rm = 0.5*(b[i]+b[i+1])
            row, _ = solve_streamtube(b[i], b[i+1], Omega8,
                                      3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows.append(row)
        res_N = np.vstack(rows)
        annuli_res[N] = res_N
        dr = np.diff(b)*Radius
        annuli_CT_save[N] = float(np.sum(dr*res_N[:,3]*NBlades/(0.5*U0**2*np.pi*Radius**2)))
        annuli_CP_save[N] = float(np.sum(dr*res_N[:,4]*res_N[:,2]*NBlades*Radius*Omega8
                                         /(0.5*U0**3*np.pi*Radius**2)))

    N_SPACING_SAVE = 20
    spacing_res = {}
    for lbl, b in {"constant": np.linspace(RootLocation_R, TipLocation_R, N_SPACING_SAVE+1),
                   "cosine":   RootLocation_R + (TipLocation_R-RootLocation_R)
                               * 0.5*(1-np.cos(np.linspace(0, np.pi, N_SPACING_SAVE+1)))}.items():
        rows = []
        for i in range(N_SPACING_SAVE):
            rm = 0.5*(b[i]+b[i+1])
            row, _ = solve_streamtube(b[i], b[i+1], Omega8,
                                      3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows.append(row)
        spacing_res[lbl] = np.vstack(rows)

    kw = dict(
        # Polar
        polar_alpha=polar_alpha, polar_cl=polar_cl, polar_cd=polar_cd,
        # Config
        cfg_Radius=Radius, cfg_NBlades=NBlades, cfg_U0=U0, cfg_rho=rho,
        cfg_RootLocation_R=RootLocation_R, cfg_TipLocation_R=TipLocation_R,
        cfg_Pitch=Pitch, cfg_CHORD_ROOT=CHORD_ROOT, cfg_CHORD_MIN=CHORD_MIN,
        cfg_CT_TARGET=CT_TARGET, cfg_TSR_DESIGN=TSR_DESIGN, cfg_DELTA_R_R=DELTA_R_R,
        # Spanwise sweep
        sweep_tsrs=np.array(TSR_SWEEP_SPAN, dtype=float),
        tsr_CT=np.array([sweep_span[t]["CT"] for t in TSR_SWEEP_SPAN]) if sweep_span else np.array([]),
        tsr_CP=np.array([sweep_span[t]["CP"] for t in TSR_SWEEP_SPAN]) if sweep_span else np.array([]),
        # Performance sweep
        sweep_tsrs_perf=np.array(sorted(sweep_perf.keys()), dtype=float) if sweep_perf else np.array([]),
        tsr_CT_perf=np.array([sweep_perf[t]["CT"] for t in sorted(sweep_perf)]) if sweep_perf else np.array([]),
        tsr_CP_perf=np.array([sweep_perf[t]["CP"] for t in sorted(sweep_perf)]) if sweep_perf else np.array([]),
        # TSR=8 specific
        results_tsr8=results_tsr8  if results_tsr8  is not None else np.array([]),
        res_nc       =res_nc       if res_nc        is not None else np.array([]),
        ct_hist_tsr8 =ct_hist_tsr8 if ct_hist_tsr8  is not None else np.array([]),
        F_tsr8       =F_tsr8       if F_tsr8        is not None else np.array([]),
        # Section-6 annuli sensitivity
        annuli_N_list =np.array(_ANNULI_N_SAVE, dtype=float),
        annuli_CT_list=np.array([annuli_CT_save[n] for n in _ANNULI_N_SAVE]),
        annuli_CP_list=np.array([annuli_CP_save[n] for n in _ANNULI_N_SAVE]),
        annuli_N4  =annuli_res[4],
        annuli_N8  =annuli_res[8],
        annuli_N16 =annuli_res[16],
        annuli_N32 =annuli_res[32],
        annuli_N64 =annuli_res[64],
        annuli_N160=annuli_res[160],
        # Section-6 spacing comparison (N=20)
        spacing_N=np.array([N_SPACING_SAVE], dtype=float),
        spacing_constant=spacing_res["constant"],
        spacing_cosine  =spacing_res["cosine"],
        # Baseline geometry and BEM result (needed by PLOT_8 comparisons)
        r_base =r_base  if r_base  is not None else np.array([]),
        c_base =c_base  if c_base  is not None else np.array([]),
        tw_base=tw_base if tw_base is not None else np.array([]),
        res_base=res_base if res_base is not None else np.array([]),
        CT_base=CT_base if CT_base is not None else np.nan,
        CP_base=CP_base if CP_base is not None else np.nan,
    )
    # Per-TSR spanwise results
    for TSR in TSR_SWEEP_SPAN:
        if TSR in sweep_data_span:
            kw[f"sweep_res_{int(TSR)}"] = sweep_data_span[TSR]
    # Per-TSR performance results
    for TSR in sorted(sweep_perf.keys()):
        if TSR in sweep_data_perf:
            kw[f"sweep_res_perf_{float(TSR)}"] = sweep_data_perf[TSR]

    np.savez(path, **kw)
    sz = sum(v.nbytes for v in kw.values() if hasattr(v, "nbytes"))
    print(f"\nBEM results saved  → {path}  ({len(kw)} arrays,  {sz//1024} KB)")


def save_opt_results(path=OPT_RESULTS_PATH):
    """
    Save optimisation results (baseline + all optimised designs).

    Contents
    --------
    Polar data, configuration scalars,
    baseline geometry + BEM result,
    analytical, cubic, and quartic designs (geometry nodes + BEM results
    + scalar CT/CP + polynomial parameters),
    actuator-disk reference CP_ad.
    """
    kw = dict(
        # Polar (needed by plot_results.py for PLOT_9/10 without bem_results.npz)
        polar_alpha=polar_alpha, polar_cl=polar_cl, polar_cd=polar_cd,
        # Config
        cfg_Radius=Radius, cfg_NBlades=NBlades, cfg_U0=U0, cfg_rho=rho,
        cfg_RootLocation_R=RootLocation_R, cfg_TipLocation_R=TipLocation_R,
        cfg_Pitch=Pitch, cfg_CHORD_ROOT=CHORD_ROOT, cfg_CHORD_MIN=CHORD_MIN,
        cfg_CT_TARGET=CT_TARGET, cfg_TSR_DESIGN=TSR_DESIGN, cfg_DELTA_R_R=DELTA_R_R,
        # Baseline
        r_base =r_base  if r_base  is not None else np.array([]),
        c_base =c_base  if c_base  is not None else np.array([]),
        tw_base=tw_base if tw_base is not None else np.array([]),
        res_base=res_base if res_base is not None else np.array([]),
        CT_base=CT_base if CT_base is not None else np.nan,
        CP_base=CP_base if CP_base is not None else np.nan,
        # Analytical
        r_anal =r_anal  if r_anal  is not None else np.array([]),
        c_anal =c_anal  if c_anal  is not None else np.array([]),
        tw_anal=tw_anal if tw_anal is not None else np.array([]),
        res_anal=res_anal if res_anal is not None else np.array([]),
        CT_anal=CT_anal if CT_anal is not None else np.nan,
        CP_anal=CP_anal if CP_anal is not None else np.nan,
        # Cubic polynomial
        r_cubic =r_cubic  if r_cubic  is not None else np.array([]),
        c_cubic =c_cubic  if c_cubic  is not None else np.array([]),
        tw_cubic=tw_cubic if tw_cubic is not None else np.array([]),
        res_cubic=res_cubic if res_cubic is not None else np.array([]),
        CT_cubic=CT_cubic if CT_cubic is not None else np.nan,
        CP_cubic=CP_cubic if CP_cubic is not None else np.nan,
        p_cubic=np.array(p_cubic, dtype=float) if p_cubic is not None else np.array([]),
        # Quartic polynomial
        r_qrt =r_qrt  if r_qrt  is not None else np.array([]),
        c_qrt =c_qrt  if c_qrt  is not None else np.array([]),
        tw_qrt=tw_qrt if tw_qrt is not None else np.array([]),
        res_qrt=res_qrt if res_qrt is not None else np.array([]),
        CT_qrt=CT_qrt if CT_qrt is not None else np.nan,
        CP_qrt=CP_qrt if CP_qrt is not None else np.nan,
        p_qrt=np.array(p_qrt, dtype=float) if p_qrt is not None else np.array([]),
        # Actuator disk reference
        CP_ad=CP_ad,
    )
    np.savez(path, **kw)
    sz = sum(v.nbytes for v in kw.values() if hasattr(v, "nbytes"))
    print(f"Opt results saved  → {path}  ({len(kw)} arrays,  {sz//1024} KB)")


if SAVE_BEM_RESULTS:
    save_bem_results()
else:
    print("\nSAVE_BEM_RESULTS=False — bem_results.npz not written.")

if SAVE_OPT_RESULTS:
    save_opt_results()
else:
    print("SAVE_OPT_RESULTS=False — opt_results.npz not written.")




#Added later for chord cl analysis
def plot_9a_baseline_pitch_sweep(
    pitch_values,
    tsr=8.0,
    figsize=(9, 5)
):
    """
    Create a Plot-9a-style figure for the baseline blade geometry at a fixed TSR,
    but for multiple pitch values.

    The plot shows:
      - C_l on the left y-axis
      - chord on the right y-axis

    Parameters
    ----------
    pitch_values : list or array-like
        Pitch values in degrees, e.g. [-4, -2, 0, 2]
    tsr : float, optional
        Tip-speed ratio at which to run the BEM evaluation.
    figsize : tuple, optional
        Matplotlib figure size.
    """

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    for i, pitch_val in enumerate(pitch_values):
        # Baseline geometry with modified pitch
        bins = make_bins()
        r_nodes = bins * Radius
        c_nodes = 3.0 * (1.0 - bins) + 1.0
        tw_nodes = -(14.0 * (1.0 - bins) + pitch_val)

        # Evaluate rotor at requested TSR
        CT_tmp, CP_tmp, res_tmp = evaluate_rotor(r_nodes, c_nodes, tw_nodes, tsr=tsr)

        # Mid-span locations and local Cl
        r_mid = res_tmp[:, 2]
        cl_local = res_tmp[:, 8]

        # Interpolate nodal chord onto annulus midpoints
        chord_mid = np.interp(r_mid, r_nodes / Radius, c_nodes)

        # Colors
        col = _tsr_color(i, len(pitch_values))
        col_chord = _lighten(col, 0.45)

        # Plot Cl and chord
        ax1.plot(
            r_mid, cl_local,
            color=col, lw=2,
            label=rf"$\beta={pitch_val:.1f}^\circ$ — $C_l$"
        )
        ax2.plot(
            r_mid, chord_mid,
            color=col_chord, lw=2, ls="--",
            label=rf"$\beta={pitch_val:.1f}^\circ$ — chord"
        )

        print(
            f"Pitch = {pitch_val:+.2f} deg  |  "
            f"TSR = {tsr:.2f}  |  "
            f"CT = {CT_tmp:.5f}  |  "
            f"CP = {CP_tmp:.5f}"
        )

    ax1.set_xlabel("r/R")
    ax1.set_ylabel(r"$C_l$ [-]")
    ax2.set_ylabel("Chord [m]")
    ax1.grid(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    fig.tight_layout()
    plt.show()

if PLOT_7:
    plot_9a_baseline_pitch_sweep(pitch_values=[-6, -4, -2, 0, 2], tsr=8.0)