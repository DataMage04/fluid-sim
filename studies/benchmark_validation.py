"""Analytical Poiseuille and published DFG 2D-1 inspired cylinder benchmarks.
All physical simulations use water at 20 C; run from repository root.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fluid_sim.core.grid import MACGrid
from fluid_sim.numerics.advection import advect_velocity
from fluid_sim.numerics.boundary_conditions import apply_boundary_conditions
from fluid_sim.numerics.diffusion import diffuse_velocity, diffusion_timestep_limit
from fluid_sim.numerics.solver import solve_pressure, project_velocity, divergence_metrics
from fluid_sim.analysis.validation import analytical_poiseuille_profile, numerical_velocity_profile, profile_error
from fluid_sim.analysis.engineering import mean_pressure_in_x_band
from fluid_sim.analysis.visualization import flow_rate_metrics

RHO, MU = 998.2, 0.001002
NU = MU / RHO
DFG_DP = 0.11752016697
# Published DFG uses rho=1, U_mean=.2, and kinematic pressure;
# compare dimensionless dp/(0.5*rho*U_mean**2).
DFG_CDP = DFG_DP / (0.5 * 1.0 * .2**2)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(path, x, curves, xlabel, ylabel):
    fig, ax=plt.subplots(figsize=(8, 5))
    for label, values in curves.items():
        ax.plot(x, values, "o-", markersize=3, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def surface_pressure(grid, side):
    """Two-fluid-cell extrapolation to geometrical front/rear cylinder point."""
    xc,yc,radius=.2,.2,.05
    j=int(np.argmin(abs((np.arange(grid.ny)+.5)*grid.dy-yc)))
    x=(np.arange(grid.nx)+.5)*grid.dx
    xsurface=xc-radius if side=="front" else xc+radius
    indices=np.flatnonzero((~grid.solid[j]) &
        ((x<xsurface-1e-12) if side=="front" else (x>xsurface+1e-12)))
    indices=indices[-2:] if side=="front" else indices[:2]
    if len(indices)!=2:
        raise ValueError("Two external fluid cells required per cylinder side")
    xa,xb=x[indices]
    pa,pb=grid.p[j,indices]
    return float(pa+(pb-pa)*(xsurface-xa)/(xb-xa))


def cylinder_probe(grid, mean_velocity):
    pfront=surface_pressure(grid,"front")
    prear=surface_pressure(grid,"rear")
    dp=pfront-prear
    return dict(p_front_pa=pfront,p_rear_pa=prear,dp_pa=dp,
                dp_coefficient=dp/(.5*RHO*mean_velocity**2))


def integrate(grid, dt, steps, mean_velocity, sample_every, cylinder=False):
    limit=diffusion_timestep_limit(grid,NU)
    if dt>limit:
        raise ValueError(f"dt={dt} exceeds explicit diffusion limit {limit}")
    y=(np.arange(grid.ny)+.5)*grid.dy
    eta=y/grid.height
    grid.u[:]=(6*mean_velocity*eta*(1-eta))[:,None]
    grid.v[:]=0
    history=[]
    start=perf_counter()
    for k in range(1,steps+1):
        for op in range(3):
            if op==0:
                apply_boundary_conditions(grid,mean_velocity,"poiseuille")
                advect_velocity(grid,dt)
            elif op==1:
                apply_boundary_conditions(grid,mean_velocity,"poiseuille")
                diffuse_velocity(grid,dt,NU)
            else:
                apply_boundary_conditions(grid,mean_velocity,"poiseuille")
                solve_pressure(grid,dt,rho=RHO)
                project_velocity(grid,dt,rho=RHO)
        if k==1 or k%sample_every==0 or k==steps:
            row=dict(step=k,physical_time_s=k*dt,
                     rms_divergence_1_s=divergence_metrics(grid)[1])
            if cylinder:
                row.update(cylinder_probe(grid,mean_velocity))
            history.append(row)
        if k%max(1,steps//8)==0 or k==steps:
            print(f"  {k}/{steps} steps",flush=True)
    return history,perf_counter()-start,limit


def run_poiseuille(args,output):
    rows=[]
    for nx,ny in args.meshes:
        print(f"Poiseuille {nx}x{ny}",flush=True)
        g=MACGrid(nx,ny,2.,1.)
        hist,runtime,limit=integrate(g,args.dt,args.steps,args.poiseuille_speed,
                                     args.sample_every)
        y,u=numerical_velocity_profile(g,1.5)
        reference=analytical_poiseuille_profile(y,1.,args.poiseuille_speed)
        err=profile_error(u,reference)
        d=numdp=mean_pressure_in_x_band(g,.25,.1)-mean_pressure_in_x_band(g,1.75,.1)
        exact=12*MU*args.poiseuille_speed*1.5
        rows.append(dict(mesh=f"{nx}x{ny}",cells=nx*ny,dx_m=g.dx,dy_m=g.dy,
            density_kg_m3=RHO,viscosity_pa_s=MU,inlet_mean_speed_m_s=args.poiseuille_speed,
            velocity_rmse_m_s=err["rmse"],velocity_relative_rmse=err["relative_rmse"],
            numerical_delta_p_pa=d,analytical_delta_p_pa=exact,
            pressure_relative_error=abs(d-exact)/abs(exact),
            mass_imbalance_m2_s=flow_rate_metrics(g)["difference"],
            rms_divergence_1_s=divergence_metrics(g)[1],
            runtime_s=runtime,dt_s=args.dt,physical_duration_s=args.dt*args.steps,
            explicit_diffusion_limit_s=limit))
        save_csv(output/"poiseuille_summary.csv",rows)
        figure(output/f"poiseuille_profile_{nx}x{ny}.png",y,
            {"Analytical":reference,"Numerical":u},"Channel height y (m)","Velocity u (m/s)")
    figure(output/"poiseuille_convergence.png",[r["dy_m"] for r in rows],
           {"Profile RMSE (%)":[100*r["velocity_relative_rmse"] for r in rows],
            "Pressure error (%)":[100*r["pressure_relative_error"] for r in rows]},
           "Grid spacing (m)","Relative error (%)")


def run_cylinder(args,output):
    rows,all_history=[],[]
    mean_velocity=20*NU/.1
    for nx,ny in args.meshes:
        print(f"DFG-inspired cylinder {nx}x{ny}, Re20",flush=True)
        g=MACGrid(nx,ny,2.2,.41)
        g.add_circle(.2,.2,.05)
        hist,runtime,limit=integrate(g,args.dt,args.steps,mean_velocity,
                                     args.sample_every,cylinder=True)
        end=cylinder_probe(g,mean_velocity)
        n=max(3,math.ceil(len(hist)*.25))
        last=np.array([r["dp_coefficient"] for r in hist[-n:]])
        variation=float(np.ptp(last)/max(np.mean(abs(last)),1e-40))
        reference_pa=DFG_CDP*.5*RHO*mean_velocity**2
        rows.append(dict(mesh=f"{nx}x{ny}",cells=nx*ny,dx_m=g.dx,dy_m=g.dy,
           solid_cells=int(np.count_nonzero(g.solid)),rho_kg_m3=RHO,mu_pa_s=MU,
           Re_diameter=20,mean_velocity_m_s=mean_velocity,dt_s=args.dt,
           physical_duration_s=args.dt*args.steps,
           numerical_front_rear_dp_pa=end["dp_pa"],
           scaled_reference_dp_pa=reference_pa,
           numerical_dp_coefficient=end["dp_coefficient"],
           published_reference_dp_coefficient=DFG_CDP,
           coefficient_error_percent=100*abs(end["dp_coefficient"]/DFG_CDP-1),
           late_variation_fraction=variation,
           quasi_steady_screening="PASS" if variation<.02 else "NOT_ESTABLISHED",
           mass_imbalance_m2_s=flow_rate_metrics(g)["difference"],
           rms_divergence_1_s=divergence_metrics(g)[1],runtime_s=runtime,
           explicit_diffusion_limit_s=limit))
        for record in hist:
            all_history.append(dict(mesh=f"{nx}x{ny}",**record))
        save_csv(output/"cylinder_summary.csv",rows)
        save_csv(output/"cylinder_history.csv",all_history)
        figure(output/f"cylinder_history_{nx}x{ny}.png",
           [r["physical_time_s"] for r in hist],
           {"Computed coefficient":[r["dp_coefficient"] for r in hist],
            "Published DFG coefficient":[DFG_CDP]*len(hist)},
           "Physical time (s)","Front-to-rear pressure coefficient")
    figure(output/"cylinder_comparison.png",[r["dy_m"] for r in rows],
           {"Computed coefficient":[r["numerical_dp_coefficient"] for r in rows],
            "DFG reference":[DFG_CDP]*len(rows)},
           "Grid spacing dy (m)","Front-to-rear pressure coefficient")


def parse_mesh(spec):
    try:
        nx,ny=(int(p) for p in spec.lower().split("x"))
        if nx<10 or ny<8:
            raise ValueError()
        return nx,ny
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid mesh {spec}") from e


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["poiseuille","cylinder"],required=True)
    p.add_argument("--meshes",nargs="+",type=parse_mesh)
    p.add_argument("--dt",type=float,default=10.)
    p.add_argument("--duration",type=float,default=12000.)
    p.add_argument("--sample-every",type=int,default=20)
    p.add_argument("--poiseuille-speed",type=float,default=.000100380685233)
    p.add_argument("--output")
    args=p.parse_args()
    if args.dt<=0 or args.duration<=0 or args.sample_every<1:
        p.error("dt, duration and sample-every must be positive")
    args.steps=round(args.duration/args.dt)
    if args.steps<5 or not np.isclose(args.dt*args.steps,args.duration,rtol=1e-12):
        p.error("duration must be a whole multiple of dt, at least 5 steps")
    args.meshes=args.meshes or ([(40,20),(80,40),(160,80)]
                                if args.mode=="poiseuille" else [(110,41)])
    output=Path(args.output or f"results/studies/benchmark_{args.mode}")
    output.mkdir(parents=True,exist_ok=True)
    metadata=dict(fluid="water at 20 C",rho_kg_m3=RHO,mu_pa_s=MU,
       dt_s=args.dt,duration_s=args.duration,mode=args.mode,
       meshes=args.meshes,
       published_source="https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html",
       published_kinematic_dp=DFG_DP,published_dp_coefficient=DFG_CDP,
       caveat="Staircase circle; outlet differs; surface pressure extrapolated; drag and lift NOT calculated.")
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    if args.mode=="poiseuille":
        run_poiseuille(args,output)
    else:
        run_cylinder(args,output)
    print("Complete:",output)

if __name__=="__main__":
    main()
