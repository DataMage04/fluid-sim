# Water at 20 °C: grid-independence and performance study

These experiments use the existing `fluid_sim` solver with **liquid water at 20 °C**, treated as incompressible and isothermal.

- Density: 998.2 kg/m³
- Dynamic viscosity: 0.001002 Pa·s
- Kinematic viscosity: 1.003806852e-6 m²/s
- Domain: 2 m × 1 m
- Mean inlet speed: 0.000100380685233 m/s
- Obstacle: x = 0.8–1.0 m, y = 0.3–0.7 m
- Obstacle-height Reynolds number: 40
- Fixed timestep: 25 s
- Total physical duration: 20,000 s (800 steps)

**These are new simulations, not the earlier hypothetical-fluid results.** Water at these very low speeds produces pressure drops on the order of micropascals. The simulated time is not the wall-clock runtime.

## Run from the repository root

Install dependencies:
```powershell
pip install -r requirements.txt
```

First, a smoke test:
```powershell
python studies/grid_performance.py --case cases/water20c_poiseuille.json --meshes 40x20 80x40 --dt 25 --duration 125 --repeats 1 --output results/studies/water20c_smoke
```

Poiseuille grid refinement:
```powershell
python studies/grid_performance.py --case cases/water20c_poiseuille.json --meshes 40x20 80x40 160x80 --repeats 1 --output results/studies/water20c_poiseuille
```

Obstacle grid refinement:
```powershell
python studies/grid_performance.py --case cases/water20c_obstacle.json --meshes 40x20 80x40 160x80 --repeats 1 --output results/studies/water20c_obstacle
```

Temporal sensitivity (fixed 80 × 40 grid, matched 20,000 s):
```powershell
python studies/grid_performance.py --case cases/water20c_poiseuille.json --meshes 80x40 --dt 12.5 --duration 20000 --repeats 1 --output results/studies/water20c_timestep_12p5
```

Repeat each mesh three times for performance estimates after the initial runs by using `--repeats 3`. The solver rebuilds and refactorizes the sparse pressure matrix at every step; fine meshes can be very slow. Do not infer grid independence solely from low divergence or matched inlet/outlet flow. Compare pressure drop and velocity across successive meshes, and separately examine timestep sensitivity.

The study creates `raw_runs.csv`, `summary.csv`, `metadata.json`, plots, and per-run CFD reports under the output folder. Runtime currently includes CFD computation and lightweight report writing; plotting and animations are disabled for the timed runs.

**Return the two `summary.csv` files and figures** for analysis and report-ready convergence discussion.
