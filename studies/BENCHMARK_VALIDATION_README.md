# Analytical Poiseuille and published CFD benchmark validation

**Working fluid for every new simulation:** water at 20 °C, density 998.2 kg/m³, dynamic viscosity 0.001002 Pa·s, kinematic viscosity approximately 1.003807×10⁻⁶ m²/s.

Two independent forms of validation are included:

**A. Analytical plane Poiseuille:** a 2 m × 1 m channel with parabolic inlet. Compare the numerical streamwise velocity profile against u(y)=6 Umean (y/H)(1−y/H), and pressure drop against 12 μ Umean L/H². Refine the grid and report relative RMSE, pressure-drop relative error, continuity residuals and runtime.

**B. Published cylinder benchmark (DFG 2D-1 inspired):** test flow around a cylinder at Re=20 with domain [0,2.2]×[0,0.41] m and cylinder centre (0.2,0.2) m, radius 0.05 m, parabolic inlet and no-slip top/bottom/cylinder. The cylinder diameter is the Reynolds characteristic length D=0.1 m. For water, Umean = Re μ/(ρ D) = 0.0002007613705 m/s, Umax ≈ 0.0003011421 m/s. The DFG reference used density=1, viscosity ν=0.001, Umean=0.2, and states reference front-to-rear kinematic pressure difference **0.11752016697**, at front (0.15,0.2) and rear (0.25,0.2). Normalize both pressures using q=0.5 ρ Umean²: DFG reference coefficient **5.8760083485**. This corresponds to approximately 0.0001182 Pa with water at the matched Reynolds number. See the [primary DFG / FEATFLOW benchmark specification](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).

## Pull, install, test

```powershell
git pull origin main
pip install -r requirements.txt
python -m pytest tests/test_benchmark_validation.py -q
```

## First run: analytical Poiseuille

Start with one resolution to verify the runner:

```powershell
python studies/benchmark_validation.py --mode poiseuille --meshes 40x20 --dt 10 --duration 1000 --output results/studies/benchmark_poiseuille_smoke
```

Then run refinement using the same physical duration and timestep:

```powershell
python studies/benchmark_validation.py --mode poiseuille --meshes 40x20 80x40 160x80 --dt 10 --duration 12000 --output results/studies/benchmark_poiseuille
```

This is computationally expensive on the finest mesh. The previous uploaded grid study already has independent analytical Poiseuille error data, so you can begin with the cylinder comparison if desired.

## Second run: published cylinder benchmark comparison

Smoke test (does NOT establish quasi-steady accuracy):

```powershell
python studies/benchmark_validation.py --mode cylinder --meshes 110x41 --dt 10 --duration 200 --output results/studies/benchmark_cylinder_smoke
```

Longer initial comparison:

```powershell
python studies/benchmark_validation.py --mode cylinder --meshes 110x41 --dt 10 --duration 12000 --output results/studies/benchmark_cylinder
```

The fine-grid extension, after confirming the baseline completes:

```powershell
python studies/benchmark_validation.py --mode cylinder --meshes 220x82 --dt 5 --duration 12000 --output results/studies/benchmark_cylinder_fine
```

**Important:** The finer run can be much slower because your sparse pressure matrix is currently rebuilt and refactorized at every timestep. You can run it later if the first result is promising. The 12,000-second duration is an initial protocol, not a claim that the velocity and pressure fields have reached a steady solution. Inspect pressure-coefficient history; extend duration to examine convergence. At Umean≈0.0002 m/s, flow traversal across the 2.2 m channel takes roughly 11,000 seconds.

## What is and is not comparable

The benchmark's published quantity is the pressure difference at two exact geometric points on the cylinder surface. Your solver uses a Cartesian cell-centre solid mask: its cylinder is **stair-stepped** rather than boundary-fitted. The script estimates front/rear surface pressure by linearly extrapolating from the two nearest *fluid cells* at the row nearest y=0.2. This is approximate, and is sensitive to grid spacing and point placement. It is not an exact surface boundary evaluation.

The original DFG outflow uses a traction / do-nothing boundary condition; your solver uses an outlet zero-gauge pressure approximation. Your numerical wall treatment around the stair-step circle is not equivalent to a boundary-fitted no-slip implementation. Therefore **report the deviation** from the published DFG pressure reference and identify these model differences, not as a strict solver validation pass/fail. Drag and lift coefficients are not computed and must not be claimed.

The source DFG benchmark uses a fixed Reynolds number and dimensionless comparison. Do **not** compare 0.11752016697 as Pa directly against water; use pressure coefficient and/or the scaled reference. Its documented drag (5.57953523384) and lift (0.010618948146) numbers are *not* computed here.

## Generated outputs

Poiseuille mode: `poiseuille_summary.csv`, each-grid velocity-profile plots, `poiseuille_convergence.png`, and `metadata.json`.

Cylinder mode: `cylinder_summary.csv`, `cylinder_history.csv`, pressure-coefficient histories, `cylinder_comparison.png`, and `metadata.json`.

**Upload those CSV files and PNG plots to GitHub** and return here for quantitative comparison and final report writing.

## Report-ready methods paragraph

An analytical verification was conducted for steady plane Poiseuille flow with water at 20 °C by comparing the numerical streamwise velocity field and pressure gradient with closed-form expressions. A further **published benchmark comparison** was performed using a geometrically similar setup to the DFG 2D-1 steady laminar flow-around-cylinder test at Re=20. The published front-to-rear kinematic pressure difference was converted to a nondimensional pressure difference based on the dynamic pressure, allowing comparison at the same Reynolds number with water properties. The cylinder was represented by a cell-centre solid mask on a Cartesian staggered grid, and surface pressures were extrapolated from nearby fluid cells. The results are interpreted as a **DFG-inspired comparative assessment**, since the Cartesian approximation, outlet boundary condition and pressure-probe method differ from the original reference discretization. (Numerical results should be inserted only after completing the runs.)
