# Quantitative static-pressure-drop comparison — water at 20 °C

This study complements the existing grid-independence results. It uses the **same configured fluid and rectangular obstacle**, comparing two solutions at matched flow rate/velocity:

1. **Clear channel**: same 2 m x 1 m channel, uniform inlet, no internal obstacle.
2. **Obstructed channel**: uniform inlet, rectangle x=0.8–1.0 m and y=0.3–0.7 m.

Fluid at 20 °C: density **998.2 kg/m³**, dynamic viscosity **0.001002 Pa·s**, kinematic viscosity approximately **1.0038×10⁻⁶ m²/s**. The characteristic length is the **0.4 m obstacle height**.

For each Reynolds number, the inlet velocity is computed from `U=Re mu/(rho D)`. Both simulations use the same grid, timestep, elapsed physical time, inlet profile and pressure sampling locations. The script disables GIFs/interactive plots and runs the current solver.

## Pull and verify

```powershell
git pull origin main
pip install -r requirements.txt
python -m pytest tests/test_pressure_loss_comparison.py -q
```

## Start with one Reynolds number (recommended)

```powershell
python studies/pressure_loss_comparison.py --reynolds 40 --nx 80 --ny 40 --dt 25 --duration 20000 --output results/studies/pressure_loss_Re40
```

Then conduct a controlled sweep:

```powershell
python studies/pressure_loss_comparison.py --reynolds 20 40 60 --nx 80 --ny 40 --dt 25 --duration 20000 --output results/studies/pressure_loss_sweep
```

Each Reynolds number requires **two CFD runs**. Based on your grid study, expect several minutes per case, potentially much longer overall. The pressure-drop histories are tracked at 20-step intervals by default; this sampling adds a little post-processing overhead.

If runs are impractically long, use `--duration 5000` for a **diagnostic pilot only**. Such short runs may not reach quasi-steady flow. Do **not** present them as converged steady-state engineering results.

## Measured quantities

At each pair of transverse x-bands (width 0.1 m) outside the obstacle:

| Pair | Upstream x (m) | Downstream x (m) |
|---|---:|---:|
| Near | 0.65 | 1.25 |
| Reference | 0.60 | 1.40 |
| Far | 0.50 | 1.50 |

For every pair, the arithmetic mean static pressure over **fluid** cells in each band is used. The reported quantities are:

- `Δp_clear = p_up,clear − p_down,clear`
- `Δp_obstacle = p_up,obstacle − p_down,obstacle`
- **Excess static pressure drop** `Δp_excess = Δp_obstacle − Δp_clear`
- Dynamic pressure `q = 0.5 rho U²`
- **Excess pressure-drop coefficient** `K_excess = Δp_excess/q`
- Extra hydraulic power per metre of out-of-plane depth `P'_excess = Δp_excess Q'`, where `Q'` is the volumetric flow per metre depth, units m²/s.

This is a comparison of **static-pressure differences**. It is **not automatically an irreversible (total-pressure) loss coefficient**, a drag coefficient, or a friction-factor measurement. To quantify irreversible pressure loss rigorously, include recovery/kinetic-energy flux terms and sufficiently distant stations.

### Important convergence limitations

The study saves a pressure-drop history and the relative range of the **last 25% of recorded excess-loss samples**. A value below 2% is marked `PASS` as a *screening indicator*, not proof of steady flow. Review its plot and, if needed, run longer; at the reference Re40 inlet velocity, the channel advective residence time is approximately 20,000 s, and Re20 takes about 40,000 s. The fixed duration of 20,000 s therefore cannot be presumed sufficient across all Reynolds numbers.

Furthermore, the earlier obstacle study gave a ~4.3% pressure-drop change between 80x40 and 160x80 grids. Quantitative obstacle results at 80x40 are **mesh-sensitive**. Once the time histories settle, rerun the most important condition (Re40) on 160x80, and examine grid and timestep sensitivity before claiming final accuracy. The Poiseuille analytical pressure-drop comparison is **separate** and already addressed in the grid-validation study.

## Output files

The output directory includes:

- `summary.csv`: one row per Reynolds number, flow conditions, pressure differences and coefficients, divergence, flow imbalance, runtime and settling indicator.
- `station_comparisons.csv`: upstream/downstream pressures and added pressure drops for three station choices.
- `pressure_history.csv`: sampled pressure drop versus physical time for each geometry and Reynolds number.
- `pressure_profiles.csv`: streamwise cross-sectional mean **static** pressure (over fluid cells).
- `pressure_profile_Re*.png` and `pressure_history_Re*.png`.
- `pressure_loss_vs_reynolds.png` and `excess_coefficient_vs_reynolds.png`.
- `study_metadata.json`, plus `runs/*/resolved_case.json` and `runs/*/output.txt`.

**Upload `summary.csv`, `station_comparisons.csv`, `pressure_history.csv`, and the figures.** We will compare the actual values, assess pressure recovery and station sensitivity, and write the experimental-method/results sections with proper qualifications.

### Suggested report subsection

**Quantitative Comparison of Static Pressure Losses in Water Flow Around a Rectangular Obstruction.** An isothermal numerical experiment was specified using water at 20 °C (ρ = 998.2 kg/m³, μ = 0.001002 Pa·s). A rectangular obstruction was introduced in a 2 m × 1 m channel and compared against an unobstructed baseline under matched operating conditions. The streamwise pressure difference was evaluated at three pairs of sampling stations, and the obstruction-associated incremental static pressure drop was calculated as the difference between obstructed and baseline pressure drops. Results were nondimensionalized by the inlet dynamic pressure. Transient histories, mass-balance metrics, and sampling-location sensitivity were assessed before interpreting final values. [Insert measured numerical results only after the simulations finish.]
