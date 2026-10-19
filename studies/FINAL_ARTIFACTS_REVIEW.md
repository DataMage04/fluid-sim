# Final engineering figures — corrected and reviewed

**Updated for the submitted water-at-20°C CFD studies.** This stage is **post-processing only**; it does **not** rerun the CFD solver.

## Regenerate figures

From the repository root:

```powershell
git pull origin main
python -m pytest tests/test_final_verified_report_artifacts.py -q
python studies/final_verified_report_artifacts.py --root results/studies --output results/final_verified_artifacts
```

On completion, the script regenerates the 15 PNGs and the existing `verified_engineering_summary.csv`, and adds **`FIGURE_REVIEW_NOTES.txt`**.

### Corrections made

- **Rectangular obstacle schematic:** only the reference sampling bands (x=0.60 m and x=1.40 m, each width 0.10 m) are drawn, rather than overlapping all three pairs. The near (0.65→1.25 m) and far (0.50→1.50 m) pairs are listed in the caption below the drawing. The 2.00×1.00 m domain, obstacle from x=0.80–1.00 and y=0.30–0.70 m, inlet and dimension annotations remain to scale.
- **Cylinder schematic:** uses a dimensioned full-domain view (2.20×0.41 m) and a separately enlarged view of the DFG-inspired circle (centre 0.20,0.20 m, D=0.10 m). Front and rear pressure probes at x=0.15 and x=0.25 m are labelled on the enlargement, not squeezed into the small cylinder in the main drawing.
- **Cylinder comparison and error charts:** the available 110×41 result is a **single mesh**, so these now use bars rather than a line implying convergence; computed and published coefficients and deviation are shown. This is a DFG-inspired approximation, not exact benchmark reproduction.
- **Pressure-loss station plots:** x-axis now states the actual sampling coordinates for near, reference and far.
- **Runtime summary:** includes the **sum of both clear-channel and obstructed-channel CFD runs** for the matched Re=40 pressure-loss experiment; states different grids and step counts, so it must not be used for standardized scaling claims.
- **Engineering summary:** picks the largest mesh by cell count instead of trusting CSV row order.
- **Numerical consistency checks:** cross-check pressure differences, nondimensional coefficients, analytical pressure-error percentages, published cylinder coefficient deviation and reference station values. An inconsistency raises an error instead of generating misleading plots.

### Interpret the finished figures cautiously

The previous obstacle grid study retains about a **4.31%** pressure-drop change between its two finest meshes. This is not sufficient proof of grid independence. The existing DFG-inspired single-grid cylinder value deviates by about **11.97%** from the published reference. The extra obstruction pressure drop is a **static pressure difference** relative to a matched clear-channel run, not a rigorously determined irreversible total-pressure loss.

The `FIGURE_REVIEW_NOTES.txt` file preserves these qualifications alongside the plots.

### Source inputs

The program requires the existing files in:

- `results/studies/water20c_poiseuille/summary.csv`
- `results/studies/water20c_obstacle/summary.csv`
- `results/studies/pressure_loss_sweep/summary.csv`
- `results/studies/pressure_loss_sweep/station_comparisons.csv`
- `results/studies/benchmark_poiseuille/poiseuille_summary.csv`
- `results/studies/benchmark_cylinder/cylinder_summary.csv`

Run from the repository root so the paths resolve correctly.

The script does not recreate physical flow fields or supply a replacement CFD validation analysis. It checks and plots the metrics already in these files.
