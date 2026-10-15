import numpy as np

from fluid_sim.analysis.engineering_plots import plot_streamwise_pressure_profile


def test_pressure_profile_plot_smoke(tmp_path):
    output = tmp_path / "pressure_profile.png"
    x = np.array([0.25, 0.75, 1.25, 1.75])
    pressure = np.array([2.0, 1.5, 0.8, 0.2])
    cp = pressure / 0.5

    plot_streamwise_pressure_profile(
        x,
        pressure,
        cp,
        upstream_x=0.25,
        downstream_x=1.75,
        filename=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
