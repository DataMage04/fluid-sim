import numpy as np
import matplotlib.pyplot as plt

def analytical_poiseuille_profile(y, height, mean_velocity):
    """
    Fully-developed laminar flow between stationary parallel plates.

    The profile is scaled so that its cross-sectional mean velocity
    equals mean_velocity.
    """

    eta = y / height

    return (
        6.0
        * mean_velocity
        * eta
        * (1.0 - eta)
    )

def numerical_velocity_profile(grid, x_location):
    """
    Extract cell-centred horizontal velocity near x_location.
    """

    u_center = 0.5 * (
        grid.u[:, :-1]
        + grid.u[:, 1:]
    )

    column = int(x_location / grid.dx)

    column = max(
        0,
        min(grid.nx - 1, column)
    )

    y = (
        np.arange(grid.ny) + 0.5
    ) * grid.dy

    u = u_center[:, column]

    return y, u

def profile_error(
    numerical,
    analytical
):
    difference = numerical - analytical

    rmse = np.sqrt(
        np.mean(difference**2)
    )

    max_error = np.max(
        np.abs(difference)
    )

    reference = np.sqrt(
        np.mean(analytical**2)
    )

    relative_rmse = (
        rmse / reference
        if reference > 0.0
        else 0.0
    )

    return {
        "rmse": rmse,
        "max_error": max_error,
        "relative_rmse": relative_rmse,
    }


def validate_poiseuille(
    grid,
    mean_velocity,
    x_location=1.5,
    filename="poiseuille_validation.png",
):
    y, u_numerical = numerical_velocity_profile(
        grid,
        x_location
    )

    u_analytical = analytical_poiseuille_profile(
        y,
        grid.height,
        mean_velocity
    )

    errors = profile_error(
        u_numerical,
        u_analytical
    )

    print()
    print("POISEUILLE VALIDATION")
    print("----------------------")

    print(
        f"Profile location:      "
        f"x = {x_location:.3f} m"
    )

    print(
        f"Numerical max speed:   "
        f"{np.max(u_numerical):.6f} m/s"
    )

    print(
        f"Analytical max speed:  "
        f"{np.max(u_analytical):.6f} m/s"
    )

    print(
        f"RMSE:                  "
        f"{errors['rmse']:.6e} m/s"
    )

    print(
        f"Maximum error:         "
        f"{errors['max_error']:.6e} m/s"
    )

    print(
        f"Relative RMSE:         "
        f"{100.0 * errors['relative_rmse']:.3f} %"
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        u_analytical,
        y,
        label="Analytical"
    )

    ax.plot(
        u_numerical,
        y,
        "o",
        markersize=4,
        label="Numerical"
    )

    ax.set_xlabel(
        "Horizontal velocity, u (m/s)"
    )

    ax.set_ylabel(
        "Channel height, y (m)"
    )

    ax.set_title(
        "Plane Poiseuille Flow Validation"
    )

    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=150
    )

    plt.close(fig)

    return errors



