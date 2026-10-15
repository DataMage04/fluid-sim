from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def compute_divergence(grid):
    """
    Calculate divergence at the centre of each cell.

    Positive divergence: more fluid leaves than enters.
    Negative divergence: more fluid enters than leaves.
    Zero divergence: inflow and outflow balance.
    """
    divergence = (
        (grid.u[:, 1:] - grid.u[:, :-1]) / grid.dx
        + (grid.v[1:, :] - grid.v[:-1, :]) / grid.dy
    )

    # We do not solve fluid equations inside solid cells.
    divergence[grid.solid] = 0.0

    return divergence

def solve_pressure(
    grid,
    dt,
    rho=1.0,
    iterations=1,
    tolerance=1e-6,
    omega=1.0,
):
    """
    Solve the discrete pressure Poisson equation directly.

    Fluid cells are pressure unknowns. Solid faces use a
    zero-normal-pressure-gradient approximation, and the
    right outlet has zero gauge pressure.

    The iterations and omega arguments remain for compatibility
    with the existing main.py call.
    """
    divergence = compute_divergence(grid)
    rhs = (rho / dt) * divergence

    fluid = ~grid.solid
    cell_ids = -np.ones((grid.ny, grid.nx), dtype=int)
    cell_ids[fluid] = np.arange(np.count_nonzero(fluid))

    n = np.count_nonzero(fluid)

    if n == 0:
        raise ValueError("The grid contains no fluid cells.")

    A = lil_matrix((n, n), dtype=float)
    b = -rhs[fluid].copy()

    dx2 = grid.dx**2
    dy2 = grid.dy**2

    for j in range(grid.ny):
        for i in range(grid.nx):
            if not fluid[j, i]:
                continue

            row = cell_ids[j, i]
            diagonal = 0.0

            # Left fluid neighbour: Neumann at inlet/solid.
            if i > 0 and fluid[j, i - 1]:
                col = cell_ids[j, i - 1]
                c = 1.0 / dx2
                A[row, col] = -c
                diagonal += c

            # Right fluid neighbour, or pressure outlet.
            if i < grid.nx - 1:
                if fluid[j, i + 1]:
                    col = cell_ids[j, i + 1]
                    c = 1.0 / dx2
                    A[row, col] = -c
                    diagonal += c
            else:
                # Dirichlet outlet pressure p = 0.
                diagonal += 1.0 / dx2

            # Previous row.
            if j > 0 and fluid[j - 1, i]:
                col = cell_ids[j - 1, i]
                c = 1.0 / dy2
                A[row, col] = -c
                diagonal += c

            # Next row.
            if j < grid.ny - 1 and fluid[j + 1, i]:
                col = cell_ids[j + 1, i]
                c = 1.0 / dy2
                A[row, col] = -c
                diagonal += c

            A[row, row] = diagonal

    old_pressure = grid.p.copy()

    # A represents -Laplacian(p), so A p = -rhs.
    pressure_solution = spsolve(A.tocsr(), b)

    grid.p[fluid] = pressure_solution
    grid.p[grid.solid] = 0.0

    max_change = np.max(np.abs(grid.p - old_pressure))

    return 1, float(max_change)


def compute_divergence(grid):
    divergence = (
        (grid.u[:, 1:] - grid.u[:, :-1]) / grid.dx
        + (grid.v[1:, :] - grid.v[:-1, :]) / grid.dy
    )

    # Only fluid cells are relevant to the measurement.
    divergence[grid.solid] = 0.0

    return divergence


def divergence_metrics(grid):
    divergence = compute_divergence(grid)
    fluid_divergence = divergence[~grid.solid]

    if fluid_divergence.size == 0:
        return 0.0, 0.0

    max_div = float(np.max(np.abs(fluid_divergence)))
    rms_div = float(np.sqrt(np.mean(fluid_divergence**2)))

    return max_div, rms_div


def project_velocity(grid, dt, rho=1.0):
    """
    Correct face velocities using the pressure gradient.

    Assumes pressure has already been solved.
    """
    # Correct horizontal velocity u.
    # u[j, i] lies between cells (j, i-1) and (j, i).
    for j in range(grid.ny):
        for i in range(1, grid.nx):
            left_fluid = not grid.solid[j, i - 1]
            right_fluid = not grid.solid[j, i]

            if left_fluid and right_fluid:
                dp_dx = (
                    grid.p[j, i] - grid.p[j, i - 1]
                ) / grid.dx

                grid.u[j, i] -= (dt / rho) * dp_dx
            else:
                grid.u[j, i] = 0.0

    # Correct horizontal velocity at the outlet.
    # The outlet pressure is assumed to be zero gauge pressure.
    for j in range(grid.ny):
        if not grid.solid[j, grid.nx - 1]:
            dp_dx = (
                0.0 - grid.p[j, grid.nx - 1]
            ) / grid.dx

            grid.u[j, -1] -= (dt / rho) * dp_dx
        else:
            grid.u[j, -1] = 0.0

    # Correct vertical velocity v.
    # v[j, i] lies between cells (j-1, i) and (j, i).
    for j in range(1, grid.ny):
        for i in range(grid.nx):
            upper_fluid = not grid.solid[j - 1, i]
            lower_fluid = not grid.solid[j, i]

            if upper_fluid and lower_fluid:
                dp_dy = (
                    grid.p[j, i] - grid.p[j - 1, i]
                ) / grid.dy

                grid.v[j, i] -= (dt / rho) * dp_dy
            else:
                grid.v[j, i] = 0.0

    # Preserve the impermeable top and bottom boundaries.
    grid.v[0, :] = 0.0
    grid.v[-1, :] = 0.0