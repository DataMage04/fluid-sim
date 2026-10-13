
import numpy as np


def bilinear_sample(field, gx, gy):
    """Sample a 2D array at fractional grid coordinates.

    gx is the column coordinate.
    gy is the row coordinate.
    """

    ny, nx = field.shape

    # Keep samples inside the available array.
    gx = np.clip(gx, 0.0, nx - 1.0)
    gy = np.clip(gy, 0.0, ny - 1.0)

    x0 = int(np.floor(gx))
    y0 = int(np.floor(gy))

    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)

    sx = gx - x0
    sy = gy - y0

    # Interpolate horizontally, then vertically.
    bottom = (
        (1.0 - sx) * field[y0, x0]
        + sx * field[y0, x1]
    )

    top = (
        (1.0 - sx) * field[y1, x0]
        + sx * field[y1, x1]
    )

    return (1.0 - sy) * bottom + sy * top


def sample_velocity(grid, x, y):
    """Sample u and v at physical coordinates x, y.

    The u and v components live on different grid locations.
    """

    # u: x = i*dx, y = (j + 0.5)*dy
    u_gx = x / grid.dx
    u_gy = y / grid.dy - 0.5

    # v: x = (i + 0.5)*dx, y = j*dy
    v_gx = x / grid.dx - 0.5
    v_gy = y / grid.dy

    u_value = bilinear_sample(grid.u, u_gx, u_gy)
    v_value = bilinear_sample(grid.v, v_gx, v_gy)

    return u_value, v_value


def advect_velocity(grid, dt):
    """Semi-Lagrangian advection of MAC velocity components."""

    # Use the old velocity field for every backtrace.
    u_old = grid.u.copy()
    v_old = grid.v.copy()

    # Temporary grid-like object with old velocities.
    class OldVelocity:
        pass

    old = OldVelocity()
    old.u = u_old
    old.v = v_old
    old.dx = grid.dx
    old.dy = grid.dy

    # -------------------------------------------------
    # Advect u: horizontal velocity at vertical faces.
    # -------------------------------------------------

    i, j = np.meshgrid(
        np.arange(grid.nx + 1),
        np.arange(grid.ny),
    )

    x = i * grid.dx
    y = (j + 0.5) * grid.dy

    u_new = np.empty_like(u_old)

    for row in range(grid.ny):
        for col in range(grid.nx + 1):
            # Backtrace using velocity at the destination.
            ux, vy = sample_velocity(
                old, x[row, col], y[row, col]
            )

            x_depart = x[row, col] - dt * ux
            y_depart = y[row, col] - dt * vy

            # Keep the departure point in the domain.
            x_depart = np.clip(
                x_depart, 0.0, grid.width
            )
            y_depart = np.clip(
                y_depart, 0.0, grid.height
            )

            u_new[row, col] = bilinear_sample(
                u_old,
                x_depart / grid.dx,
                y_depart / grid.dy - 0.5,
            )

    # -------------------------------------------------
    # Advect v: vertical velocity at horizontal faces.
    # -------------------------------------------------

    i, j = np.meshgrid(
        np.arange(grid.nx),
        np.arange(grid.ny + 1),
    )

    x = (i + 0.5) * grid.dx
    y = j * grid.dy

    v_new = np.empty_like(v_old)

    for row in range(grid.ny + 1):
        for col in range(grid.nx):
            ux, vy = sample_velocity(
                old, x[row, col], y[row, col]
            )

            x_depart = x[row, col] - dt * ux
            y_depart = y[row, col] - dt * vy

            x_depart = np.clip(
                x_depart, 0.0, grid.width
            )
            y_depart = np.clip(
                y_depart, 0.0, grid.height
            )

            v_new[row, col] = bilinear_sample(
                v_old,
                x_depart / grid.dx - 0.5,
                y_depart / grid.dy,
            )

    grid.u[:, :] = u_new
    grid.v[:, :] = v_new
