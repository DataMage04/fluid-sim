import numpy as np


class MACGrid:
    def __init__(
        self,
        nx=100,
        ny=50,
        width=2.0,
        height=1.0,
    ):
        # Number of cells
        self.nx = nx
        self.ny = ny

        # Physical domain dimensions (metres)
        self.width = width
        self.height = height

        # Grid spacing
        self.dx = width / nx
        self.dy = height / ny

        # Pressure at cell centres
        self.p = np.zeros((ny, nx), dtype=float)

        # Horizontal velocity at vertical faces
        self.u = np.zeros((ny, nx + 1), dtype=float)

        # Vertical velocity at horizontal faces
        self.v = np.zeros((ny + 1, nx), dtype=float)

        # True for solid cells, False for fluid cells
        self.solid = np.zeros((ny, nx), dtype=bool)

    def add_rectangle(
        self,
        x_min,
        x_max,
        y_min,
        y_max,
    ):
        """Add a rectangular solid obstruction.

        Coordinates are specified in metres.
        """

        i_start = max(0, int(round(x_min / self.dx)))
        i_end = min(self.nx, int(round(x_max / self.dx)))

        j_start = max(0, int(round(y_min / self.dy)))
        j_end = min(self.ny, int(round(y_max / self.dy)))

        if i_start >= i_end or j_start >= j_end:
            raise ValueError(
                "The rectangle does not overlap the grid."
            )

        self.solid[j_start:j_end, i_start:i_end] = True


    def add_circle(self, x_center, y_center, radius):
        """Mark pressure cells inside a circle as solid.

        Cell-centre rasterisation gives a staircase representation of the
        surface; it is NOT a boundary-fitted circular mesh. Convergence of
        surface-sensitive benchmark quantities must be checked separately.
        """
        if radius <= 0.0:
            raise ValueError("Circle radius must be positive.")
        if not (0.0 < x_center - radius and x_center + radius < self.width
                and 0.0 < y_center - radius and y_center + radius < self.height):
            raise ValueError("The circle must be strictly inside the channel.")
        x, y = self.cell_centres()
        mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
        if not np.any(mask):
            raise ValueError("The grid is too coarse to represent the circle.")
        self.solid |= mask

    def cell_centres(self):
        """Return the x and y coordinates of pressure cells."""

        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy

        return np.meshgrid(x, y)

    def u_coordinates(self):
        """Coordinates of horizontal velocity samples."""

        x = np.arange(self.nx + 1) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy

        return np.meshgrid(x, y)

    def v_coordinates(self):
        """Coordinates of vertical velocity samples."""

        x = (np.arange(self.nx) + 0.5) * self.dx
        y = np.arange(self.ny + 1) * self.dy

        return np.meshgrid(x, y)
