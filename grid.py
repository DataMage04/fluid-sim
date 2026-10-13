
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

        i_start = max(0, int(x_min / self.dx))
        i_end = min(self.nx, int(x_max / self.dx))

        j_start = max(0, int(y_min / self.dy))
        j_end = min(self.ny, int(y_max / self.dy))

        if i_start >= i_end or j_start >= j_end:
            raise ValueError(
                "The rectangle does not overlap the grid."
            )

        self.solid[j_start:j_end, i_start:i_end] = True

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
