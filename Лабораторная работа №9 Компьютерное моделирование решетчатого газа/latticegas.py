import numpy as np


class LatticeGas:

    def __init__(self, parametrs: dict, obstacle: dict):

        self.nx = parametrs['nx']
        self.ny = parametrs['ny']
        self.u_lb = parametrs['u_lb']
        self.Re = parametrs['Re']

        self.obstacle_params = obstacle.copy()

        self.v = np.array([
            [1, 1], [1, 0], [1, -1],
            [0, 1], [0, 0], [0, -1],
            [-1, 1], [-1, 0], [-1, -1]
        ], dtype=float)

        self.alpha = np.array([
            1/36, 1/9, 1/36,
            1/9, 4/9, 1/9,
            1/36, 1/9, 1/36
        ], dtype=float)

        r = obstacle['r']
        self.nu = (self.u_lb * r) / self.Re
        self.w = 1.0 / (3.0 * self.nu + 0.5)

        self.f_in = np.zeros((9, self.nx, self.ny))
        self.f_out = np.zeros((9, self.nx, self.ny))

        self.rho = np.ones((self.nx, self.ny))
        self.u = np.zeros((self.nx, self.ny, 2))

        self.obstacle = self.add_cylinder(
            obstacle['xc'], obstacle['yc'], obstacle['r'], (self.nx, self.ny)
        )

        self.field_den = []
        self.field_u = []
        self.field_p = []

        self._initialize_fields()

    def _initialize_fields(self):

        self.u[:, :, 0] = self.u_lb
        self.u[:, :, 1] = 0.0

        self.rho[:, :] = 1.0

        for i in range(9):
            self.f_in[i] = self.calc_f_eq_i(i, self.u, self.rho)

    @staticmethod
    def add_cylinder(xc: int, yc: int, r: int, shape: tuple):
        nx, ny = shape
        x, y = np.ogrid[:nx, :ny]
        mask = (x - xc)**2 + (y - yc)**2 <= r**2

        if (xc + r >= nx - 1 or xc - r <= 0 or
                yc + r >= ny - 1 or yc - r <= 0):
            raise ValueError("Circle extends beyond domain boundaries")

        coords = np.where(mask)
        return {'x': coords[0].tolist(), 'y': coords[1].tolist()}

    @staticmethod
    def calc_outflow(f_in: np.ndarray):
        f_in[6, -1, :] = f_in[6, -2, :]
        f_in[7, -1, :] = f_in[7, -2, :]
        f_in[8, -1, :] = f_in[8, -2, :]

    @staticmethod
    def calc_u(density: np.ndarray, f_in: np.ndarray, v: np.ndarray):

        ux = np.zeros_like(density)
        uy = np.zeros_like(density)

        for i in range(9):
            ux += f_in[i] * v[i, 0]
            uy += f_in[i] * v[i, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            ux = ux / density
            uy = uy / density

        ux = np.nan_to_num(ux, nan=0.0)
        uy = np.nan_to_num(uy, nan=0.0)

        return np.stack([ux, uy], axis=-1)

    def calc_f_eq_i(self, i: int, u: np.ndarray, rho: np.ndarray):
        v_dot_u = self.v[i, 0] * u[:, :, 0] + self.v[i, 1] * u[:, :, 1]
        u_sqr = u[:, :, 0]**2 + u[:, :, 1]**2

        return rho * self.alpha[i] * (1.0 + 3.0 * v_dot_u + 4.5 * v_dot_u**2 - 1.5 * u_sqr)

    def calc_inflow(self):
        self.u[0, :, 0] = self.u_lb
        self.u[0, :, 1] = 0.0

        rho_known = np.sum(self.f_in[[0, 1, 2, 3, 4, 5], 0, :], axis=0)
        self.rho[0, :] = rho_known / (1 - self.u[0, :, 0])

        for i in [6, 7, 8]:
            self.f_in[i, 0, :] = self.calc_f_eq_i(
                i, self.u[0:1, :, :], self.rho[0:1, :])[0]

    def calc_f_out(self):
        for i in range(9):
            f_eq = self.calc_f_eq_i(i, self.u, self.rho)
            self.f_out[i] = self.f_in[i] - self.w * (self.f_in[i] - f_eq)

    def bounce_back(self):
        obstacle_mask = np.zeros((self.nx, self.ny), dtype=bool)
        obstacle_mask[self.obstacle['x'], self.obstacle['y']] = True

        for i in range(9):
            opp = 8 - i
            self.f_out[opp, obstacle_mask] = self.f_in[i, obstacle_mask]

    def collision_and_stream(self):
        for i in range(9):
            self.f_in[i] = np.roll(
                np.roll(self.f_out[i], self.v[i, 0], axis=0),
                self.v[i, 1], axis=1
            )

    def _save_snapshot(self):
        u_magnitude = np.sqrt(self.u[:, :, 0]**2 + self.u[:, :, 1]**2)

        self.field_u.append(u_magnitude.copy())
        self.field_den.append(self.rho.copy())
        self.field_p.append(self.rho.copy() / 3.0)

    def solve(self, n_step: int = 100000, step_frame: int = 250):
        print(f"Starting simulation for {n_step} steps...")
        print(
            f"Parameters: nx={self.nx}, ny={self.ny}, u_lb={self.u_lb:.4f}, Re={self.Re}")
        print(f"Viscosity: nu={self.nu:.6f}, Relaxation: w={self.w:.6f}")

        for step in range(n_step):
            self.calc_outflow(self.f_in)

            self.rho = np.sum(self.f_in, axis=0)
            self.u = self.calc_u(self.rho, self.f_in, self.v)

            self.calc_inflow()

            self.calc_f_out()

            self.bounce_back()

            self.collision_and_stream()

            if step % step_frame == 0:
                self._save_snapshot()

            if step % (n_step // 10) == 0:
                progress = step / n_step * 100
                max_u = np.max(
                    np.sqrt(self.u[:, :, 0]**2 + self.u[:, :, 1]**2))
                print(f"Progress: {progress:.1f}% - Max velocity: {max_u:.4f}")

        print("Simulation completed!")