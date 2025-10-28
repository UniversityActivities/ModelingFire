import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from enum import Enum

SIZE = 50
STABILIZATION_STEPS = 200
MEASUREMENT_STEPS = 300
PROB_VERTICAL_PRIORITY = 0.3
MOVE_PROB = 0.9
PRIORITY_HOLD_STEPS = 5

class CellState(Enum):
    EMPTY = 0
    ROAD = 1
    VEHICLE = 2

def is_intersection_cell(row, col, grid_size):
    center = grid_size // 2
    return center - 1 <= row <= center + 1 and center - 1 <= col <= center + 1

def is_intersection_clear(grid, grid_size):
    center = grid_size // 2
    for i, j in itertools.product(range(center - 1, center + 2), repeat=2):
        if grid[i, j] == CellState.VEHICLE.value:
            return False
    return True

def update_grid_state(current_grid, next_grid, grid_size, vertical_has_priority):
    moved_vehicles = 0
    intersection_rows_cols = [grid_size // 2 - 1, grid_size // 2, grid_size // 2 + 1]
    intersection_occupied = not is_intersection_clear(current_grid, grid_size)

    for i in range(grid_size):
        for j in intersection_rows_cols:
            if current_grid[i, j] == CellState.VEHICLE.value:
                if np.random.rand() > MOVE_PROB:
                    continue
                next_i = (i - 1) % grid_size


                if is_intersection_cell(i, j, grid_size):
                    after_exit = (next_i - 1) % grid_size
                    if current_grid[after_exit, j] == CellState.VEHICLE.value:
                        continue


                if is_intersection_cell(i, j, grid_size) and intersection_occupied and not vertical_has_priority:
                    continue

                if current_grid[next_i, j] == CellState.ROAD.value:
                    next_grid[next_i, j] = CellState.VEHICLE.value
                    next_grid[i, j] = CellState.ROAD.value
                    moved_vehicles += 1

    for j in range(grid_size):
        for i in intersection_rows_cols:
            if current_grid[i, j] == CellState.VEHICLE.value:
                if np.random.rand() > MOVE_PROB:
                    continue
                next_j = (j + 1) % grid_size


                if is_intersection_cell(i, j, grid_size):
                    after_exit = (next_j + 1) % grid_size
                    if current_grid[i, after_exit] == CellState.VEHICLE.value:
                        continue


                if is_intersection_cell(i, j, grid_size) and intersection_occupied and vertical_has_priority:
                    continue

                if current_grid[i, next_j] == CellState.ROAD.value:
                    next_grid[i, next_j] = CellState.VEHICLE.value
                    next_grid[i, j] = CellState.ROAD.value
                    moved_vehicles += 1

    return next_grid, moved_vehicles

def run_simulation(vehicle_density, show_visualization=False):
    grid_size = SIZE
    np.random.seed(0)

    grid = np.array([[CellState.EMPTY.value for _ in range(grid_size)] for _ in range(grid_size)])
    for i in range(grid_size):
        grid[grid_size // 2 - 1, i] = CellState.ROAD.value
        grid[grid_size // 2, i] = CellState.ROAD.value
        grid[grid_size // 2 + 1, i] = CellState.ROAD.value
        grid[i, grid_size // 2 - 1] = CellState.ROAD.value
        grid[i, grid_size // 2] = CellState.ROAD.value
        grid[i, grid_size // 2 + 1] = CellState.ROAD.value

    total_road_cells = np.sum(grid == CellState.ROAD.value)
    vehicle_count = int(total_road_cells * vehicle_density)
    road_cell_positions = np.argwhere(grid == CellState.ROAD.value)
    selected_vehicle_positions = np.random.choice(len(road_cell_positions), vehicle_count, replace=False)
    for idx in selected_vehicle_positions:
        x, y = road_cell_positions[idx]
        grid[x, y] = CellState.VEHICLE.value

    next_grid = grid.copy()

    if show_visualization:
        color_scheme = ['white', 'lightgray', 'red']
        custom_cmap = mcolors.ListedColormap(color_scheme)
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        visualization = ax.matshow(grid, cmap=custom_cmap)
        plt.title(f"ρ={vehicle_density:.2f} — Симуляция перекрёстка")

    total_flow_rate = 0
    total_steps = STABILIZATION_STEPS + MEASUREMENT_STEPS


    vertical_has_priority = np.random.rand() < PROB_VERTICAL_PRIORITY
    hold_counter = PRIORITY_HOLD_STEPS

    for step in range(total_steps):
        grid = next_grid.copy()

        hold_counter -= 1
        if hold_counter <= 0:
            vertical_has_priority = np.random.rand() < PROB_VERTICAL_PRIORITY
            hold_counter = PRIORITY_HOLD_STEPS

        next_grid, vehicles_moved = update_grid_state(grid, next_grid, grid_size, vertical_has_priority)

        if step >= STABILIZATION_STEPS:
            total_flow_rate += vehicles_moved

        if show_visualization and step % 2 == 0:
            visualization.set_data(grid)
            ax.set_title(f"ρ={vehicle_density:.2f}, шаг={step}, поток={vehicles_moved}")
            plt.pause(0.03)

    if show_visualization:
        plt.ioff()
        plt.show()

    return total_flow_rate / MEASUREMENT_STEPS

if __name__ == "__main__":
    density_values = np.linspace(0.05, 1.0, 15)
    flow_results = []

    print("\n--- Расчёт фундаментальной диаграммы ---")
    for density in density_values:
        flow_rate = run_simulation(density, show_visualization=False)
        flow_results.append(flow_rate)
        print(f"Плотность ρ = {density:.2f}, Поток I = {flow_rate:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(density_values, flow_results, 'o-', color='crimson', linewidth=2, markersize=6)
    plt.xlabel('Плотность потока ρ')
    plt.ylabel('Поток I')
    plt.title('Фундаментальная диаграмма: перекрёсток без светофора ')
    plt.grid(True)
    plt.show()

    for density in [0.1, 0.3, 0.5, 0.8]:
        print(f"\n=== Демонстрация ρ={density:.2f} ===")
        run_simulation(density, show_visualization=True)
