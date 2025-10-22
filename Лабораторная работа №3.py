import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import pandas as pd


class GreenbergHastingsWithMetrics:
    def __init__(self, size=80, spontaneous_prob=0.002):
        self.size = size
        self.spontaneous_prob = spontaneous_prob
        self.grid = np.zeros((size, size), dtype=int)

        # История статистики
        self.stats_history = {
            'step': [],
            'resting': [],
            'excited': [],
            'refractory': [],
            'total_activity': [],
            'active_clusters': [],
            'avg_cluster_size': [],
            'wave_front_size': []
        }

        # Порог для вспышек
        self.activity_threshold = size * size * 0.1
        self.in_flash = False
        self.flash_start_step = 0
        self.flashes = []

        # Инициализация сетки
        self.initialize_grid()
        self.colors = ['black', 'red', 'yellow']
        self.cmap = ListedColormap(self.colors)

    def initialize_grid(self):
        """Инициализация случайными возбужденными клетками"""
        num_active = max(5, int(self.size * self.size * 0.01))
        for _ in range(num_active):
            i, j = np.random.randint(0, self.size, 2)
            self.grid[i, j] = 1

    def update(self):
        """Обновление сетки по правилам модели"""
        new_grid = self.grid.copy()
        excited_count = 0

        for i in range(self.size):
            for j in range(self.size):
                current_state = self.grid[i, j]

                if current_state == 0:  # Покой
                    excited_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = (i + di) % self.size, (j + dj) % self.size
                            if self.grid[ni, nj] == 1:
                                excited_neighbor = True
                                break

                    if excited_neighbor or np.random.random() < self.spontaneous_prob:
                        new_grid[i, j] = 1
                        excited_count += 1

                elif current_state == 1:  # Возбуждение
                    new_grid[i, j] = 2
                    excited_count += 1

                elif current_state == 2:  # Рефрактерность
                    new_grid[i, j] = 0

        self.grid = new_grid
        return excited_count

    def calculate_metrics(self, step, excited_count):
        """Вычисление и сохранение метрик"""
        states, counts = np.unique(self.grid, return_counts=True)
        stats_dict = {0: 0, 1: 0, 2: 0}
        for state, count in zip(states, counts):
            stats_dict[state] = count

        active_clusters = self.find_active_clusters()
        active_cluster_count = len(active_clusters)
        avg_cluster_size = np.mean([len(cluster) for cluster in active_clusters]) if active_clusters else 0
        wave_front_size = self.calculate_wave_front()
        total_activity = stats_dict[1]

        self.detect_flashes(step, total_activity)

        self.stats_history['step'].append(step)
        self.stats_history['resting'].append(stats_dict[0])
        self.stats_history['excited'].append(stats_dict[1])
        self.stats_history['refractory'].append(stats_dict[2])
        self.stats_history['total_activity'].append(total_activity)
        self.stats_history['active_clusters'].append(active_cluster_count)
        self.stats_history['avg_cluster_size'].append(avg_cluster_size)
        self.stats_history['wave_front_size'].append(wave_front_size)

    def find_active_clusters(self):
        """Поиск кластеров возбужденных клеток"""
        visited = set()
        clusters = []

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1 and (i, j) not in visited:
                    cluster = []
                    stack = [(i, j)]

                    while stack:
                        x, y = stack.pop()
                        if (x, y) not in visited and self.grid[x, y] == 1:
                            visited.add((x, y))
                            cluster.append((x, y))

                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = (x + dx) % self.size, (y + dy) % self.size
                                if (nx, ny) not in visited:
                                    stack.append((nx, ny))

                    if cluster:
                        clusters.append(cluster)

        return clusters

    def calculate_wave_front(self):
        """Подсчет размера волнового фронта"""
        front_size = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = (i + di) % self.size, (j + dj) % self.size
                        if self.grid[ni, nj] == 0:
                            front_size += 1
                            break
        return front_size

    def detect_flashes(self, step, activity):
        """Определение начала и конца вспышек"""
        if activity > self.activity_threshold and not self.in_flash:
            self.in_flash = True
            self.flash_start_step = step
            self.flash_peak_activity = activity
            print(f"⚡ ВСПЫШКА НАЧАЛАСЬ на шаге {step}, активность: {activity}")

        elif activity <= self.activity_threshold and self.in_flash:
            self.in_flash = False
            duration = step - self.flash_start_step
            self.flashes.append({
                'start': self.flash_start_step,
                'end': step,
                'peak': self.flash_peak_activity,
                'duration': duration
            })
            print(f"✅ ВСПЫШКА ЗАКОНЧИЛАСЬ на шаге {step}, длительность: {duration}")

        elif self.in_flash and activity > self.flash_peak_activity:
            self.flash_peak_activity = activity


def run_animation():
    """Анимация клеточного автомата"""
    print("ЗАПУСК АНИМАЦИИ...")
    print("Черный = Покой | Красный = Возбуждение | Желтый = Рефрактерность")
    model = GreenbergHastingsWithMetrics(size=60, spontaneous_prob=0.003)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(model.grid, cmap=model.cmap, vmin=0, vmax=2, interpolation='nearest')
    ax.set_title('Клеточный автомат Гринберга–Хастингса\n(закройте окно для просмотра метрик)')
    ax.set_xticks([])
    ax.set_yticks([])

    def animate(frame):
        excited_count = model.update()
        model.calculate_metrics(frame, excited_count)
        im.set_array(model.grid)
        ax.set_title(f'Шаг {frame + 1}/300')
        return [im]

    anim = FuncAnimation(fig, animate, frames=300, interval=100, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

    return model


def plot_metrics(model):
    """Построение графиков метрик (каждый график в отдельном окне)"""
    stats = pd.DataFrame(model.stats_history)

    # 1. Распределение состояний
    plt.figure(figsize=(8, 5))
    plt.plot(stats['step'], stats['resting'], label='Покой (0)')
    plt.plot(stats['step'], stats['excited'], label='Возбуждение (1)')
    plt.plot(stats['step'], stats['refractory'], label='Рефрактерность (2)')
    plt.title('Распределение состояний')
    plt.xlabel('Шаг')
    plt.ylabel('Клетки')
    plt.legend()
    plt.grid()
    plt.show()

    # 2. Общая активность
    plt.figure(figsize=(8, 5))
    plt.plot(stats['step'], stats['total_activity'], color='red')
    plt.axhline(y=model.activity_threshold, color='gray', linestyle='--', label='Порог вспышки')
    plt.title('Общая активность')
    plt.xlabel('Шаг')
    plt.ylabel('Возбужденные клетки')
    plt.legend()
    plt.grid()
    plt.show()

    # 3. Число активных кластеров
    plt.figure(figsize=(8, 5))
    plt.plot(stats['step'], stats['active_clusters'], color='purple')
    plt.title('Число активных кластеров')
    plt.xlabel('Шаг')
    plt.ylabel('Кластеры')
    plt.grid()
    plt.show()

    # 4. Средний размер кластера
    plt.figure(figsize=(8, 5))
    plt.plot(stats['step'], stats['avg_cluster_size'], color='orange')
    plt.title('Средний размер кластера')
    plt.xlabel('Шаг')
    plt.ylabel('Размер')
    plt.grid()
    plt.show()

def main():
    print("=" * 60)
    print("КЛЕТОЧНЫЙ АВТОМАТ ГРИНБЕРГА–ХАСТИНГСА С АНАЛИЗОМ МЕТРИК")
    print("=" * 60)

    model = run_animation()
    print("\n Построение графиков метрик...")
    plot_metrics(model)
    print("\n Анализ завершен!")


if __name__ == "__main__":
    main()

