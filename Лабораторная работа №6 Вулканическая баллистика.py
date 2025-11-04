from numpy.random import Generator, default_rng
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import simpy as sim
import time


class Bomb:
    def __init__(self,
                 t_erup: float,
                 r0: np.ndarray,
                 v0: np.ndarray,
                 mass: float,
                 radius: float):
        self.t_erup = t_erup
        self.r0 = r0
        self.v0 = v0
        self.m = mass
        self.R = radius
        self.collided = False
        self.ground_time = None
        self.ground_process = None
        self.collision_processes = []
        self.id = id(self)  # Уникальный идентификатор

    def position(self, t: float) -> np.ndarray:
        """Рассчитать радиус-вектор в момент времени `t`."""
        dt = t - self.t_erup
        if dt < 0:
            return self.r0
        g = np.array([0.0, 0.0, -9.81])
        return self.r0 + self.v0 * dt + 0.5 * g * dt ** 2

    def velocity(self, t: float) -> np.ndarray:
        """Рассчитать вектор скорости в момент времени `t`."""
        dt = t - self.t_erup
        if dt < 0:
            return self.v0
        g = np.array([0.0, 0.0, -9.81])
        return self.v0 + g * dt

    def is_collided(self) -> bool:
        """Столкивался ли камень."""
        return self.collided

    def xy_fall(self) -> np.ndarray:
        """Координаты точки падения."""
        if self.ground_time is None:
            return np.array([0.0, 0.0])
        pos = self.position(self.ground_time)
        return pos[:2]


# Глобальные переменные
flyings: List[Bomb] = []
fallens: List[Bomb] = []

# Параметры модели
H = 1000.0  # Высота
h = 100.0  # Глубина кратера
R_crater = 50.0  # Радиус кратера

mu_v = 80.0  # Скорость
sigma_v = 15.0  # СКО скорости

mu_theta = 0.8  # Угол
sigma_theta = 0.2  # СКО угла
alpha_min, alpha_max = 0, 2 * np.pi  # Диапазон азимута

R_min = 0.3  # Минимальный радиус
R_max = 1.0  # Максимальный радиус
rho_density = 2500.0  # Плотность материала

tau_mean = 0.1  # Время между выбросами
n_eruptions = 15  # Количество выбросов
bombs_per_eruption = 3  # Количество бомб

# Генератор случайных чисел
rs = default_rng(42)


def get_user_choice():
    """Получить выбор пользователя."""
    print("\n" + "=" * 60)
    print("Лабораторная работа №2: Моделирование вулканической баллистики")
    print("=" * 60)

    print("\nВыберите режим моделирования:")
    print("1 - Моделирование БЕЗ учета столкновений")
    print("2 - Моделирование С учетом столкновений")
    print("3 - Сравнительный анализ (оба режима)")
    print("4 - Настроить параметры модели")
    print("0 - Выход")

    while True:
        try:
            choice = int(input("\nВаш выбор (0-4): "))
            if 0 <= choice <= 4:
                return choice
            else:
                print("Пожалуйста, введите число от 0 до 4")
        except ValueError:
            print("Пожалуйста, введите корректное число")


def customize_parameters():
    """Настройка параметров модели пользователем."""
    global H, h, R_crater, mu_v, sigma_v, mu_theta, sigma_theta
    global R_min, R_max, rho_density, tau_mean, n_eruptions, bombs_per_eruption

    print("\n--- Настройка параметров модели ---")
    print("Оставьте поле пустым для использования значения по умолчанию")

    def get_float_input(prompt, default):
        try:
            value = input(f"{prompt} [{default}]: ")
            return float(value) if value.strip() else default
        except ValueError:
            print(f"Используется значение по умолчанию: {default}")
            return default

    def get_int_input(prompt, default):
        try:
            value = input(f"{prompt} [{default}]: ")
            return int(value) if value.strip() else default
        except ValueError:
            print(f"Используется значение по умолчанию: {default}")
            return default

    # Геометрические параметры
    print("\n--- Геометрические параметры ---")
    H = get_float_input("Высота вулкана (м)", H)
    h = get_float_input("Глубина кратера (м)", h)
    R_crater = get_float_input("Радиус кратера (м)", R_crater)

    # Параметры скорости
    print("\n--- Параметры скорости ---")
    mu_v = get_float_input("Средняя скорость выброса (м/с)", mu_v)
    sigma_v = get_float_input("СКО скорости (м/с)", sigma_v)

    # Параметры углов
    print("\n--- Параметры углов ---")
    mu_theta = get_float_input("Средний угол возвышения (рад)", mu_theta)
    sigma_theta = get_float_input("СКО угла возвышения (рад)", sigma_theta)

    # Параметры бомб
    print("\n--- Параметры бомб ---")
    R_min = get_float_input("Минимальный радиус бомбы (м)", R_min)
    R_max = get_float_input("Максимальный радиус бомбы (м)", R_max)
    rho_density = get_float_input("Плотность материала (кг/м³)", rho_density)

    # Временные параметры
    print("\n--- Временные параметры ---")
    tau_mean = get_float_input("Среднее время между выбросами (с)", tau_mean)
    n_eruptions = get_int_input("Количество выбросов", n_eruptions)
    bombs_per_eruption = get_int_input("Количество бомб за выброс", bombs_per_eruption)

    print("\nПараметры успешно обновлены!")
    print_current_parameters()


def print_current_parameters():
    """Вывод текущих параметров модели."""
    print("\nТекущие параметры модели:")
    print(f"  Высота вулкана: {H} м")
    print(f"  Глубина кратера: {h} м")
    print(f"  Радиус кратера: {R_crater} м")
    print(f"  Средняя скорость: {mu_v} м/с")
    print(f"  СКО скорости: {sigma_v} м/с")
    print(f"  Средний угол: {mu_theta} рад")
    print(f"  СКО угла: {sigma_theta} рад")
    print(f"  Радиус бомб: {R_min}-{R_max} м")
    print(f"  Плотность: {rho_density} кг/м³")
    print(f"  Время между выбросами: {tau_mean} с")
    print(f"  Количество выбросов: {n_eruptions}")
    print(f"  Бомб за выброс: {bombs_per_eruption}")


def gen_bombs(env: sim.Environment, n: int) -> List[Bomb]:
    """Сгенерировать `n` бомб (камней)."""
    bombs = []
    for _ in range(n):
        # Случайное положение в кратере
        rho = rs.uniform(0, R_crater)
        phi = rs.uniform(alpha_min, alpha_max)
        zeta = rs.uniform(0, h)

        x0 = rho * np.cos(phi)
        y0 = rho * np.sin(phi)
        z0 = H - h + zeta
        r0 = np.array([x0, y0, z0])

        # Случайная начальная скорость
        v_mag = max(10, rs.normal(mu_v, sigma_v))  # Минимальная скорость 10 м/с
        theta = max(0.1, rs.normal(mu_theta, sigma_theta))  # Минимальный угол 0.1 рад
        psi = rs.uniform(alpha_min, alpha_max)

        vx = v_mag * np.cos(theta) * np.cos(psi)
        vy = v_mag * np.cos(theta) * np.sin(psi)
        vz = v_mag * np.sin(theta)
        v0 = np.array([vx, vy, vz])

        # Параметры бомбы
        radius = rs.uniform(R_min, R_max)
        mass = (4 / 3) * np.pi * radius ** 3 * rho_density

        bomb = Bomb(env.now, r0, v0, mass, radius)
        bombs.append(bomb)

    return bombs


def when_ground(b: Bomb) -> float:
    """Рассчитать время падения камня на землю."""
    g = 9.81
    z0 = b.r0[2]
    vz0 = b.v0[2]

    # Решение квадратного уравнения
    discriminant = vz0 ** 2 + 2 * g * z0
    if discriminant < 0:
        return 0.0

    t1 = (-vz0 - np.sqrt(discriminant)) / g
    t2 = (-vz0 + np.sqrt(discriminant)) / g

    # Выбираем положительный корень
    t_fall = max(t1, t2)
    return max(0.0, t_fall)


def when_collision(b1: Bomb, b2: Bomb) -> Optional[float]:
    """Рассчитать момент времени столкновения камня b1 с камнем b2."""
    try:
        # Вектор относительного положения
        dr = b2.r0 - b1.r0
        # Вектор относительной скорости
        dv = b2.v0 - b1.v0
        # Сумма радиусов
        R_sum = b1.R + b2.R

        # Коэффициенты квадратного уравнения
        A = np.dot(dv, dv)
        if abs(A) < 1e-10:  # Избегаем деления на ноль
            return None

        B = 2 * np.dot(dr, dv)
        C = np.dot(dr, dr) - R_sum ** 2

        # Решение уравнения
        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return None

        t1 = (-B - np.sqrt(discriminant)) / (2 * A)
        t2 = (-B + np.sqrt(discriminant)) / (2 * A)

        # Выбираем минимальный положительный корень
        valid_times = []
        for t in [t1, t2]:
            if t >= 0 and t <= 30:  # Ограничение на время столкновения
                valid_times.append(t)

        if not valid_times:
            return None

        t_collision = min(valid_times)
        # Учитываем время выброса
        max_erup_time = max(b1.t_erup, b2.t_erup)
        collision_time = t_collision + max_erup_time

        return collision_time

    except Exception as e:
        return None


def calc_collision(t: float, b1: Bomb, b2: Bomb) -> Tuple[np.ndarray, np.ndarray]:
    """Рассчитать скорости бомб b1 и b2 после их столкновения в момент времени t."""
    try:
        # Позиции и скорости в момент столкновения
        r1 = b1.position(t)
        r2 = b2.position(t)
        v1 = b1.velocity(t)
        v2 = b2.velocity(t)
        # Вектор от b2 к b1
        r12 = r1 - r2
        distance = np.linalg.norm(r12)

        if distance < 1e-10:  # Избегаем деления на ноль
            return v1, v2

        # Единичный вектор нормали
        n = r12 / distance
        # Относительная скорость
        v_rel = v1 - v2
        # Скорость вдоль нормали
        vn = np.dot(v_rel, n)

        # Если объекты удаляются, столкновения нет
        if vn > 0:
            return v1, v2

        # Коэффициент восстановления
        e = 0.8
        # Импульс
        J = -(1 + e) * vn / (1 / b1.m + 1 / b2.m)
        # Новые скорости
        v1_new = v1 + (J / b1.m) * n
        v2_new = v2 - (J / b2.m) * n

        return v1_new, v2_new

    except Exception as e:
        return b1.velocity(t), b2.velocity(t)


def ground_process(env: sim.Environment, bomb: Bomb):
    """Процесс обработки падения бомбы."""
    try:
        # Вычисляем время падения
        t_fall = when_ground(bomb)
        if t_fall > 100:  # Ограничение времени полета
            t_fall = 100

        bomb.ground_time = env.now + t_fall

        # Ждем момента падения
        yield env.timeout(t_fall)

        # Обрабатываем падение
        if bomb in flyings:
            flyings.remove(bomb)
            fallens.append(bomb)
            # Очищаем процессы столкновений
            for proc in bomb.collision_processes[:]:
                if proc.is_alive:
                    proc.interrupt()
            bomb.collision_processes.clear()

    except sim.Interrupt:
        # Процесс был прерван (столкновение)
        pass


def collision_process(env: sim.Environment, b1: Bomb, b2: Bomb, collision_time: float):
    """Процесс обработки столкновения."""
    try:
        # Ждем момента столкновения
        wait_time = collision_time - env.now
        if wait_time > 0:
            yield env.timeout(wait_time)

        # Проверяем, что бомбы еще летят
        if b1 not in flyings or b2 not in flyings:
            return

        print(f"СТОЛКНОВЕНИЕ! Время: {env.now:.2f} с")

        # Обновляем траектории
        v1_new, v2_new = calc_collision(collision_time, b1, b2)

        # Обновляем параметры бомб
        b1.r0 = b1.position(collision_time)
        b1.v0 = v1_new
        b1.t_erup = collision_time
        b1.collided = True

        b2.r0 = b2.position(collision_time)
        b2.v0 = v2_new
        b2.t_erup = collision_time
        b2.collided = True

        print(f"Бомба 1: новая скорость {np.linalg.norm(v1_new):.1f} м/с")
        print(f"Бомба 2: новая скорость {np.linalg.norm(v2_new):.1f} м/с")

        # Очищаем старые процессы
        for proc in b1.collision_processes[:]:
            if proc.is_alive and proc != env.active_process:
                proc.interrupt()
        for proc in b2.collision_processes[:]:
            if proc.is_alive and proc != env.active_process:
                proc.interrupt()

        # Прерываем процессы падения и создаем новые
        if b1.ground_process and b1.ground_process.is_alive:
            b1.ground_process.interrupt()
        if b2.ground_process and b2.ground_process.is_alive:
            b2.ground_process.interrupt()

        # Запускаем новые процессы падения
        b1.ground_process = env.process(ground_process(env, b1))
        b2.ground_process = env.process(ground_process(env, b2))

    except sim.Interrupt:
        pass


def collision_monitor(env: sim.Environment):
    """Мониторинг столкновений между бомбами."""
    checked_pairs = set()

    while len(flyings) > 0 and env.now < 60:  # Максимальное время 60 секунд
        current_flyings = flyings.copy()
        n = len(current_flyings)

        new_collisions = 0

        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = current_flyings[i], current_flyings[j]

                # Пропускаем если один из объектов уже упал
                if b1 not in flyings or b2 not in flyings:
                    continue

                # Проверяем уникальность пары
                pair_id = tuple(sorted([b1.id, b2.id]))
                if pair_id in checked_pairs:
                    continue

                # Вычисляем время столкновения
                t_collision = when_collision(b1, b2)

                if t_collision is not None and t_collision > env.now:
                    # Запускаем процесс обработки столкновения
                    proc = env.process(collision_process(env, b1, b2, t_collision))
                    b1.collision_processes.append(proc)
                    b2.collision_processes.append(proc)
                    checked_pairs.add(pair_id)
                    new_collisions += 1

        if new_collisions > 0:
            print(f"Обнаружено {new_collisions} потенциальных столкновений")

        # Ждем перед следующей проверкой
        yield env.timeout(0.2)


def eruption_process(env: sim.Environment):
    """Процесс извержения вулкана."""
    for eruption_num in range(n_eruptions):
        # Генерируем новые бомбы
        new_bombs = gen_bombs(env, bombs_per_eruption)

        for bomb in new_bombs:
            flyings.append(bomb)
            # Запускаем процесс падения
            bomb.ground_process = env.process(ground_process(env, bomb))

        print(f"Выброс {eruption_num + 1} в время {env.now:.2f} с")

        # Ждем до следующего выброса
        if eruption_num < n_eruptions - 1:
            yield env.timeout(rs.exponential(tau_mean))

    print("Все выбросы завершены")


def simulate(enable_collisions: bool = True):
    """Основная функция моделирования."""
    global flyings, fallens

    # Сброс глобальных переменных
    flyings, fallens = [], []

    # Создание среды SimPy
    env = sim.Environment()

    # Запуск процесса извержения
    env.process(eruption_process(env))

    # Запуск мониторинга столкновений (если включено)
    if enable_collisions:
        env.process(collision_monitor(env))

    # Запуск моделирования
    print("Начало моделирования...")
    start_time = time.time()

    try:
        # Запускаем до завершения всех событий или по таймауту
        env.run(until=60)  # Максимальное время 60 секунд
    except Exception as e:
        print(f"Ошибка при моделировании: {e}")

    # Принудительно завершаем все процессы
    for bomb in flyings[:]:
        if bomb.ground_process and bomb.ground_process.is_alive:
            bomb.ground_process.interrupt()
        flyings.remove(bomb)
        fallens.append(bomb)

    collided_count = sum(1 for bomb in fallens if bomb.collided)
    print(f"Моделирование завершено за {time.time() - start_time:.1f} с")
    print(f"Упало бомб: {len(fallens)}")
    print(f"Столкнувшихся бомб: {collided_count}")

    return fallens.copy()


def draw_volcano(ax, volcano_color='#8B4513', crater_color='#FF4500'):
    """Нарисовать схему вулкана на графике."""
    # Основание вулкана (конус)
    base_radius = R_crater * 3
    x_base = np.linspace(-base_radius, base_radius, 100)
    y_base_pos = np.sqrt(base_radius ** 2 - x_base ** 2)
    y_base_neg = -y_base_pos

    ax.fill_between(x_base, y_base_pos, y_base_neg, color=volcano_color, alpha=0.7, label='Склон вулкана')

    # Кратер
    theta = np.linspace(0, 2 * np.pi, 100)
    x_crater = R_crater * np.cos(theta)
    y_crater = R_crater * np.sin(theta)

    ax.fill(x_crater, y_crater, color=crater_color, alpha=0.9, label='Кратер')
    ax.plot(x_crater, y_crater, 'k-', linewidth=2)

    # Центр кратера
    ax.plot(0, 0, 'ro', markersize=8, label='Центр извержения')


def plot_single_mode(results, title):
    """Построение графика для одного режима."""
    fig, ax = plt.subplots(figsize=(12, 10))

    if not results:
        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(title, fontsize=14)
        plt.show()
        return

    # Рисуем вулкан
    draw_volcano(ax)

    # Собираем координаты и статистику
    coords = []
    collided_status = []
    collided_count = 0

    for bomb in results:
        xy = bomb.xy_fall()
        coords.append(xy)
        collided_status.append(bomb.collided)
        if bomb.collided:
            collided_count += 1

    coords = np.array(coords)
    collided_status = np.array(collided_status)

    # Рисуем точки падения
    if len(coords) > 0:
        # Разделяем точки по признаку столкновения
        normal_mask = ~collided_status
        collided_mask = collided_status

        if np.any(normal_mask):
            ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
                       c='blue', alpha=0.7, label=f'Без столкновений ({np.sum(normal_mask)})',
                       s=30, edgecolors='black', linewidth=0.5)

        if np.any(collided_mask):
            ax.scatter(coords[collided_mask, 0], coords[collided_mask, 1],
                       c='red', alpha=0.8, label=f'После столкновений ({np.sum(collided_mask)})',
                       s=60, edgecolors='darkred', linewidth=1, marker='*')

    # Настройки графика
    ax.set_xlabel('Координата X, м', fontsize=12)
    ax.set_ylabel('Координата Y, м', fontsize=12)
    ax.set_title(f'{title}\nВсего бомб: {len(results)}', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Устанавливаем разумные пределы
    if len(coords) > 0:
        max_range = max(2000, np.max(np.abs(coords)) * 1.2)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)

    # Добавляем информационную таблицу
    info_text = (f'Параметры вулкана:\n'
                 f'Высота: {H:.0f} м\n'
                 f'Глубина кратера: {h:.0f} м\n'
                 f'Радиус кратера: {R_crater:.0f} м\n'
                 f'Выбросов: {n_eruptions}\n'
                 f'Бомб за выброс: {bombs_per_eruption}\n'
                 f'Столкнулось: {collided_count}')

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
            fontsize=10, linespacing=1.5)

    plt.tight_layout()
    plt.show()


def main():
    """Основная функция программы."""
    while True:
        choice = get_user_choice()

        if choice == 0:
            print("Выход из программы...")
            break

        elif choice == 1:
            print("\n--- Моделирование БЕЗ учета столкновений ---")
            print_current_parameters()
            results = simulate(enable_collisions=False)
            plot_single_mode(results, "Моделирование БЕЗ учета столкновений")

        elif choice == 2:
            print("\n--- Моделирование С учетом столкновений ---")
            print_current_parameters()
            results = simulate(enable_collisions=True)
            plot_single_mode(results, "Моделирование С учетом столкновений")

        elif choice == 3:
            print("\n--- Сравнительный анализ ---")
            print_current_parameters()
            print("\nЗапуск моделирования БЕЗ столкновений...")
            results_no_coll = simulate(enable_collisions=False)
            print("\nЗапуск моделирования С учетом столкновений...")
            results_with_coll = simulate(enable_collisions=True)

            # Выводим оба графика
            if results_no_coll:
                plot_single_mode(results_no_coll, "БЕЗ учета столкновений")
            if results_with_coll:
                plot_single_mode(results_with_coll, "С учетом столкновений")

        elif choice == 4:
            customize_parameters()

        # Предложение продолжить
        if choice != 0:
            continue_choice = input("\nПродолжить работу? (y/n): ").lower()
            if continue_choice != 'y':
                print("Выход из программы...")
                break


if __name__ == "__main__":
    main()