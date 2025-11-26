import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
import warnings


# =====================================================
# ===================== ЗАДАНИЕ 1 =====================
# =====================================================

print("\n===================== ЗАДАНИЕ 1 =====================\n")

def s_exact(t, e):
    return np.exp(-t / e)

def euler_solve(t0, t1, s0, e, n):
    t = np.linspace(t0, t1, n + 1)
    dt = t[1] - t[0]
    s = np.zeros(n + 1)
    s[0] = s0
    for i in range(n):
        s[i + 1] = s[i] + dt * (-s[i] / e)
    return t, s

# Параметры
eps1 = 0.1
eps2 = 0.01

print("Аналитическое решение s(t) = exp(-t/eps)")

t_plot = np.linspace(0, 1, 500)
s1 = s_exact(t_plot, eps1)
s2 = s_exact(t_plot, eps2)

# Графики аналитики
plt.figure()
plt.plot(t_plot, s1, label="eps=0.1")
plt.plot(t_plot, s2, label="eps=0.01")
plt.title("Задание 1 — Аналитические решения")
plt.grid()
plt.legend()
plt.show()

# Численные решения Эйлера
t_e1, s_e1 = euler_solve(0, 1, 1, eps2, 50)
t_e2, s_e2 = euler_solve(0, 1, 1, eps2, 500)

plt.figure()
plt.plot(t_e1, s_e1, label="dt=0.02")
plt.plot(t_e2, s_e2, label="dt=0.002")
plt.title("Задание 1 — Метод Эйлера при eps=0.01")
plt.grid()
plt.legend()
plt.show()

# Ошибка
err = np.abs(s_e2 - s_exact(t_e2, eps2))
plt.figure()
plt.plot(t_e2, err)
plt.title("Задание 1 — Ошибка метода Эйлера")
plt.grid()
plt.show()



# =====================================================
# ===================== ЗАДАНИЕ 2 =====================
# =====================================================

warnings.simplefilter("always", UserWarning)  # показывать все ворнинги

print("\n===================== ЗАДАНИЕ 2 =====================\n")

# Константы
g = 9.81
rho = 1.240
a_sound = 340.0

def Cx(M):
    return 0.44

# Начальные данные
x0, y0 = 0, 0
V0 = 100
theta0 = np.deg2rad(30)
mass = 300  # увеличенная масса для устойчивости интегратора
S = np.pi * (0.25)**2  # площадь поперечного сечения

# Профиль тяги
def thrust(t):
    if t < 0.5: return 400*(t/0.5)
    if t < 2:   return 400
    return 400*np.exp(-(t-2)/2)

# Функция для интеграции
def f2(t, Y):
    x, y, Vx, Vy = Y
    V = np.hypot(Vx, Vy)

    # Сопротивление
    if V > 1e-8:
        M = V / a_sound
        Fd = 0.5 * rho * V**2 * Cx(M) * S
        ax_drag = -(Fd / mass) * (Vx / V)
        ay_drag = -(Fd / mass) * (Vy / V)
    else:
        ax_drag = ay_drag = 0.0

    # Тяга
    P = thrust(t)
    if V > 1e-8:
        ux, uy = Vx / V, Vy / V
    else:
        ux, uy = np.cos(theta0), np.sin(theta0)

    ax_thr = (P / mass) * ux
    ay_thr = (P / mass) * uy

    ax = ax_drag + ax_thr
    ay = ay_drag + ay_thr - g

    return np.array([Vx, Vy, ax, ay])

# Начальное состояние
Y0 = np.array([x0, y0, V0*np.cos(theta0), V0*np.sin(theta0)])

# Интегратор dopri5
solver = integ.ode(f2).set_integrator("dopri5", atol=1e-7, rtol=1e-7, nsteps=10000)
solver.set_initial_value(Y0, 0)

# Инициализация массивов
t_list = [solver.t]
Y_list = [solver.y.copy()]

t_max = 20
t_grid = np.linspace(0, t_max, 3000)

# Интегрирование с проверкой успешности
for t_val in t_grid[1:]:
    solver.integrate(t_val)
    if not solver.successful():
        print("Интегратор не смог выполнить шаг, прерывание")
        break

    Y_list.append(solver.y.copy())
    t_list.append(solver.t)

    if solver.y[1] < 0:  # момент приземления
        break

# Преобразуем в массивы
Y_arr = np.array(Y_list)
t_arr = np.array(t_list)

# Защита от 1D массива
if Y_arr.ndim == 1:
    Y_arr = Y_arr[np.newaxis, :]

# Линейная интерполяция для точного времени приземления
if len(Y_arr) > 1 and Y_arr[-1,1] < 0:
    y1, y0_ = Y_arr[-1,1], Y_arr[-2,1]
    t1, t0_ = t_arr[-1], t_arr[-2]
    x1, x0_ = Y_arr[-1,0], Y_arr[-2,0]
    t_land = t0_ + (0 - y0_) * (t1 - t0_) / (y1 - y0_)
    x_land = x0_ + (0 - y0_) * (x1 - x0_) / (y1 - y0_)
else:
    t_land = t_arr[-1]
    x_land = Y_arr[-1,0]

print(f"Время полёта: {t_land:.3f} с")
print(f"Дальность: {x_land:.3f} м")

# ---- Графики ----

# Траектория
plt.figure()
plt.plot(Y_arr[:,0], Y_arr[:,1], label="Траектория")
plt.scatter([x_land], [0], color='red', label="Приземление")
plt.title("Задание 2 — Траектория")
plt.xlabel("x, м")
plt.ylabel("y, м")
plt.grid()
plt.legend()
plt.show()

# Профиль тяги
plt.figure()
plt.plot(t_arr, [thrust(t) for t in t_arr])
plt.title("Профиль тяги P(t)")
plt.xlabel("t, с")
plt.ylabel("P, Н")
plt.grid()
plt.show()

# Компоненты скорости
plt.figure()
plt.plot(t_arr, Y_arr[:,2], label="Vx")
plt.plot(t_arr, Y_arr[:,3], label="Vy")
plt.title("Компоненты скорости")
plt.xlabel("t, с")
plt.ylabel("Скорость, м/с")
plt.grid()
plt.legend()
plt.show()