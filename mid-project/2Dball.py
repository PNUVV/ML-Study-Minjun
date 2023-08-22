import numpy as np
import matplotlib.pyplot as plt

def euler_method(k, t, v, theta):
    # 초기 조건 설정
    x0, y0 = 0.0, 0.0  # 초기 위치 (x, y)
    vx0 = v * np.cos(np.deg2rad(theta))  # 초기 x 방향 속도
    vy0 = v * np.sin(np.deg2rad(theta))  # 초기 y 방향 속도

    # 공기저항을 고려한 운동 방정식
    def ode_system(t, y):
        x, y, vx, vy = y
        dxdt = vx
        dydt = vy
        dvxdt = -k * vx
        dvydt = -9.8 - k * vy
        return np.array([dxdt, dydt, dvxdt, dvydt])

    # Euler 메서드를 사용한 미분 방정식 풀이
    dt = 0.01  # 시간 간격
    num_steps = int(t / dt)
    trajectory = np.zeros((num_steps, 2))
    trajectory[0] = [x0, y0]
    state = np.array([x0, y0, vx0, vy0])

    for step in range(1, num_steps):
        state = state + dt * ode_system(step * dt, state)
        trajectory[step] = state[:2]

    return trajectory

# 그래프 출력
def plot_trajectory(trajectory):
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Projectile Motion with Air Resistance')
    plt.grid(True)
    plt.show()

# 입력값 설정
k = 0.1  # 공기저항 계수
t = 15.0  # 예측할 시간
v = 200.0  # 초기 발사 속도
theta = 45.0  # 발사 각도 degree

trajectory = euler_method(k, t, v, theta)
plot_trajectory(trajectory)