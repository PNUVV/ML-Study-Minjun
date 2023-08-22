import numpy as np
import matplotlib.pyplot as plt

# 초기값 설정
k = 0.5  # 공기저항 계수
v = 20  # 속도 (m/s)
theta = 25  # 발사각도 (degree)
theta_rad = np.deg2rad(theta)  # 발사각도 (radian)
g = 9.8  # 중력 가속도 (m/s^2)
dt = 0.01  # 시간 간격 (s)

# 초기 속도 설정
vx0 = v * np.cos(theta_rad)
vy0 = v * np.sin(theta_rad)

# 초기 위치 및 속도 설정
x0, y0 = 0, 0
vx, vy = vx0, vy0

# 리스트 초기화
t_list = [0]
x_list = [x0]
y_list = [y0]

# Euler 방법으로 궤적 예측
while y_list[-1] >= 0:
    t = t_list[-1] + dt
    
    # 속도 업데이트
    vx -= k * v * vx * dt
    vy -= (k * v * vy + g) * dt
    
    # 위치 업데이트
    x = x_list[-1] + vx * dt
    y = y_list[-1] + vy * dt
    
    # 리스트에 추가
    t_list.append(t)
    x_list.append(x)
    y_list.append(y)

# 그래프 그리기
plt.plot(x_list, y_list)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.gca().set_aspect("equal", adjustable="box")  # x와 y 축의 scale을 같게 설정
plt.show()

# 시간 출력
print("도달 시간:", t_list[-1])