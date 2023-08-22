import torch
import torch.nn as nn
from torchdiffeq import odeint

class SpringODE(nn.Module):
    def __init__(self, k):
        super(SpringODE, self).__init__()
        self.k = k
    
    def forward(self, t, x):
        # x[0]: 변위, x[1]: 속도
        dxdt = torch.zeros_like(x)
        dxdt[0] = x[1]
        dxdt[1] = -self.k * x[0]
        return dxdt

# 모델 초기화
k = 1.0  # 스프링 상수
model = SpringODE(k)

# 초기 조건
t0 = 0.0
t1 = 10.0
x0 = torch.tensor([0.0, 1.0])  # 초기 변위와 속도

# 시간 값 생성
t = torch.linspace(t0, t1, 1000)

# 변위 계산
x = odeint(model, x0, t)

# 변위 시간에 따른 플롯
import matplotlib.pyplot as plt

plt.plot(t.numpy(), x[:, 0].numpy())
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement vs Time')
plt.show()