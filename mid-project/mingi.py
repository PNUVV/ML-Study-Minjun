import numpy as np
import torch
import torch.nn as nn
import neurodiffeq as nde
import matplotlib.pyplot as plt

from neurodiffeq import diff

from neurodiffeq.callbacks import ActionCallback

from neurodiffeq.conditions import IVP
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.callbacks import MonitorCallback

# 2차 미분 방정식을 정의합니다 (예시로 u'' + u' + 2u = 0 를 사용합니다).pip install --upgrade neurodiffeq
def ode_system(u, t):
#    return [diff(u, t, order=2)+u]
    return [diff(u, t, order=2)+diff(u, t, order=1)+2*u]

# 초기 조건과 독립 변수 범위를 설정합니다.
#initial_condition = [nde.conditions.IVP(t_0=0.0, u_0=0.0, u_0_prime=1.0)]  # u(0) = 0, u'(0) = 1
initial_condition = [nde.conditions.IVP(t_0=0.0, u_0=1.0, u_0_prime=0.0)]   # u(0) = 1, u'(0) = 0


# 10 에포크마다 중간 결과를 출력하는 커스텀 콜백을 정의합니다.
class CustomMonitor(ActionCallback):
    def __call__(self, solver):
        if (solver.local_epoch + 1) % 10 == 0:
            print(f'Epoch {solver.local_epoch + 1}, Train Loss: {solver.metrics_history["train_loss"][-1]:.4f}, Valid Loss: {solver.metrics_history["valid_loss"][-1]:.4f}')

# 커스텀 콜백을 인스턴스화합니다.
class PredictionMonitor(ActionCallback):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val
        self.predictions = []

    def __call__(self, solver):
        if (solver.local_epoch + 1) % 10 == 0:
            prediction = solver.model.predict(self.x_val)
            self.predictions.append(prediction)

            plt.figure(figsize=(10, 5))
            plt.plot(self.y_val, self.y_val, 'r', label='y=x line')
            plt.scatter(self.y_val, prediction, label=f'Epoch {solver.local_epoch + 1}')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.legend()
            plt.grid(True)
            plt.show()

custom_monitor = CustomMonitor()

# Generator1D 클래스의 인스턴스를 생성하여 훈련, 검증 데이터를 생성합니다.
train_gen = nde.generators.Generator1D(size=1000, t_min=0.0, t_max=5.0, method='equally-spaced-noisy')
valid_gen = nde.generators.Generator1D(size=200, t_min=0.0, t_max=5.0, method='equally-spaced')

#method에 대해서
#If set to 'uniform',
#   the points will be drew from a uniform distribution Unif(t_min, t_max).
#If set to 'equally-spaced',
#   the points will be fixed to a set of linearly-spaced points that go from t_min to t_max.
#If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points.
#If set to 'log-spaced', the points will be fixed to a set of log-spaced points that go from t_min to t_max.
#If set to 'log-spaced-noisy', a normal noise will be added to the previously mentioned set of points,
#If set to 'chebyshev1' or 'chebyshev', the points are chebyshev nodes of the first kind over (t_min, t_max).
#If set to 'chebyshev2', the points will be chebyshev nodes of the second kind over [t_min, t_max].

# 신경망 모델 정의 (여기서는 단순한 Fully Connected Neural Network를 사용합니다).
neural_net = [nde.solvers.FCNN(n_hidden_layers=6, n_hidden_units=60, actv=torch.nn.Tanh)]

# metrics 함수를 정의합니다.
def mse_metric(outputs, t):
    t = t.detach()
#    analytic_solution = np.sin(t)
    analytic_solution = \
        np.exp(-t/2.) * (np.cos(np.sqrt(7) * t / 2.) + np.sin(np.sqrt(7) * t / 2.)/np.sqrt(7.)) # 해석적 솔루션
    return torch.mean((outputs - analytic_solution)**2)

# metrics 딕셔너리를 생성합니다.
metrics = {'mse': mse_metric}


