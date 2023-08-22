import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network representing the dynamics
class ODEF(nn.Module):
    def forward(self, t, x):
        return self.net(x)

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

# Parameters
input_dim = 2  # 2D motion
hidden_dim = 200
output_dim = 2  # 2D motion

# Neural ODE definition
func = ODEF(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(func.parameters(), lr=0.01)

# Generate some training data (circular motion)
t = torch.linspace(0, 2*np.pi, 100).reshape(-1)
x_train = torch.cat((torch.sin(t).reshape(-1, 1), torch.cos(t).reshape(-1, 1)), dim=1)

# Splitting into training and test sets
train_size = int(0.8 * len(x_train))
x_train_data, x_test_data = x_train[:train_size], x_train[train_size:]

# Training loop
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    x_pred_train = odeint(func, x_train_data[0], t[:train_size]).squeeze()
    loss = ((x_pred_train - x_train_data) ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 2x2 그리드의 서브플롯 생성
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Training Loss 그래프
axes[0, 0].plot(losses)
axes[0, 0].set_title('Training Loss (MSE)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')

# True trajectory 및 Neural ODE approximation 그래프
axes[0, 1].plot(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(), label='True trajectory')
x_pred = odeint(func, x_train[0], t)
axes[0, 1].plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
axes[0, 1].legend()
axes[0, 1].set_xlabel('X Position')
axes[0, 1].set_ylabel('Y Position')
axes[0, 1].set_title('2D Motion')

# Test Data Prediction 그래프
x_pred_test = odeint(func, x_test_data[0], t[train_size:]).squeeze()
axes[1, 0].plot(x_test_data[:, 0].detach().numpy(), x_test_data[:, 1].detach().numpy(), label='True test trajectory')
axes[1, 0].plot(x_pred_test[:, 0].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Predicted test trajectory')
axes[1, 0].legend()
axes[1, 0].set_title('Test Data Prediction')

# Contour Plot 그래프
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.sqrt(X**2 + Y**2)  # Example: distance from the origin
contour = axes[1, 1].contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, ax=axes[1, 1])
axes[1, 1].set_title('Contour Plot (Example)')
axes[1, 1].set_xlabel('Initial Position')
axes[1, 1].set_ylabel('Initial Velocity')

# 레이아웃 조정 및 그래프 출력
plt.tight_layout()
plt.show()