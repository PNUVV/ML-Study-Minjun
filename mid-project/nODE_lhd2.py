from pyDOE import lhs
import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, max_error
from prettytable import PrettyTable

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

# Neural ODE definition
func = ODEF(2, 20, 2)
optimizer = torch.optim.Adam(func.parameters(), lr=0.01)

# Latin Hypercube Design (LHD) to generate samples
n_samples = 100
lhd_samples = lhs(2, samples=n_samples)

# Using LHD samples for circular motion
t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32)
x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)

# Splitting into training and test sets
train_size = int(0.8 * len(x_lhd))
x_train, x_test = x_lhd[:train_size], x_lhd[train_size:]
t_train, t_test = t_lhd[:train_size], t_lhd[train_size:]

# Training loop
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    x_pred_train = odeint(func, x_train[0], t_train).squeeze()
    loss = ((x_pred_train - x_train) ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Create a 2x2 grid of subplots for the first 4 plots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plotting the training loss
axs[0, 0].plot(losses)
axs[0, 0].set_title('Training Loss (MSE)')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')

# Plotting the true trajectory and the Neural ODE approximation
axs[0, 1].plot(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(), label='True trajectory')
x_pred = odeint(func, x_train[0], t_train)
axs[0, 1].plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
axs[0, 1].legend()
axs[0, 1].set_xlabel('X Position')
axs[0, 1].set_ylabel('Y Position')
axs[0, 1].set_title('2D Motion')

# Testing the model
x_pred_test = odeint(func, x_test[0], t_test).squeeze()
axs[1, 0].plot(x_test[:, 0].detach().numpy(), x_test[:, 1].detach().numpy(), label='True test trajectory')
axs[1, 0].plot(x_pred_test[:, 0].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Predicted test trajectory')
axs[1, 0].legend()
axs[1, 0].set_title('Test Data Prediction')

# Contour plot for specific time and acceleration
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.sqrt(X**2 + Y**2)
axs[1, 1].contourf(X, Y, Z, levels=20, cmap='viridis')
axs[1, 1].set_title('Contour Plot (Example)')
axs[1, 1].set_xlabel('Initial Position')
axs[1, 1].set_ylabel('Initial Velocity')

# Subtitle title and layout adjustments
plt.tight_layout()
plt.show()

# Testing the model for 'Actual vs Predicted Plot'
x_pred_test = odeint(func, x_test[0], t_lhd[train_size:]).squeeze()

# Graphical Validation
plt.scatter(x_test[:, 0].detach().numpy(), x_pred_test[:, 0].detach().numpy(), label='X Position')
plt.scatter(x_test[:, 1].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Y Position')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.title('Actual vs Predicted Plot')
plt.show()

# Numerical Validation
slope, intercept, r_value, _, _ = linregress(x_test.flatten(), x_pred_test.flatten())
r_squared = r_value**2
mean_abs_rel_residual = mean_absolute_error(x_test, x_pred_test) / (x_test.abs().mean())
max_abs_rel_residual = max_error(x_test, x_pred_test) / (x_test.abs().max())

# Summarize the results
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["Squared correlation coefficient (r^2)", r_squared])
table.add_row(["Mean absolute relative residual", mean_abs_rel_residual])
table.add_row(["Maximum of absolute relative residuals", max_abs_rel_residual])

# Print the results table
print(table)