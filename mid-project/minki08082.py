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
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

# Neural ODE definition
func = ODEF(2, 50, 2)
optimizer = torch.optim.Adam(func.parameters(), lr=0.001)

# Latin Hypercube Design (LHD) to generate samples
n_samples = 1000
lhd_samples = lhs(2, samples=n_samples)

# Using LHD samples for circular motion
# float64 타입 이슈
t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32) # dtype added here
x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)


# Splitting into training and test sets
train_size = int(0.6 * len(x_lhd))
val_size = int(0.2 * len(x_lhd))
x_train, x_val, x_test = x_lhd[:train_size], x_lhd[train_size:train_size+val_size], x_lhd[train_size+val_size:]
t_train, t_val, t_test = t_lhd[:train_size], t_lhd[train_size:train_size+val_size], t_lhd[train_size+val_size:]

# ...

# Training loop
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    x_pred_train = odeint(func, x_train[0], t_train).squeeze()
    loss = ((x_pred_train - x_train) ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(epoch)

 
 # Plotting the training loss
plt.plot(losses)
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plotting the true trajectory and the Neural ODE approximation
# plt.plot(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(), label='True trajectory')
plt.plot(x_lhd[:, 0].detach().numpy(), x_lhd[:, 1].detach().numpy(), label='True trajectory')
x_pred = odeint(func, x_train[0], t_train)
plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
# Testing the model
x_pred_valid = odeint(func, x_val[0], t_val).squeeze()
x_pred_test = odeint(func, x_test[0], t_test).squeeze()
# Plotting the true test trajectory and the predicted test trajectory
plt.plot(x_pred_valid[:, 0].detach().numpy(), x_pred_valid[:, 1].detach().numpy(), label='Predicted validation trajectory')
plt.plot(x_pred_test[:, 0].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Predicted test trajectory')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Motion prediction')
plt.show()


# Numerical Validation
slope, intercept, r_value, _, _ = linregress(x_test.flatten().detach().numpy(), x_pred_test.flatten().detach().numpy())

r_squared = r_value**2
mean_abs_rel_residual = mean_absolute_error(x_test.detach().numpy(), x_pred_test.detach().numpy()) / (x_test.abs().mean())
# Calculate max absolute relative residual for multioutput
max_abs_rel_residual = max(np.max(np.abs(x_test.detach().numpy() - x_pred_test.detach().numpy()), axis=0) / x_test.abs().max())



# Summarize the results
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["Squared correlation coefficient (r^2)", r_squared])
table.add_row(["Mean absolute relative residual", mean_abs_rel_residual])
table.add_row(["Maximum of absolute relative residuals", max_abs_rel_residual])

print(table)



# # Graphical Validation
# plt.scatter(x_test[:, 0].detach().numpy(), x_pred_test[:, 0].detach().numpy(), label='X Position')
# plt.scatter(x_test[:, 1].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Y Position')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.legend()
# plt.title('Actual vs Predicted Plot')
# plt.show()