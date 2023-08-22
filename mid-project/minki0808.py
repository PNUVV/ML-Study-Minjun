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
func = ODEF(2, 30, 2)
optimizer = torch.optim.Adam(func.parameters(), lr=0.001)

# Latin Hypercube Design (LHD) to generate samples
n_samples = 1000
lhd_samples = lhs(2, samples=n_samples)

# Using LHD samples for circular motion
# float64 타입 이슈
t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32) # dtype added here
x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)


# Splitting into training and test sets
train_size = int(0.8 * len(x_lhd))

x_train, x_test = x_lhd[:train_size], x_lhd[train_size:]
t_train, t_test = t_lhd[:train_size], t_lhd[train_size:]



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
plt.plot(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(), label='True trajectory')
x_pred = odeint(func, x_train[0], t_train)
plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Motion')
plt.show()

# Testing the model
x_pred_test = odeint(func, x_test[0], t_test).squeeze()
plt.plot(x_test[:, 0].detach().numpy(), x_test[:, 1].detach().numpy(), label='True test trajectory')
plt.plot(x_pred_test[:, 0].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Predicted test trajectory')
plt.legend()
plt.title('Test Data Prediction')
plt.show()

# Contour plot for specific time and acceleration
# This part can be customized based on the specific modeling of the system
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.sqrt(X**2 + Y**2)  # Example: distance from the origin
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Contour Plot (Example)')
plt.xlabel('Initial Position')
plt.ylabel('Initial Velocity')
plt.show()

# Testing the model
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



# from scipy.stats import probplot, normaltest
# import seaborn as sns

# # Generating example data for demonstration
# np.random.seed(42)
# x_values = np.linspace(0, 2 * np.pi, 100)
# y_actual = np.sin(x_values) + 0.1 * np.random.normal(size=x_values.shape)
# y_predicted = np.sin(x_values) + 0.2 * np.random.normal(size=x_values.shape)

# # Calculating residuals for the example data
# residuals_example = y_actual - y_predicted

# # Plotting residuals
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals_example, kde=True, bins=20)
# plt.title('Distribution of Residuals (Example)')
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
# plt.show()

# # Q-Q plot of residuals
# probplot(residuals_example, plot=plt)
# plt.title('Q-Q Plot of Residuals (Example)')
# plt.show()

# # Normality test
# k2, p_value_example = normaltest(residuals_example)
# print("Normality test p-value (Example):", p_value_example)


# from scipy.interpolate import griddata

# # Create a meshgrid for the contour plot
# num_points = 100
# X, Y = np.meshgrid(np.linspace(-1, 0, num_points), np.linspace(0.25, 1, num_points))

# # Create points and values for interpolation
# points = x_test.detach().numpy()
# values_true = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
# values_pred = np.sqrt(x_pred_test[:, 0].detach().numpy()**2 + x_pred_test[:, 1].detach().numpy()**2)

# # Interpolate the values for the contour plot
# Z_true = griddata(points, values_true, (X, Y), method='cubic')
# Z_pred = griddata(points, values_pred, (X, Y), method='cubic')

# # Plot the true contour
# plt.contour(X, Y, Z_true, levels=20, colors='r', linestyles='dashed', label='True motion')

# # Plot the predicted contour
# plt.contour(X, Y, Z_pred, levels=20, colors='b', linestyles='solid', label='Predicted motion')

# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.title('Contour Plot Comparison: True vs Predicted Motion')
# plt.legend()
# plt.show()


# print("X range in training:", x_train[:, 0].min().item(), "to", x_train[:, 0].max().item())
# print("Y range in training:", x_train[:, 1].min().item(), "to", x_train[:, 1].max().item())

# print("X range in prediction:", x_pred_test[:, 0].min().item(), "to", x_pred_test[:, 0].max().item())
# print("Y range in prediction:", x_pred_test[:, 1].min().item(), "to", x_pred_test[:, 1].max().item())
