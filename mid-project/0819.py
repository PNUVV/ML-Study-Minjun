# from torchdiffeq import odeint
# from prettytable import PrettyTable
# from torch import nn
# import os
# import matplotlib.pyplot as plt
# from scipy.stats import linregress
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# from pyDOE import lhs
# import torch
# import pandas as pd
# import copy
# from pandas import DataFrame
# import datetime
# import re
# from scipy.interpolate import griddata


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# print(f'Using device: {device}')

# def train_ode_model(x_data, t_data, hidden_dim,num_layers, learning_rate, epochs):
#     # Neural ODE Model Definition
#     class ODEFunc(nn.Module):
#         def __init__(self, class_num_layers, class_hidden_dim):
#             super(ODEFunc, self).__init__()
#             layers = []
#             layers.append(nn.Linear(2, class_hidden_dim))
#             for _ in range(class_num_layers):
#                 layers.append(nn.ELU())
#                 layers.append(nn.Linear(class_hidden_dim, class_hidden_dim))
#             layers.append(nn.ELU())
#             layers.append(nn.Linear(class_hidden_dim, 2))
#             self.net = nn.Sequential(*layers)
#             self.nfe = 0
    
#         def forward(self, t, x):
#             self.nfe += 1
#             return self.net(x)
    
#     start_time = datetime.datetime.now()
    


#     func = ODEFunc(num_layers, hidden_dim).to(device)
#     optimizer = torch.optim.Adam(func.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     # Training Loop
#     losses = []
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         x_pred = odeint(func, x_data[0].to(device), t_data.to(device)).squeeze()
#         loss = criterion(x_pred, x_data)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())


#     end_time = datetime.datetime.now()
#     elapsed_time = end_time - start_time
#     print(f'Training finished. Elapsed Time: {elapsed_time}')
#     return func, losses


# def plot_and_save_graph(x_data, x_pred, title, save_path):
#     """
#     Plot the true and predicted trajectory and save the plot to a file.

#     :param x_data: Ground truth data
#     :param x_pred: Predicted data
#     :param title: Title of the plot
#     :param save_path: Path to save the plot
#     """
#     plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(), label='True trajectory')
#     plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Predicted trajectory')
#     plt.legend()
#     plt.title(title)
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.savefig(save_path)
#     plt.close() # Close the plot to avoid displaying it in the notebook

# # This function will also be used inside the "train_ode_models" function.    

# def numerical_validation(x_data, x_pred):
#     """
#     Perform numerical validation on the predicted data.

#     :param x_data: Ground truth data
#     :param x_pred: Predicted data
#     :return: r_squared, mean_abs_rel_residual, max_abs_rel_residual
#     """
#     slope, intercept, r_value, _, _ = linregress(x_data.flatten().detach().numpy(), x_pred.flatten().detach().numpy())
#     r_squared = r_value**2
#     mean_abs_rel_residual = mean_absolute_error(x_data.detach().numpy(), x_pred.detach().numpy()) / (x_data.abs().mean())
#     max_abs_rel_residual = max(np.max(np.abs(x_data.detach().numpy() - x_pred.detach().numpy()), axis=0) / x_data.abs().max())
    
#     return r_squared, mean_abs_rel_residual, max_abs_rel_residual

# def train_ode_models(n_samples, hidden_dim,num_layers, learning_rate, epochs, save_path):
#     """
#     Train ODE models for different epochs and save the results.

#     :param n_samples: Number of samples
#     :param hidden_dim: Hidden dimension size
#     :param learning_rate: Learning rate
#     :param epochs_list: List of epochs for training
#     :param save_path: Path to save the results
#     :return: best_func, x_train, t_train, x_test, t_test, max_r_squared_model
#     """
#     # Latin Hypercube Design (LHD) sample generation
#     lhd_samples = lhs(2, samples=n_samples)
#     t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32)
#     x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)

#     # Data splitting
#     train_size = int(0.7 * len(x_lhd))
#     val_size = int(0.15 * len(x_lhd))
    
#     x_train = x_lhd[:train_size].to(device)
#     t_train = t_lhd[:train_size].to(device)
    
#     x_val = x_lhd[train_size:train_size + val_size].to(device)
#     t_val = t_lhd[train_size:train_size + val_size].to(device)
    
#     x_test = x_lhd[train_size + val_size:].to(device)
#     t_test = t_lhd[train_size + val_size:].to(device)

#     best_func = None
#     best_r_squared = -float('inf') # Initialize with negative infinity
#     validation_results = pd.DataFrame(columns=['Epochs', 'Hidden_Dim', 'Samples', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])

#     # Model selection and result saving logic
#     for idx in range(5):
#         func, losses = train_ode_model(x_train, t_train, hidden_dim,num_layers, learning_rate, epochs) # Function to be defined

#         x_pred_val = odeint(func, x_val[0], t_val).squeeze() # odeint to be defined
#         r_squared, mean_abs_rel_residual, max_abs_rel_residual = numerical_validation(x_val, x_pred_val)

#         # Save validation results
#         validation_results.loc[idx] = [epochs, hidden_dim, n_samples, 'Validation', r_squared, mean_abs_rel_residual, max_abs_rel_residual]

#         # Select best model
#         if r_squared > best_r_squared:
#             best_r_squared = r_squared
#             best_func = copy.deepcopy(func)

#     # Save CSV
#     validation_results.to_csv(f"{save_path}/validation_results_{hidden_dim}_{n_samples}.csv", index=False)
#     max_r_squared_model = validation_results['R_Squared'].idxmax()

#     return best_func, x_train, t_train, x_test, t_test, max_r_squared_model

# # The functions "train_ode_model" and "odeint" are to be defined as per the user's ODE model and training procedure.

# def save_results(x_pred, x_data, data_type, epochs, hidden_dim, n_samples, save_path):
#     # Plot the true trajectory and Neural ODE approximation
#     plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(), label='True trajectory')
#     plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
#     plt.legend()
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.title('2D Motion')
#     plt.savefig(f"{save_path}/{data_type}_2D_Motion_epochs_{epochs}_hidden_{hidden_dim}_samples_{n_samples}.png")
#     plt.show()

#     # Numerical Validation
#     slope, intercept, r_value, _, _ = linregress(x_data.flatten().detach().numpy(), x_pred.flatten().detach().numpy())
#     r_squared = r_value**2
#     mean_abs_rel_residual = mean_absolute_error(x_data.detach().numpy(), x_pred.detach().numpy()) / (x_data.abs().mean())
#     max_abs_rel_residual = max(np.max(np.abs(x_data.detach().numpy() - x_pred.detach().numpy()), axis=0) / x_data.abs().max())
#     table = PrettyTable()
#     table.field_names = ["Metric", "Value"]
#     table.add_row(["Squared correlation coefficient (r^2)", r_squared])
#     table.add_row(["Mean absolute relative residual", mean_abs_rel_residual])
#     table.add_row(["Maximum of absolute relative residuals", max_abs_rel_residual])
#     print(table)
    
#     # Save numerical validation
#     with open(f"{save_path}/{data_type}_numerical_validation_epochs_{epochs}_hidden_{hidden_dim}_samples_{n_samples}.txt", "w") as file:
#         file.write(str(table))
        

# def save_loss_and_validation(losses, validation_results, save_path, hidden_dim,n_samples,idx):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     ax1.plot(losses)
#     ax1.set_title('Training Loss (MSE)')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax2.plot(validation_results['Epochs'], validation_results['R_Squared'])
#     ax2.set_title('R_Squared Validation')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('R_Squared')
#     plt.savefig(f"{save_path}/loss_and_validation_{hidden_dim}_{n_samples}_{idx}.png")
#     plt.close()

# def save_2d_and_actual_vs_predicted(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

#     # 2D Motion Plot
#     ax1.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(), label='True trajectory')
#     ax1.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
#     ax1.legend()
#     ax1.set_xlabel('X Position')
#     ax1.set_ylabel('Y Position')
#     ax1.set_title('2D Motion')

#     # Actual vs Predicted Plot
#     ax2.scatter(x_data[:, 0].detach().numpy(), x_pred[:, 0].detach().numpy(), label='X Position')
#     ax2.scatter(x_data[:, 1].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Y Position')
#     ax2.plot(torch.linspace(-1, 1, 1000), torch.linspace(-1, 1, 1000), color='red', linewidth=1.2)
#     ax2.set_xlabel('Actual')
#     ax2.set_ylabel('Predicted')
#     ax2.legend()
#     ax2.set_title('Actual vs Predicted Plot')
#     plt.grid()

#     file_suffix = f"{data_type}_hidden_dim_{hidden_dim}_samples_{n_samples}_epochs_{epochs}"
#     plt.savefig(f"{save_path}/{file_suffix}.png")
#     plt.close()

# def return_numerical_validation(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
#     slope, intercept, r_value, _, _ = linregress(x_data.flatten().detach().numpy(), x_pred.flatten().detach().numpy())
#     r_squared = r_value**2
#     mean_abs_rel_residual = mean_absolute_error(x_data.detach().numpy(), x_pred.detach().numpy()) / (x_data.abs().mean())
#     max_abs_rel_residual = max(np.max(np.abs(x_data.detach().numpy() - x_pred.detach().numpy()), axis=0) / x_data.abs().max())

#     return r_squared, mean_abs_rel_residual, max_abs_rel_residual   

# def ensure_list(*values):
#     max_length = max(len(value) if isinstance(value, list) else 1 for value in values)
#     return [[value] * max_length if not isinstance(value, list) else value * (max_length // len(value)) for value in values]



# def save_2d_radius_difference_contour_fixed(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
#     # 실제 경로와 예측한 경로의 반지름 계산
#     actual_radius = torch.sqrt(x_data[:, 0]**2 + x_data[:, 1]**2)
#     pred_radius = torch.sqrt(x_pred[:, 0]**2 + x_pred[:, 1]**2)

#     # 반지름 차이 계산
#     radius_difference = actual_radius - pred_radius

#     # 컨투어 플롯을 위한 격자 데이터 생성
#     x = np.linspace(x_data[:, 0].min(), x_data[:, 0].max(), 100)
#     y = np.linspace(x_data[:, 1].min(), x_data[:, 1].max(), 100)
#     X, Y = np.meshgrid(x, y)

#     # 격자 위에 반지름 차이 보간
#     zz = griddata((x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy()), radius_difference.detach().numpy(), (X, Y), method='linear')

#     # 반지름 차이의 컨투어 플롯 생성
#     plt.contourf(X, Y, zz, levels=100, cmap="viridis")

#     # 실제 경로와 예측한 경로 그래프 생성
#     plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(), label='True trajectory')
#     plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label=f'{data_type} trajectory')
#     plt.legend()
#     plt.title(f'2D Motion Radius Difference Contour\nHidden Dim: {hidden_dim}, Samples: {n_samples}, Epochs: {epochs}')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')

#     # 플롯 저장
#     plot_path = os.path.join(save_path, f'{data_type}_radius_difference_contour.png')
#     plt.savefig(plot_path)
#     plt.close()  # 플롯을 노트북에 표시하지 않기 위해 닫아줍니다.

# def plot_radius_deviation_histogram(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
#     # Calculate the radius for actual and predicted data
#     actual_radius = torch.sqrt(x_data[:, 0]**2 + x_data[:, 1]**2)
#     pred_radius = torch.sqrt(x_pred[:, 0]**2 + x_pred[:, 1]**2)

#     # Calculate the difference in radius (deviation)
#     radius_deviation = actual_radius - pred_radius

#     # Plotting the histogram of the radius deviation
#     plt.hist(radius_deviation.detach().numpy(), bins=20, color="blue", edgecolor="black", alpha=0.7)
#     plt.title(f'Radius Deviation Histogram\nHidden Dim: {hidden_dim}, Samples: {n_samples}, Epochs: {epochs}')
#     plt.xlabel('Radius Deviation')
#     plt.ylabel('Frequency')

#     # Calculate and display mean and standard deviation
#     mean_deviation = radius_deviation.mean().item()
#     std_deviation = radius_deviation.std().item()
#     plt.axvline(mean_deviation, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_deviation:.2f}\nStd Dev: {std_deviation:.2f}')
#     plt.legend()

#     # Saving the plot
#     plot_path = os.path.join(save_path, f'{data_type}_radius_deviation_histogram.png')
#     plt.savefig(plot_path)
#     plt.close() # Close the plot to avoid displaying it in the notebook

# # You can use this function in your existing code to visualize the distribution of radius deviation between actual and predicted trajectory

# ##########################################################################################################################

# n_samples = 10000
# num_layers = 2
# hidden_dim = 32
# learning_rate = 0.01
# epochs = 300

# ##########################################################################################################################


# n_samples_list, hidden_dim_list, learning_rate_list, epochs_list = ensure_list(n_samples, hidden_dim,learning_rate, epochs)

# save_path_csv = "results"
# if not os.path.exists(save_path_csv):
#         os.makedirs(save_path_csv)

# # 결과를 저장할 DataFrame 생성
# results_df_train = pd.DataFrame(columns=['N_Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
# results_df_test = pd.DataFrame(columns=['N_Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])

# timestamp = datetime.datetime.now().strftime("%m-%d_%H %M")

# for n_samples, hidden_dim, learning_rate, epochs in zip(n_samples_list, hidden_dim_list, learning_rate_list, epochs_list):
#     # 결과 저장 경로
#     save_path = f"{save_path_csv}/{timestamp}_samples_{n_samples}_hidden_{hidden_dim}_lr_{learning_rate}"
    
#     # 디렉토리가 없으면 생성
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)


#     # 최적의 모델 훈련
    
#     # best_func, x_train, t_train, x_test, t_test, max_r_squared_model = train_ode_models(n_samples, hidden_dim,num_layers, learning_rate, epochs, save_path)
#     # x_pred_test_best = odeint(best_func, x_test[0], t_test).squeeze()
#     # x_pred_train_best = odeint(best_func, x_train[0], t_train).squeeze()
#     best_func, x_train, t_train, x_test, t_test, max_r_squared_model = train_ode_models(n_samples, hidden_dim, num_layers, learning_rate, epochs, save_path)
#     x_pred_test_best = odeint(best_func, x_test[0], t_test).squeeze().cpu()
#     x_pred_train_best = odeint(best_func, x_train[0], t_train).squeeze().cpu()
#     print(f"Training with n_samples={n_samples}, hidden_dim={hidden_dim}, lr={learning_rate}, epochs={epochs}")
    
    
#     # 훈련 데이터의 2D 그래프와 Actual vs Predicted Plot 저장
    
#     # save_2d_and_actual_vs_predicted(x_train, x_pred_train_best, 'train', hidden_dim, n_samples, epochs, save_path)

#     # train 반지름 잔차 컨투어
#     save_2d_radius_difference_contour_fixed(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
#     # train 잔차 분포표
#     plot_radius_deviation_histogram(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
#     # train 데이터의 2D 그래프와 Actual vs Predicted Plot 저장
#     save_2d_and_actual_vs_predicted(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    
#     # test 반지름 잔차 컨투어
#     save_2d_radius_difference_contour_fixed(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
#     # test 잔차 분포표
#     plot_radius_deviation_histogram(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
#     # test 데이터의 2D 그래프와 Actual vs Predicted Plot 저장
#     save_2d_and_actual_vs_predicted(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    
#     # 수치 검증 결과를 CSV로 저장 (훈련 데이터)
#     r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train =return_numerical_validation(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path_csv)
    
#     # 수치 검증 결과를 CSV로 저장 (테스트 데이터)
#     r_squared_test, mean_abs_rel_residual_test, max_abs_rel_residual_test =return_numerical_validation(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path_csv)
    
#     # 결과 DataFrame에 추가 (훈련 데이터)
#     results_df_train.loc[len(results_df_train)] = [n_samples, hidden_dim, learning_rate, epochs, 'Train', r_squared_train, mean_abs_rel_residual_train.item(), max_abs_rel_residual_train.item()]

#     # 결과 DataFrame에 추가 (테스트 데이터)
#     results_df_test.loc[len(results_df_test)] = [n_samples, hidden_dim, learning_rate, epochs, 'Test', r_squared_test, mean_abs_rel_residual_test.item(), max_abs_rel_residual_test.item()]

# # 전체 결과를 CSV 파일로 저장
# results_df_train.to_csv(f"{save_path_csv}/numerical_train_{timestamp}.csv", index=False)
# results_df_test.to_csv(f"{save_path_csv}/numerical_test_{timestamp}.csv", index=False)

from pyDOE import lhs
import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, max_error
from prettytable import PrettyTable
import datetime
import re
from scipy.interpolate import griddata


use_cuda = torch.cuda.is_available()
device = torch.device("cuda")
print(f'Using device: {device}')

class ODEF(nn.Module):
    def forward(self, t, x):
        return self.net(x)

    def __init__(self, in_dim, hidden_dim, out_dim,dropout_prob):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) 
        )

# Neural ODE definition
func = ODEF(2,3000, 2, 0.2).to(device)

#최적화
# optimizer = torch.optim.Adam(func.parameters(), lr=0.0035)
optimizer = torch.optim.AdamW(func.parameters(), lr=0.004, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.03, amsgrad=False)

n_samples = 10000
lhd_samples = lhs(2, samples=n_samples)

t_lhd = torch.linspace(0, 2*np.pi, 1000) #균일 시계열
# t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32)
x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)
train_size = int(0.7 * len(x_lhd))
val_size = int(0.15 * len(x_lhd))

x_train = x_lhd[:train_size].to(device)
t_train = t_lhd[:train_size].to(device)

x_val = x_lhd[train_size:train_size + val_size].to(device)
t_val = t_lhd[train_size:train_size + val_size].to(device)

x_test = x_lhd[train_size + val_size:].to(device)
t_test = t_lhd[train_size + val_size:].to(device)
counter = 0
best_loss = float(5) # 임의 정의
patience = 30 #이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
batch_size= 3500# 배치 크기 정의
losses = []
# func.train()
start_time=datetime.datetime.now()
for epoch in range(100):
    optimizer.zero_grad()
    for batch_start in range(0, train_size, batch_size):
        batch_x_train = x_train[batch_start:batch_start+batch_size]
        batch_t_train = t_train[batch_start:batch_start+batch_size]
        batch_x_pred_train = odeint(func, batch_x_train[0], batch_t_train).squeeze()
        batch_x_pred_train = batch_x_pred_train.to(device)
        loss = ((batch_x_pred_train - batch_x_train) ** 2).mean()
        loss.backward()
    optimizer.step()
    
    # Validation
    with torch.no_grad():
        # func.eval() # 모델을 평가 모드로 설정
        x_pred_val = odeint(func, x_val[0], t_val).squeeze()
        val_loss = ((x_pred_val - x_val) ** 2).mean()
        # func.train()
    
    losses.append(loss.item())
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(func.state_dict(), 'best_model.pt')  # Save the best model
    else:
        counter += 1

    if epoch>500:
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 10 == 0:
            print("Epoch: {:3} | Loss: {:.9f} | Val Loss: {:.9f}".format(epoch, loss.item(), val_loss.item()))
end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print(f'Training finished. Elapsed Time: {elapsed_time}')
# with torch.no_grad():
#             for name, param in func.named_parameters(): 
#                 print(f"{name}: {param.data}")# 레이어의 weight와 bias 출력
# print("Epoch: {:3} | Loss: {:.6f} | Val Loss: {:.6f}".format(epoch, loss.item(), val_loss.item()))  


# Plotting the training loss and 2D Motion Data Prediction in subplots  
fig, (ax1, ax2), = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the training loss
ax1.plot(losses)
ax1.set_title('Training Loss (MSE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
x_pred_train = odeint(func, x_train[0], t_train)
x_pred_test = odeint(func, x_test[0], t_test).squeeze()
x_pred_train_cpu = x_pred_train.cpu().detach().numpy()
x_pred_test_cpu = x_pred_test.cpu().detach().numpy()
ax2.plot(x_lhd[:, 0].cpu().detach().numpy(), x_lhd[:, 1].cpu().detach().numpy(), label='True trajectory')
ax2.plot(x_pred_train_cpu[:, 0], x_pred_train_cpu[:, 1], label='Neural ODE approximation')
ax2.plot(x_pred_val[:, 0].cpu().detach().numpy(), x_pred_val[:, 1].cpu().detach().numpy(), label='Predicted validation trajectory')
ax2.plot(x_pred_test_cpu[:, 0], x_pred_test_cpu[:, 1], label='Predicted test trajectory')
# ax2.plot(x_lhd[:, 0].detach().numpy(), x_lhd[:, 1].detach().numpy(), label='True trajectory')
# ax2.plot(x_pred_train[:, 0].detach().numpy(), x_pred_train[:, 1].detach().numpy(), label='Neural ODE approximation')
# ax2.plot(x_pred_val[:, 0].detach().numpy(), x_pred_val[:, 1].detach().numpy(), label='Predicted validation trajectory')
# ax2.plot(x_pred_test[:, 0].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Predicted test trajectory')
ax2.legend()
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_title('2D Motion Data Prediction')
plt.tight_layout()
plt.show()


# Numerical Validation
slope, intercept, r_value, _, _ = linregress(x_val.flatten().detach().numpy().to(device), x_pred_val.flatten().detach().numpy().to(device))
r_squared = r_value**2
mean_abs_rel_residual = mean_absolute_error(x_val.detach().numpy(), x_pred_val.detach().numpy()) / (x_val.abs().mean())
max_abs_rel_residual = max(np.max(np.abs(x_val.detach().numpy() - x_pred_val.detach().numpy()), axis=0) / x_val.abs().max())
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["Squared correlation coefficient (r^2)", r_squared])
table.add_row(["Mean absolute relative residual", mean_abs_rel_residual])
table.add_row(["Maximum of absolute relative residuals", max_abs_rel_residual])
print(table)
