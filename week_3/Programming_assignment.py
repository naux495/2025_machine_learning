import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 定義 Runge 函數及其導函數
def runge_function(x):
    return 1 / (1 + 25 * x ** 2)
def runge_derivative(x):
    # 導函數: f'(x) = -50x / (1 + 25x^2)^2
    return -50 * x / (1 + 25 * x ** 2) ** 2

# 生成全部100個點
x_all = np.linspace(-1, 1, 100)
y_all = runge_function(x_all)

# 拆分為訓練集和驗證集，比例50%訓練、50%驗證
x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.5, random_state=42)

# 轉換為 PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
x_val_tensor = torch.FloatTensor(x_val).unsqueeze(1)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

# 定義神經網路架構
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.Tanh()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1500
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_output = model(x_val_tensor)
        val_loss = criterion(val_output, y_val_tensor)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}")

# 排序驗證集點
sort_idx = np.argsort(x_val)
x_val_sorted = x_val[sort_idx]
y_val_sorted = y_val[sort_idx]
y_pred_sorted = model(torch.FloatTensor(x_val_sorted).unsqueeze(1)).detach().numpy().squeeze()

plt.figure(figsize=(10,5))
plt.plot(x_val_sorted, y_val_sorted, label="True Runge Function")
plt.plot(x_val_sorted, y_pred_sorted, label="NN Prediction")
plt.legend()
plt.title("Runge Function vs Neural Network Prediction")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Training/Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# 誤差計算 (function)
mse = np.mean((y_val_sorted - y_pred_sorted) ** 2)
max_error = np.max(np.abs(y_val_sorted - y_pred_sorted))
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Maximum Error: {max_error:.6f}")

# NN函數導函數計算（autograd）
x_val_torch = torch.FloatTensor(x_val_sorted).unsqueeze(1)
x_val_torch.requires_grad = True
with torch.no_grad():
    pass # 保證參數不更新
# 重新前向需能反向
y_pred = model(x_val_torch)
y_pred.backward(torch.ones_like(y_pred))
nn_deriv = x_val_torch.grad.detach().numpy().squeeze()

# Runge解析導函數
runge_deriv = runge_derivative(x_val_sorted)

# 誤差計算 (derivative)
mse_deriv = np.mean((runge_deriv - nn_deriv) ** 2)
max_error_deriv = np.max(np.abs(runge_deriv - nn_deriv))
print(f"Derivative MSE: {mse_deriv:.6f}")
print(f"Derivative Maximum Error: {max_error_deriv:.6f}")

plt.figure(figsize=(10,5))
plt.plot(x_val_sorted, runge_deriv, label="True Runge Derivative")
plt.plot(x_val_sorted, nn_deriv, label="NN Prediction Derivative")
plt.legend()
plt.title("Runge Function Derivative vs Neural Network Derivative")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.show()

