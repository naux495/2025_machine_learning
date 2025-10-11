import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 載入資料
df = pd.read_csv("classification.csv")

# 只用標籤0和1，且特徵是經度和緯度
X = df[['lon', 'lat']].values
y = df['label'].values

# 過濾只保留標籤0和1的資料
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

# 分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 計算mu, phi, sigma
def fit_gda(X, y):
    labels = np.unique(y)
    mu = np.array([X[y == l].mean(axis=0) for l in labels])
    phi = np.array([np.mean(y == l) for l in labels])
    sigma = sum([((X[y == l] - mu[i]).T @ (X[y == l] - mu[i])) for i, l in enumerate(labels)]) / len(X)
    return mu, phi, sigma, labels

mu, phi, sigma, classes = fit_gda(X_train, y_train)

# 預測函式
def predict_gda(X, mu, phi, sigma, classes):
    inv_sigma = np.linalg.inv(sigma)
    probs = []
    for i, class_label in enumerate(classes):
        a = -0.5 * np.sum((X - mu[i]) @ inv_sigma * (X - mu[i]), axis=1)
        a += np.log(phi[i])
        probs.append(a)
    probs = np.vstack(probs)
    return classes[np.argmax(probs, axis=0)]

# 預測測試集
y_pred = predict_gda(X_test, mu, phi, sigma, classes)

# 計算準確率
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy: {:.4f}".format(accuracy))

# 顏色設定
colors = {0: 'blue', 1: 'red'}

# 畫訓練集圖
plt.figure(figsize=(8,6))
for cls in classes:
    plt.scatter(X_train[y_train == cls, 0], X_train[y_train == cls, 1], 
                c=colors[cls], label=f'Class {cls}', s=20)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Training Set - GDA Classes 0 and 1")
plt.legend()
plt.show()

# 畫測試集圖
plt.figure(figsize=(8,6))
for cls in classes:
    plt.scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], 
                c=colors[cls], label=f'Class {cls}', s=20)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Test Set - GDA Classes 0 and 1")
plt.legend()
plt.show()
