import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, max_error
import matplotlib.pyplot as plt

# 讀取資料並清除空值
df = pd.read_csv('regression.csv')
df = df.dropna(subset=['lon', 'lat', 'value'])
X = df[['lon', 'lat']]
y = df['value']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# 預測
y_pred = regressor.predict(X_test)

# 計算MSE與Max error
mse = mean_squared_error(y_test, y_pred)
max_err = max_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Max Error: {max_err}")

# 輸出每個點的經緯度、真實值與預測值
output_df = X_test.copy()
output_df['true_value'] = y_test.values
output_df['predicted_value'] = y_pred
output_df.to_csv('regression_prediction_with_coordinates.csv', index=False)

print("經緯度及其真實溫度與預測溫度已輸出到 regression_prediction_with_coordinates.csv")

# 繪圖保持不變
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Random Forest Regression: True vs Predicted Temperature')
plt.tight_layout()
plt.savefig('regression_result.png')
plt.show()
