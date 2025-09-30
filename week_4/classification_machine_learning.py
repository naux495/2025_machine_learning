import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv('classification.csv')

# 檢查與前處理
df = df.dropna(subset=['lon', 'lat', 'label'])  # 去除空值
X = df[['lon', 'lat']]
y = df['label']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練隨機森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型預測
y_pred = clf.predict(X_test)

# 準確度與詳細指標
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 畫圖：左為真實標籤，右為預測標籤
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test['lon'], X_test['lat'], c=y_test, cmap='bwr', s=10, alpha=0.6)
plt.title('True Labels')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(1, 2, 2)
plt.scatter(X_test['lon'], X_test['lat'], c=y_pred, cmap='bwr', s=10, alpha=0.6)
plt.title('Predicted Labels')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.savefig('classification_rf_visualization.png')
plt.show()
