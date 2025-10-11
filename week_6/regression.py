import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#載入classification dataset並訓練模型
cls_df = pd.read_csv("classification.csv")
cls_df = cls_df.dropna(subset=["lon", "lat", "label"])
X_cls = cls_df[["lon", "lat"]]
y_cls = cls_df["label"]
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_cls_train, y_cls_train)

#載入regression dataset並訓練模型
reg_df = pd.read_csv("regression.csv")
reg_df = reg_df.dropna(subset=["lon", "lat", "value"])
X_reg = reg_df[["lon", "lat"]]
y_reg = reg_df["value"]
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg_train, y_reg_train)

# Combine datasets for prediction (use regression test set here)
X_combined = X_reg_test.copy().reset_index(drop=True)

# Piecewise defined function
def combined_model(X):
    cls_pred = cls_model.predict(X)
    reg_pred = reg_model.predict(X)
    output = np.where(cls_pred == 1, reg_pred, -999)
    return output, cls_pred

# Apply combined model
h_x, cls_pred = combined_model(X_combined)

# Save results to DataFrame
result_df = X_combined.copy()
result_df["regression_value"] = reg_model.predict(X_combined)
result_df["classification_label"] = cls_pred
result_df["combined_output_h_x"] = h_x
result_df.to_csv("combined_model_output.csv", index=False)

# Plot visualization
plt.figure(figsize=(8,6))
mask_valid = result_df["classification_label"] == 1
sc = plt.scatter(result_df.loc[mask_valid, "lon"], result_df.loc[mask_valid, "lat"],
                 c=result_df.loc[mask_valid, "regression_value"], cmap="viridis", marker="o", s=35, label="C(x)=1, R(x)")
plt.scatter(result_df.loc[~mask_valid, "lon"], result_df.loc[~mask_valid, "lat"],
            c="red", marker="x", s=25, label="C(x)=0, h(x)=-999")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Piecewise Model Result $h(\\vec{x})$")
plt.legend()

# 增加顏色-數值對應色盤
cbar = plt.colorbar(sc)
cbar.set_label('Temperature (Regression Prediction)')

plt.tight_layout()
plt.savefig("piecewise_combined_model_plot_with_colorbar.png")
plt.show()
