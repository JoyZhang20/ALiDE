import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

# ================== 读取数据（你已有） ==================
xlsx_path = "../output/acc_vs.xlsx"  # ← 修改为你自己的路径
sheet_name = "Sheet1"
df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

feature_cols = [
    'area', 'circularity', 'convexity', 'clusters',
    'max_area_ratio', 'boundary_complexity', 'temporal_stability'
]
target_col = 'error'

X = df[feature_cols].values
y = df[target_col].values

# ================== 训练/验证划分 ==================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ================== SGD回归：标准化 + SGDRegressor ==================
# 说明：
# - StandardScaler：特征标准化（强烈建议）
# - SGDRegressor：SGD优化的线性回归
# - early_stopping=True：用训练集内部再切一小部分做早停（可选）
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(
        loss="squared_error",     # 回归常用
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=5000,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# ================== 验证集评估 ==================
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("==== Validation Metrics ====")
print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R^2 : {r2:.6f}")

# ================== Permutation Importance（在验证集上算） ==================
# scoring：回归推荐用 "neg_mean_absolute_error" 或 "neg_root_mean_squared_error"
# n_repeats：重复次数越大越稳定，但更慢
pi = permutation_importance(
    model,
    X_val, y_val,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

importances_mean = pi.importances_mean
importances_std = pi.importances_std

# 排序（从重要到不重要）
idx = np.argsort(importances_mean)[::-1]

print("\n==== Permutation Importance (Validation) ====")
for i in idx:
    print(f"{feature_cols[i]:>22s}: {importances_mean[i]: .6f} ± {importances_std[i]:.6f}")

# 也给你一个表，方便复制到论文/Excel
print("feature\timportance_mean\timportance_std")
for f, m, s in zip(
        np.array(feature_cols)[idx],
        importances_mean[idx],
        importances_std[idx]):
    print(f"{f}\t{m:.6f}\t{s:.6f}")


# # ================== 可选：画图 ==================
# plt.figure(figsize=(8, 4.5))
# plt.bar(range(len(feature_cols)), importances_mean[idx])
# plt.xticks(range(len(feature_cols)), np.array(feature_cols)[idx], rotation=30, ha="right")
# plt.ylabel("Permutation importance (Δ score)")
# plt.title("Permutation Importance on Validation Set")
# plt.tight_layout()
# plt.show()
