import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# ===================== 读取数据 =====================
xlsx_path = "../output/acc_vs.xlsx"  # ← 修改为你自己的路径
sheet_name = "Sheet1"
df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

feature_cols = [
    'area', 'circularity'
]
target_col = 'error'

X = df[feature_cols].values
y = df[target_col].values.astype(np.float64)

# 可选：去掉缺失值（如果你的表里可能有NaN）
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X, y = X[mask], y[mask]

# ===================== 切分数据 =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== 标准化（强烈建议） =====================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# ===================== SGD 回归器 =====================
# loss='squared_error' 即最常见的线性回归平方损失
# early_stopping 这里不用（因为我们手动记录loss并可自定义逻辑）
reg = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=1e-4,
    learning_rate="invscaling",
    eta0=0.01,
    power_t=0.25,
    max_iter=1,        # 我们用 partial_fit 控制 epoch，这里设 1
    tol=None,          # 禁用内部提前停止
    random_state=42,
    warm_start=True
)

# ===================== 训练并记录 loss =====================
n_epochs = 300
train_losses = []
val_losses = []

best_val = float("inf")
best_state = None
patience = 30
bad_epochs = 0

for epoch in range(1, n_epochs + 1):
    # partial_fit：每次相当于再走一遍数据（一个 epoch）
    reg.partial_fit(X_train_s, y_train)

    # 记录训练/验证 loss（MSE）
    y_pred_train = reg.predict(X_train_s)
    y_pred_val = reg.predict(X_val_s)

    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)

    train_losses.append(train_mse)
    val_losses.append(val_mse)

    # 简单 early stopping（按验证集MSE）
    if val_mse < best_val - 1e-12:
        best_val = val_mse
        best_state = {
            "coef_": reg.coef_.copy(),
            "intercept_": reg.intercept_.copy(),
            "t_": reg.t_,
            "n_iter_": getattr(reg, "n_iter_", None),
        }
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"[EarlyStop] epoch={epoch}, best_val_mse={best_val:.6g}")
            break

    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch={epoch:4d}  train_mse={train_mse:.6g}  val_mse={val_mse:.6g}")

# 恢复最佳参数（可选但推荐）
if best_state is not None:
    reg.coef_ = best_state["coef_"]
    reg.intercept_ = best_state["intercept_"]
    reg.t_ = best_state["t_"]

# ===================== 保存 loss 记录 =====================
os.makedirs("../output", exist_ok=True)
loss_csv_path = "../output/sgd_losses.csv"
pd.DataFrame({
    "epoch": np.arange(1, len(train_losses) + 1),
    "train_mse": train_losses,
    "val_mse": val_losses
}).to_csv(loss_csv_path, index=False)
print(f"Saved losses to: {loss_csv_path}")

# ===================== 保存模型（含 scaler） =====================
model_path = "../output/sgd_regressor_with_scaler.joblib"
joblib.dump({"scaler": scaler, "model": reg, "feature_cols": feature_cols}, model_path)
print(f"Saved model to: {model_path}")

# ===================== （可选）如何加载并预测 =====================
# bundle = joblib.load(model_path)
# scaler2, model2 = bundle["scaler"], bundle["model"]
# X_new = df[feature_cols].values[:5]
# y_new_pred = model2.predict(scaler2.transform(X_new))
# print(y_new_pred)
