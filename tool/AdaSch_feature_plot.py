import numpy as np
import matplotlib.pyplot as plt

# ================= 数据 =================
features = [
    "convexity",
    "circularity",
    "area",
    "num",
    "max_area_ratio",
    "clusters",
    "boundary_complexity",
]

importance_mean = np.array([
    0.00982,
    0.008997,
    0.006804,
    0.002128,
    0.001261,
   -0.000724,
   -0.001181,
])

importance_std = np.array([
    0.00286,
    0.001337,
    0.001304,
    0.001095,
    0.001309,
    0.00112,
    0.000398,
])

# ================= 排序（从重要到不重要） =================
idx = np.argsort(importance_mean)
features = np.array(features)[idx]
importance_mean = importance_mean[idx]
importance_std = importance_std[idx]

y_pos = np.arange(len(features))

# ================= 作图 =================
plt.figure(figsize=(7, 4.5))

plt.barh(
    y_pos,
    importance_mean,
    xerr=importance_std,
    align="center",
    capsize=4
)

# 0 参考线（非常重要）
plt.axvline(0, linewidth=1)

plt.yticks(y_pos, features)
plt.xlabel("Permutation Importance (Δ MAE)")
plt.title("Feature Importance with Uncertainty")

plt.tight_layout()
plt.show()
