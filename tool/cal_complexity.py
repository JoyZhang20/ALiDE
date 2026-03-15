from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from tool.JoyTool import *
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_complexity(contours, param_num=5, min_area=100, eps=30, min_samples=2):
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not valid_contours:
        return 0.0,[0,0,0,0,0][:param_num]

    total_area = sum(cv2.contourArea(cnt) for cnt in valid_contours)

    circularities = []
    convexities = []
    max_area = 0

    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        circularities.append(circularity)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        convexities.append(convexity)

        max_area = max(max_area, area)

    avg_circularity = np.mean(circularities)
    avg_convexity = np.mean(convexities)
    max_area_ratio = max_area / total_area if total_area > 0 else 0

    # 聚类：空间分布复杂度
    points = [tuple(cnt.mean(axis=0)[0]) for cnt in valid_contours]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    cluster_count = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    # 归一化
    frame_area = 640 * 480
    normalized_area = total_area / frame_area
    normalized_circularity = 1 - avg_circularity
    normalized_convexity = 1 - avg_convexity
    normalized_clusters = cluster_count / 10
    normalized_max_area_ratio = max_area_ratio

    feature_vector = [[
        normalized_area,
        normalized_circularity,
        normalized_convexity,
        normalized_clusters,
        normalized_max_area_ratio
    ]]

    return predict_complexity([feature_vector[0][:param_num]]), feature_vector[0][:param_num]
    # return feature_vector


def predict_complexity(sample):
    # regressor = MLPRegressor(max_iter=1000, tol=1e-3, learning_rate='adaptive', eta0=0.01)
    # start = time.time()
    with open('data/regressor_model.pkl', 'rb') as f:
        regressor = pickle.load(f)
    # sample = [[0.43, 0.276, 0.754, 0.3]]
    predicted_error = regressor.predict(sample)
    # end = time.time()
    # print(f"耗时：{end-start}")
    return round(predicted_error[0],3)


def calculate_complexity(frame, back_sub, min_area=100, eps=30, min_samples=2):
    # 应用背景减除
    fg_mask = back_sub.apply(frame)

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return get_complexity(contours)

def retrain_regressor(X_new, y_new):
    with open('data/regressor_model.pkl', 'rb') as f:
        regressor = pickle.load(f)
    regressor.partial_fit(X_new, y_new)
    with open('data/regressor_model.pkl', 'wb') as f:
        pickle.dump(regressor, f)

def compute_spearman_correlation(X, y, feature_names):
    """
    计算每个特征与标签之间的 Spearman 相关系数
    """
    print("\n📊 Spearman Rank Correlation Analysis")
    print("-" * 50)

    results = []
    for i, name in enumerate(feature_names):
        coef, p_value = spearmanr(X[:, i], y)
        results.append((name, coef, p_value))
        print(f"Feature: {name:15s} | Spearman ρ = {coef:+.4f} | p-value = {p_value:.4e}")

    return results
def train_regressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='adaptive', eta0=0.01)
    # regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)

    # compute_spearman_correlation(X, y, feature_cols)

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ 预测完成！")
    print(f"📉 MSE（均方误差）: {mse:.4f}")
    print(f"📈 R² Score（拟合优度）: {r2:.4f}")
    # with open('data/regressor_model.pkl', 'wb') as f:
    #     pickle.dump(regressor, f)




def Init_regressor(param_num=5):
    """读取excel中的数据训练一个回归器"""
    xlsx_path = "output/acc_vs.xlsx"  # ← 修改为你自己的路径
    sheet_name = "Sheet1"  # ← 如果有多个表，请指定名称或索引
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    feature_cols = ['area', 'circularity', 'convexity', 'clusters', 'max_area_ratio']  # ← 修改为你的特征列
    feature_cols = feature_cols[:param_num]
    target_col = 'error'  # ← 目标是小模型和大模型结果的误差
    X = df[feature_cols].values
    y = df[target_col].values
    train_regressor(X,y)
    print("成功重置回归器！")

def feature_change_threshold_ablation(X, y,
                                      test_size=0.2,
                                      random_state=42):
    key_th = 0.5
    eps = 0.09
    # --- split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # --- train regressor ---
    regressor = SGDRegressor(
        max_iter=1000, tol=1e-3, learning_rate='adaptive', eta0=0.01, random_state=random_state
    )
    regressor.fit(X_train, y_train)

    # =========================================================
    # 1) Baseline: no temporal reuse, predict every sample
    # =========================================================
    t0 = time.perf_counter()
    # =========================================================
    # 1) Baseline: no temporal reuse, predict every sample (frame-wise)
    # =========================================================
    y_pred_base = np.zeros(len(X_test), dtype=np.float32)
    base_calls = 0
    base_time = 0.0

    for i, x in enumerate(X_test):
        st = time.perf_counter()
        pred = regressor.predict(x.reshape(1, -1))[0]
        ed = time.perf_counter()

        y_pred_base[i] = pred
        base_time += (ed - st)
        base_calls += 1

    t1 = time.perf_counter()
    base_time = t1 - t0
    base_calls = len(X_test)
    # print(f"y_pred_base={y_pred_base}")
    # print(f"y_pred_base={y_pred_base}")
    # key_pred_base = (y_pred_base > key_th).astype(np.int32)
    # print(f"len(y_pred_base)={len(y_pred_base)}")
    key_gt = (y_test > key_th).astype(np.int32)  # 如果你把“关键数据”定义为真实error>0.2，这样对齐
    key_pred_base = (y_pred_base > key_th).astype(np.int32)
    base_key_acc = (key_pred_base == key_gt).mean()
    key_gt = (y_test > key_th).astype(np.int32)  # 如果你把“关键数据”定义为真实error>0.2，这样对齐
    print(f"len(key_pred_base)={sum(key_gt)}")

    # =========================================================
    # 2) Temporal reuse: if ||x_t - x_{t-1}|| < eps, reuse prev
    #    IMPORTANT: X_test needs to be in time order to make sense.
    # =========================================================
    y_pred_reuse = np.zeros(len(X_test), dtype=np.float32)
    pred_calls = 0
    pred_time = 0.0
    reuse_cnt = 0

    prev_x = None
    prev_pred = None

    for i, x in enumerate(X_test):
        if prev_x is None:
            # first frame must predict
            st = time.perf_counter()
            pred = regressor.predict(x.reshape(1, -1))[0]
            ed = time.perf_counter()
            pred_time += (ed - st)
            pred_calls += 1
        else:
            diff = np.linalg.norm(x - prev_x, ord=2)  # ||x_t - x_{t-1}||
            # print(f"diff={diff}")
            if diff < eps:
                pred = prev_pred              # reuse: g_hat_t = g_hat_{t-1}
                reuse_cnt += 1
            else:
                st = time.perf_counter()
                pred = regressor.predict(x.reshape(1, -1))[0]  # invoke: f_theta(x_t)
                ed = time.perf_counter()
                pred_time += (ed - st)
                pred_calls += 1

        y_pred_reuse[i] = pred
        prev_x = x
        prev_pred = pred

    key_pred_reuse = (y_pred_reuse > key_th).astype(np.int32)

    reuse_mae = mean_absolute_error(y_test, y_pred_reuse)
    reuse_rmse = mean_squared_error(y_test, y_pred_reuse) ** 0.5
    reuse_key_acc = (key_pred_reuse == key_gt).mean()

    # --- baseline / reuse 的 RoI 过滤结果（是否复杂 RoI）---
    key_pred_base = (y_pred_base > key_th).astype(np.int32)
    key_pred_reuse = (y_pred_reuse > key_th).astype(np.int32)

    # 1) 过滤RoI“准确性”：复用后相对 baseline 的决策一致性（不是对真实值）
    decision_agree = (key_pred_reuse == key_pred_base).mean()

    # 2) 也给一个集合层面的稳定性：Jaccard（可选但很有用）
    base_set = set(np.where(key_pred_base == 1)[0].tolist())
    reuse_set = set(np.where(key_pred_reuse == 1)[0].tolist())
    inter = len(base_set & reuse_set)
    union = len(base_set | reuse_set)
    jaccard = inter / union if union > 0 else 1.0  # 都为空则视为完全一致

    # --- 开销统计（baseline vs reuse）---
    # baseline：predict一次性向量化；reuse：逐帧条件调用predict（pred_only）
    speedup_calls = base_calls / max(pred_calls, 1)
    speedup_time = base_time / max(pred_time, 1e-12)
    print(f"{round(float(decision_agree),4)}\t{round(base_key_acc,4)}\t{round(reuse_key_acc,4)}\t{round(float(base_time),4)}\t{round(float(pred_time),4)}\t{round(float(speedup_time),4)}")
    summary = {
        # ====== Accuracy (relative to baseline) ======
        "decision_agreement(reuse_vs_base)": float(decision_agree),
        # "jaccard_complex_roi_set(reuse_vs_base)": float(jaccard),
        "base_key_acc": base_key_acc,
        "reuse_key_acc":reuse_key_acc,
        # ====== Overhead ======
        # "BASE_predict_calls": int(base_calls),
        "BASE_total_time_sec": float(base_time),
        # "BASE_avg_time_per_pred_ms": float(base_time / max(base_calls, 1) * 1000.0),

        # "REUSE_predict_calls": int(pred_calls),
        # "REUSE_reused_frames": int(reuse_cnt),
        # "REUSE_reuse_ratio": float(reuse_cnt / max(len(X_test), 1)),
        "REUSE_total_time_sec(pred_only)": float(pred_time),
        # "REUSE_avg_time_per_pred_ms(pred_only)": float(pred_time / max(pred_calls, 1) * 1000.0),

        # "speedup_calls(BASE/REUSE)": float(speedup_calls),
        "speedup_time(BASE/REUSE_pred_only)": float(speedup_time),
    }


    return regressor, summary

if __name__ == '__main__':
    xlsx_path = "../output/acc_vs.xlsx"  # ← 修改为你自己的路径
    sheet_name = "Sheet1"  # ← 如果有多个表，请指定名称或索引
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    feature_cols = ['area', 'circularity', 'convexity', 'clusters', 'max_area_ratio','boundary_complexity','temporal_stability']  # ← 修改为你的特征列
    target_col = 'error'  # ← 目标是小模型和大模型结果的误差
    X = df[feature_cols].values
    y = df[target_col].values
    # train_regressor(X,y)

    regressor, summary = feature_change_threshold_ablation(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # 打印对比结果
    for k, v in summary.items():
        print(f"{k}: {v}")

    """添加一组数据重训练回归器"""
    # X=[[0.44,0.187,0.819,0.4],[0.53,0.102,0.774,0.9],[0.35,0.068,0.766,0.4]]
    # y=[0.175,0.381,0.308]
    # retrain_regressor(X,y)



