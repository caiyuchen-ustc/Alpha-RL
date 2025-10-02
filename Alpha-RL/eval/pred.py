import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

def load_dicts(base_path, filename="first_u_vectors.pt", start_step=None, end_step=None):
    """加载所有 step 的向量字典（按数值排序，可选 start_step 和 end_step）"""
    all_dicts = []
    step_names = os.listdir(base_path)

    # 只保留以 global_step_ 开头的文件夹
    step_names = [s for s in step_names if s.startswith("global_step_")]

    # 按数值排序
    step_names = sorted(step_names, key=lambda x: int(x.split("_")[-1]))

    for step_name in step_names:
        try:
            step = int(step_name.split("_")[-1])
        except ValueError:
            print(f"⚠️ 跳过 {step_name}，无法解析 step")
            continue

        # 如果设置了 start_step / end_step，就过滤
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue

        step_path = os.path.join(base_path, step_name, filename)
        if not os.path.exists(step_path):
            raise FileNotFoundError(f"❌ 文件不存在: {step_path}")

        data = torch.load(step_path, map_location="cpu")
        all_dicts.append((step, data))

    if not all_dicts:
        raise RuntimeError(f"未找到任何 step ∈ [{start_step}, {end_step}] 的 {filename} 文件")

    return all_dicts


def get_common_keys(all_dicts):
    """找到所有 step 中共同的 key"""
    common_keys = None
    for _, d in all_dicts:
        keys = set(d.keys())
        common_keys = keys if common_keys is None else common_keys & keys
    return common_keys if common_keys else set()


def gather_by_steps_for_key(all_dicts, y, key, selected_steps):
    """收集某个 key 在不同 step 的向量和对应准确率"""
    X, y_sub, used_steps = [], [], []
    for step, d in all_dicts:
        if step in selected_steps and key in d:
            v = d[key].cpu().numpy().astype(np.float64).ravel()
            if np.linalg.norm(v) < 1e-12:
                continue
            X.append(v)
            y_sub.append(y[step])
            used_steps.append(step)
    if not X:
        return np.array([]), np.array([]), []
    # import pdb
    # pdb.set_trace()
    return np.vstack(X), np.array(y_sub), used_steps


def predict_all_keys_pls(
    base_path,
    y,
    target_y=1.0,
    filename="first_u_vectors.pt",
    start_step=1,
    end_step=8,
    step_indices=None,
    save_file="predicted_vectors_y1.pt",
    min_samples=3,
    n_components=2,        # 允许多分量，与你原先可视化一致
    scale=False,           # 我们手动只标准化 X，y 不缩放
    r2_filter_file=None,
    r2_threshold=None,
    plot_file="r2_distribution.png",
    plot_kde=True,
    eps=1e-12
):
    # ---------- 加载 ----------
    all_dicts_full = load_dicts(base_path, filename=filename,start_step=start_step,end_step=end_step)
    if not all_dicts_full:
        raise RuntimeError(f"未找到任何 {filename}")

    steps_all = [s for s, _ in all_dicts_full]
    if step_indices:
        selected_steps = sorted(set(int(s) for s in step_indices))
    else:
        selected_steps = [s for s in steps_all if start_step <= s <= end_step]

    selected_steps_set = set(selected_steps)
    print(f"🧮 使用的 step: {selected_steps}")

    all_dicts = [(s, d) for (s, d) in all_dicts_full if s in selected_steps_set]
    common_keys = get_common_keys(all_dicts)
    if not common_keys:
        raise RuntimeError("未找到公共 key")

    # 可选：按外部 R² 文件过滤（口径由你提供的文件决定）
    allowed = set(common_keys)
    if r2_filter_file and os.path.exists(r2_filter_file) and r2_threshold is not None:
        with open(r2_filter_file, "r") as f:
            r2_map = json.load(f)
        allowed = {k for k in common_keys if r2_map.get(k, 0.0) >= float(r2_threshold)}
        print(f"R^2 过滤：阈值 {r2_threshold}，保留 {len(allowed)}/{len(common_keys)} 个 key。")

    predictions = {}
    r2_scores = {}  # 保存 comp1 的 R²（与原代码一致）
    ok, fail = 0, 0

    for key in common_keys:
        if key not in allowed:
            continue

        X, y_sub, used = gather_by_steps_for_key(all_dicts, y, key, selected_steps)

        if X.ndim != 2 or len(y_sub) < min_samples:
            fail += 1
            continue
        if np.std(y_sub) < eps or np.linalg.norm(X) < eps:
            print(f"⚠️ Skip {key}: 零方差/零范数")
            fail += 1
            continue

        try:
            # ---------- 标准化 X（与原代码一致） ----------
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)  # (N, D)

            # ---------- PLS：X -> y ----------
            # 有时特征/样本数会限制 n_components，上限做个安全裁剪
            n_comp_eff = int(min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
            n_comp_eff = max(1, n_comp_eff)

            pls = PLSRegression(n_components=n_comp_eff, scale=False)
            # sklearn 接受 (N,) 或 (N,1)，这里用 (N,1)
            pls.fit(X_scaled, y_sub.reshape(-1, 1))

            # 得分矩阵（T），载荷（P）
            T = pls.x_scores_                  # (N, n_comp_eff)
            P = pls.x_loadings_                # (D, n_comp_eff)

            # ---------- R² 计算（与原口径一致：y ~ T[:,0]） ----------
            # comp1 单变量回归
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            reg1 = LinearRegression().fit(T[:, [0]], y_sub)
            y_hat_1 = reg1.predict(T[:, [0]])
            r2_1 = r2_score(y_sub, y_hat_1)
            r2_scores[key] = float(r2_1)

            # 仅供参考：多分量整体回归的 R²（打印，不存）
            reg_all = LinearRegression().fit(T, y_sub)
            y_hat_all = reg_all.predict(T)
            r2_all = r2_score(y_sub, y_hat_all)

            # ---------- 预测 y=target_y 的向量 ----------
            a = float(reg_all.intercept_)
            beta = reg_all.coef_.astype(np.float64)  # (n_comp_eff,)

            if n_comp_eff == 1:
                # 标量反解
                if abs(beta[0]) < eps:
                    raise RuntimeError("beta≈0，无法反解 t*")
                t_star = np.array([(target_y - a) / beta[0]], dtype=np.float64)  # (1,)
            else:
                # 最小二乘投影到 beta 方向
                beta_norm2 = float(np.dot(beta, beta))
                if beta_norm2 < eps:
                    raise RuntimeError("‖beta‖≈0，无法反解 t*")
                t_star = ((target_y - a) / beta_norm2) * beta  # (n_comp_eff,)

            # 回到 X 的标准化空间：x_scaled* = t* P^T
            X_scaled_pred = np.dot(t_star.reshape(1, -1), P.T)  # (1, D)
            X_pred = scaler_X.inverse_transform(X_scaled_pred)[0]  # (D,)

            predictions[key] = torch.tensor(X_pred, dtype=torch.float32)

            # ---------- 诊断输出 ----------
            idx_best = int(np.argmax(y_sub))
            x_best = X[idx_best]
            denom = np.linalg.norm(X_pred) * np.linalg.norm(x_best)
            cos_sim = float(np.dot(X_pred, x_best) / denom) if denom > eps else 0.0

            print(
                f"[{key}] R^2(comp1)={r2_1:.4f} | R^2(all)={r2_all:.4f} | "
                f"Pred‖x‖={np.linalg.norm(X_pred):.6f} | "
                f"CosSim(pred, best)={cos_sim:.6f} | steps={used}"
            )
            ok += 1
        except Exception as e:
            print(f"⚠️ Skip {key} (steps={used}): {e}")
            fail += 1

    # ---------- 保存 ----------
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    torch.save(predictions, save_file)
    print(f"✅ Saved {len(predictions)} predicted vectors to: {save_file} (ok={ok}, fail={fail})")

    # import pdb
    # pdb.set_trace()
    return predictions, r2_scores



# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    base_path = ""
    y = np.array([], dtype=np.float64)

    for i in range(10,12):
        predict_all_keys_pls(
            base_path=base_path,
            y=y,
            target_y=0.8,
            filename="first_au_vectors.pt",
            start_step=5,
            end_step=i,
            save_file=f"",
            min_samples=3,
            n_components=1,
            scale=False,
            plot_file="r2_distribution.png"
        )
