import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def enforce_quat_continuity_and_positive_w(q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_xyzw: (N,4) in [x,y,z,w]
    - normalize
    - enforce time continuity: if dot(q[i], q[i-1]) < 0 then flip
    - enforce qw >= 0 (per-sample, as required)
    """
    q = q_xyzw.copy()

    # normalize explicitly
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1e-12, None)

    # continuity first
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]

    # enforce qw >= 0 (per sample)
    for i in range(len(q)):
        if q[i, 3] < 0:
            q[i] = -q[i]

    # normalize again (safety)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1e-12, None)

    return q


def solve_question_1(
    csv_path="documents/tracking.csv",
    out_dir="solutions/q1",
    tracking_quat_is_WB=True,  # True: tracking gives ^W q_B ; False: tracking gives ^B q_W
):
    # ---------- 1) read data ----------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"tracking.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    t = df["t"].values.astype(float)
    q_track = df[["qx", "qy", "qz", "qw"]].values.astype(float)  # [x,y,z,w]

    # ---------- 2) build ^B R_D(t) ----------
    omega = 0.5
    alpha = np.pi / 12

    cos_wt = np.cos(omega * t)
    sin_wt = np.sin(omega * t)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    R_BD = np.zeros((len(t), 3, 3), dtype=float)
    R_BD[:, 0, 0] = cos_wt
    R_BD[:, 0, 1] = -sin_wt * cos_a
    R_BD[:, 0, 2] = sin_wt * sin_a

    R_BD[:, 1, 0] = sin_wt
    R_BD[:, 1, 1] = cos_wt * cos_a
    R_BD[:, 1, 2] = -cos_wt * sin_a

    R_BD[:, 2, 0] = 0.0
    R_BD[:, 2, 1] = sin_a
    R_BD[:, 2, 2] = cos_a

    r_BD = R.from_matrix(R_BD)

    # ---------- 3) get ^W R_B from tracking ----------
    r_track = R.from_quat(q_track)

    if tracking_quat_is_WB:
        r_WB = r_track
    else:
        r_WB = r_track.inv()

    # ---------- 4) compose: ^W R_D = ^W R_B * ^B R_D ----------
    r_WD = r_WB * r_BD
    q_WD = r_WD.as_quat()  # (N,4) [x,y,z,w]

    # ---------- 5) continuity + qw>=0 + normalize ----------
    q_WD = enforce_quat_continuity_and_positive_w(q_WD)

    # ---------- 6) save outputs ----------
    os.makedirs(out_dir, exist_ok=True)

    # save csv
    out_csv = os.path.join(out_dir, "q_WD.csv")
    out_df = pd.DataFrame({
        "t": t,
        "qx": q_WD[:, 0],
        "qy": q_WD[:, 1],
        "qz": q_WD[:, 2],
        "qw": q_WD[:, 3],
    })
    out_df.to_csv(out_csv, index=False)

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, q_WD[:, 0], label="q_x", linewidth=1.5)
    plt.plot(t, q_WD[:, 1], label="q_y", linewidth=1.5)
    plt.plot(t, q_WD[:, 2], label="q_z", linewidth=1.5)
    plt.plot(t, q_WD[:, 3], label="q_w", linewidth=1.5)

    plt.title("End-Effector Quaternion in World Frame", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Quaternion", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.xlim(t[0], t[-1])

    out_png = os.path.join(out_dir, "q_WD.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved:\n- {out_csv}\n- {out_png}")


if __name__ == "__main__":
    # 如果你不确定 tracking.csv 是 ^W q_B 还是 ^B q_W
    # 可以先保持 True 试一次；如果结果明显不合理，再改 False
    solve_question_1(tracking_quat_is_WB=True)
