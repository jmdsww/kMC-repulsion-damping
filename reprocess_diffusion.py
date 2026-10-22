"""Post-processing: entanglement density from the diffusion coefficient.

Reads msd.csv from every finished run, extracts D from the diffusive part of
the MSD curve, sets Phi_e = D_0 / D and writes summary_diffusion.csv.
The kMC simulations are not rerun.
"""
import os
import glob
import json
import numpy as np
import pandas as pd


# diffusion coefficient from the MSD curve

def extract_diffusion(msd, times, t_start_frac=0.5, t_end_frac=0.9):
    """Fit D from the diffusive part of the MSD curve, MSD(t) = 6 D t + b.

    msd, times: MSD values and the matching lag times.
    t_start_frac, t_end_frac: fit window as a fraction of the curve length.
    Returns D, or NaN when the window is too short or has no valid points.
    """
    if len(msd) < 20:
        return np.nan

    n = len(msd)
    i0 = int(n * t_start_frac)
    i1 = int(n * t_end_frac)

    if i1 <= i0:
        return np.nan

    t_fit = times[i0:i1]
    msd_fit = msd[i0:i1]

    valid = (t_fit > 0) & (msd_fit > 0)
    if not np.any(valid):
        return np.nan

    t_fit = t_fit[valid]
    msd_fit = msd_fit[valid]

    if len(t_fit) < 3:
        return np.nan

    # MSD = 6 D t + b
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    D = coeffs[0] / 6.0

    return float(D)


# D for every finished run

def collect_all_diffusion(output_dir='output'):
    results = []

    sim_dirs = sorted(glob.glob(os.path.join(output_dir, 'sim_*_rep*')))
    n_total = len(sim_dirs)

    print(f"找到 {n_total} 个任务目录")

    for idx, sim_dir in enumerate(sim_dirs):
        msd_path = os.path.join(sim_dir, 'msd.csv')
        meta_path = os.path.join(sim_dir, 'metadata.json')
        summary_path = os.path.join(sim_dir, 'summary.json')

        if not all(os.path.exists(p) for p in [msd_path, meta_path, summary_path]):
            continue

        try:
            msd_df = pd.read_csv(msd_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            D = extract_diffusion(msd_df['msd'].values, msd_df['t_lag'].values)

            results.append({
                'sim_id': meta['sim_id'],
                'seed': meta['seed'],
                'gamma': meta['gamma'],
                'epsilon': meta['epsilon'],
                'T': meta['T'],
                'rho': meta['rho'],
                'z_mean': summary['z_mean'],
                'D': D,
                't_final': meta.get('t_final', np.nan),
                'n_samples': meta.get('n_samples', np.nan),
            })
        except Exception as e:
            print(f"  跳过 {sim_dir}: {e}")

        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx + 1}/{n_total}")

    df = pd.DataFrame(results)
    return df


# Phi_e = D_0 / D

def compute_Phi_e(df, ref_gamma=0.5, ref_rho=0.5,
                   ref_epsilon=1.0, ref_T=1.0):
    """Set Phi_e = D_0 / D.

    D_0 comes from the reference state gamma = 0.5, rho = 0.5, epsilon = 1.0,
    T = 1.0, the sparsest and least damped case, where D_0 is the largest and
    Phi_e >= 1. If that state is missing, the smallest D in the data is used.
    """
    ref_mask = (
        (df['gamma'] == ref_gamma) &
        (df['rho'] == ref_rho) &
        (df['epsilon'] == ref_epsilon) &
        (df['T'] == ref_T)
    )

    if ref_mask.sum() == 0:
        print(f"警告：找不到参考条件 γ={ref_gamma}, ρ={ref_rho}, "
              f"ϵ={ref_epsilon}, T={ref_T}")
        print("使用全体最小 D 作为参考")
        D_0 = df['D'].min()
    else:
        D_0 = df[ref_mask]['D'].mean()

    print(f"\n参考扩散系数 D_0 = {D_0:.6f}")
    print(f"  （取自 γ={ref_gamma}, ρ={ref_rho}, "
          f"ϵ={ref_epsilon}, T={ref_T}）")

    df['Phi_e_D'] = D_0 / df['D']

    return df, D_0


# driver

def main():
    print("=" * 60)
    print("后处理：用扩散系数 D 定义缠结度")
    print("=" * 60)

    # collect D for every finished run
    print("\n[1/3] 提取扩散系数...")
    df = collect_all_diffusion('output')
    print(f"成功提取 {len(df)} 个任务的 D")
    print(f"有效 D: {df['D'].notna().sum()}")

    # Phi_e relative to D_0
    print("\n[2/3] 计算 Φ_e = D_0 / D...")
    df, D_0 = compute_Phi_e(df)

    # write the table
    print("\n[3/3] 保存结果...")
    output_path = 'output/summary_diffusion.csv'
    df.to_csv(output_path, index=False)
    print(f"保存到 {output_path}")

    # summary statistics
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)

    print(f"\n总任务数: {len(df)}")
    print(f"有效 D: {df['D'].notna().sum()}")
    print(f"有效 Φ_e: {df['Phi_e_D'].notna().sum()}")

    print(f"\nρ 值分布:")
    print(df['rho'].value_counts().sort_index())

    print(f"\n各 γ 的平均 D:")
    grouped = df.groupby('gamma')['D'].agg(['mean', 'std', 'count'])
    print(grouped)

    print(f"\n各 γ 的平均 Φ_e:")
    grouped = df.groupby('gamma')['Phi_e_D'].agg(['mean', 'std', 'count'])
    print(grouped)

    print(f"\n各 ρ 的平均 Φ_e:")
    grouped = df.groupby('rho')['Phi_e_D'].agg(['mean', 'std', 'count'])
    print(grouped)

    print(f"\n各 T 的平均 Φ_e:")
    grouped = df.groupby('T')['Phi_e_D'].agg(['mean', 'std', 'count'])
    print(grouped)

    print(f"\n各 ϵ 的平均 Φ_e:")
    grouped = df.groupby('epsilon')['Phi_e_D'].agg(['mean', 'std', 'count'])
    print(grouped)

    # Kramers scaling check
    print("\n" + "=" * 60)
    print("验证 Kramers 标度: D ∝ 1/γ")
    print("=" * 60)

    mask = (
        (df['epsilon'] == 1.0) &
        (df['T'] == 1.0) &
        (df['rho'] == 0.7)
    )
    sub = df[mask].dropna(subset=['D'])

    if len(sub) > 0:
        grouped = sub.groupby('gamma')['D'].agg(['mean', 'std'])
        print("\n固定 ϵ=1.0, T=1.0, ρ=0.7:")
        print(grouped)

        gammas = grouped.index.values
        D_means = grouped['mean'].values

        if len(gammas) >= 2:
            log_g = np.log(gammas)
            log_D = np.log(D_means)
            coeffs = np.polyfit(log_g, log_D, 1)
            print(f"\n拟合: D ∝ γ^({coeffs[0]:.3f})")
            print(f"理论: D ∝ γ^(-1)")
            print(f"偏差: {abs(coeffs[0] - (-1)):.3f}")


if __name__ == '__main__':
    main()