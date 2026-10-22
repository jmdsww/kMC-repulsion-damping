"""Build Origin-ready CSV files.

Reads output/summary_diffusion.csv plus the msd.csv, gr.csv and analysis.csv
of the matching runs, and writes 20 tidy CSVs (one per figure) into
output/origin_data/.
"""

import os
import glob
import json
import numpy as np
import pandas as pd


INPUT_SUMMARY = 'output/summary_diffusion.csv'
OUTPUT_DIR = 'output/origin_data'

FIXED = {
    'epsilon': 1.0,
    'T': 1.0,
    'rho': 0.7,
    'gamma': 1.0,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def grouped_stats(df, x_col, y_col, fixed_dict):
    sub = df.copy()
    for k, v in fixed_dict.items():
        if k != x_col:
            sub = sub[sub[k] == v]

    grouped = sub.groupby(x_col)[y_col].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = [x_col, f'{y_col}_mean', f'{y_col}_std', 'n']
    return grouped


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  保存: {path} ({len(df)} 行)")


def find_sim_dir(df, gamma, epsilon, T, rho, rep_idx=0):
    """Return the output directory of one parameter combination."""
    mask = (
        (df['gamma'] == gamma) &
        (df['epsilon'] == epsilon) &
        (df['T'] == T) &
        (df['rho'] == rho)
    )
    sub = df[mask]
    if len(sub) <= rep_idx:
        return None
    row = sub.iloc[rep_idx]
    sim_id = int(row['sim_id'])
    seed = int(row['seed'])
    return f'output/sim_{sim_id:04d}_rep{seed}'


def align_and_save(data_dict, filename):
    """Pad the arrays to a common length and save them as one table."""
    if not data_dict:
        return
    max_len = max(len(v) for v in data_dict.values())
    aligned = {}
    for k, v in data_dict.items():
        if len(v) < max_len:
            v = np.concatenate([v, np.full(max_len - len(v), np.nan)])
        aligned[k] = v
    out = pd.DataFrame(aligned)
    save_csv(out, filename)


def main():
    print("=" * 60)
    print("为 Origin 准备数据")
    print("=" * 60)

    print(f"\n读取 {INPUT_SUMMARY}...")
    df = pd.read_csv(INPUT_SUMMARY)
    print(f"  {len(df)} 行, 列: {list(df.columns)}")

    rhos_available = sorted(df['rho'].unique())
    print(f"  ρ 值: {rhos_available}")

    # D against one parameter at a time
    print("\n[1] D vs 单参数...")

    g = grouped_stats(df, 'gamma', 'D',
                      {'epsilon': FIXED['epsilon'],
                       'T': FIXED['T'],
                       'rho': FIXED['rho']})
    save_csv(g, 'fig01_D_vs_gamma.csv')

    g = grouped_stats(df, 'rho', 'D',
                      {'gamma': FIXED['gamma'],
                       'epsilon': FIXED['epsilon'],
                       'T': FIXED['T']})
    save_csv(g, 'fig02_D_vs_rho.csv')

    g = grouped_stats(df, 'T', 'D',
                      {'gamma': FIXED['gamma'],
                       'epsilon': FIXED['epsilon'],
                       'rho': FIXED['rho']})
    save_csv(g, 'fig03_D_vs_T.csv')

    g = grouped_stats(df, 'epsilon', 'D',
                      {'gamma': FIXED['gamma'],
                       'T': FIXED['T'],
                       'rho': FIXED['rho']})
    save_csv(g, 'fig04_D_vs_epsilon.csv')

    # Phi_e against one parameter at a time
    print("\n[2] Φ_e vs 单参数...")

    g = grouped_stats(df, 'gamma', 'Phi_e_D',
                      {'epsilon': FIXED['epsilon'],
                       'T': FIXED['T'],
                       'rho': FIXED['rho']})
    save_csv(g, 'fig05_Phi_e_vs_gamma.csv')

    g = grouped_stats(df, 'rho', 'Phi_e_D',
                      {'gamma': FIXED['gamma'],
                       'epsilon': FIXED['epsilon'],
                       'T': FIXED['T']})
    save_csv(g, 'fig06_Phi_e_vs_rho.csv')

    g = grouped_stats(df, 'T', 'Phi_e_D',
                      {'gamma': FIXED['gamma'],
                       'epsilon': FIXED['epsilon'],
                       'rho': FIXED['rho']})
    save_csv(g, 'fig07_Phi_e_vs_T.csv')

    g = grouped_stats(df, 'epsilon', 'Phi_e_D',
                      {'gamma': FIXED['gamma'],
                       'T': FIXED['T'],
                       'rho': FIXED['rho']})
    save_csv(g, 'fig08_Phi_e_vs_epsilon.csv')

    # D against the mean contact number
    print("\n[3] D vs z_mean...")

    mask = (df['gamma'] == FIXED['gamma']) & \
           (df['epsilon'] == FIXED['epsilon']) & \
           (df['T'] == FIXED['T'])
    sub = df[mask]
    g = sub.groupby('rho').agg(
        z_mean_mean=('z_mean', 'mean'),
        z_mean_std=('z_mean', 'std'),
        D_mean=('D', 'mean'),
        D_std=('D', 'std'),
        n=('D', 'count')
    ).reset_index()
    save_csv(g, 'fig09_D_vs_zmean_scan_rho.csv')

    # MSD curves for different gamma
    print("\n[4] MSD 曲线（不同 γ）...")

    msd_data = {}
    for gamma in sorted(df['gamma'].unique()):
        sim_dir = find_sim_dir(df, gamma, FIXED['epsilon'],
                                FIXED['T'], FIXED['rho'], rep_idx=0)
        if sim_dir is None:
            continue
        msd_path = os.path.join(sim_dir, 'msd.csv')
        if not os.path.exists(msd_path):
            print(f"  跳过 γ={gamma}: {msd_path} 不存在")
            continue
        msd_df = pd.read_csv(msd_path)
        msd_data[f't_gamma{gamma}'] = msd_df['t_lag'].values
        msd_data[f'msd_gamma{gamma}'] = msd_df['msd'].values

    align_and_save(msd_data, 'fig10_msd_vs_gamma.csv')

    # MSD curves for different rho
    print("\n[5] MSD 曲线（不同 ρ）...")

    msd_data = {}
    for rho in sorted(df['rho'].unique()):
        sim_dir = find_sim_dir(df, FIXED['gamma'], FIXED['epsilon'],
                                FIXED['T'], rho, rep_idx=0)
        if sim_dir is None:
            continue
        msd_path = os.path.join(sim_dir, 'msd.csv')
        if not os.path.exists(msd_path):
            continue
        msd_df = pd.read_csv(msd_path)
        msd_data[f't_rho{rho}'] = msd_df['t_lag'].values
        msd_data[f'msd_rho{rho}'] = msd_df['msd'].values

    align_and_save(msd_data, 'fig11_msd_vs_rho.csv')

    # g(r) for different rho
    print("\n[6] g(r) 曲线（不同 ρ）...")

    gr_data = {}
    for rho in sorted(df['rho'].unique()):
        sim_dir = find_sim_dir(df, FIXED['gamma'], FIXED['epsilon'],
                                FIXED['T'], rho, rep_idx=0)
        if sim_dir is None:
            continue
        gr_path = os.path.join(sim_dir, 'gr.csv')
        if not os.path.exists(gr_path):
            continue
        gr_df = pd.read_csv(gr_path)
        gr_data[f'r_rho{rho}'] = gr_df['r'].values
        gr_data[f'gr_rho{rho}'] = gr_df['g_r'].values

    align_and_save(gr_data, 'fig12_gr_vs_rho.csv')

    # g(r) for different gamma
    print("\n[7] g(r) 曲线（不同 γ）...")

    gr_data = {}
    for gamma in sorted(df['gamma'].unique()):
        sim_dir = find_sim_dir(df, gamma, FIXED['epsilon'],
                                FIXED['T'], FIXED['rho'], rep_idx=0)
        if sim_dir is None:
            continue
        gr_path = os.path.join(sim_dir, 'gr.csv')
        if not os.path.exists(gr_path):
            continue
        gr_df = pd.read_csv(gr_path)
        gr_data[f'r_gamma{gamma}'] = gr_df['r'].values
        gr_data[f'gr_gamma{gamma}'] = gr_df['g_r'].values

    align_and_save(gr_data, 'fig13_gr_vs_gamma.csv')

    # D as a function of gamma and rho
    print("\n[8] 相图 D(γ, ρ)...")

    mask = (df['epsilon'] == FIXED['epsilon']) & (df['T'] == FIXED['T'])
    sub = df[mask]
    pivot = sub.groupby(['gamma', 'rho'])['D'].mean().unstack()
    pivot.index.name = 'gamma'
    pivot.columns.name = 'rho'
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'fig14_phase_D_gamma_rho.csv'))
    print(f"  保存: fig14_phase_D_gamma_rho.csv (shape: {pivot.shape})")

    # Phi_e as a function of gamma and rho
    print("\n[9] 相图 Φ_e(γ, ρ)...")

    mask = (df['epsilon'] == FIXED['epsilon']) & (df['T'] == FIXED['T'])
    sub = df[mask]
    pivot = sub.groupby(['gamma', 'rho'])['Phi_e_D'].mean().unstack()
    pivot.index.name = 'gamma'
    pivot.columns.name = 'rho'
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'fig15_phase_Phi_e_gamma_rho.csv'))
    print(f"  保存: fig15_phase_Phi_e_gamma_rho.csv (shape: {pivot.shape})")

    # energy and contacts during equilibration
    print("\n[10] 能量平衡...")

    sim_dir = find_sim_dir(df, 1.0, 1.0, 1.0, 0.7, rep_idx=0)
    if sim_dir is not None:
        analysis_path = os.path.join(sim_dir, 'analysis.csv')
        if os.path.exists(analysis_path):
            analysis = pd.read_csv(analysis_path)
            out = analysis[['time', 'E_total', 'z_mean']].copy()
            save_csv(out, 'fig16_energy_equilibration.csv')

    # MSD curves for different epsilon
    print("\n[11] MSD 曲线（不同 ε）...")

    msd_data = {}
    for eps in sorted(df['epsilon'].unique()):
        sim_dir = find_sim_dir(df, FIXED['gamma'], eps,
                                FIXED['T'], FIXED['rho'], rep_idx=0)
        if sim_dir is None:
            continue
        msd_path = os.path.join(sim_dir, 'msd.csv')
        if not os.path.exists(msd_path):
            continue
        msd_df = pd.read_csv(msd_path)
        msd_data[f't_epsilon{eps}'] = msd_df['t_lag'].values
        msd_data[f'msd_epsilon{eps}'] = msd_df['msd'].values

    align_and_save(msd_data, 'fig17_msd_vs_epsilon.csv')

    # g(r) for different epsilon
    print("\n[12] g(r) 曲线（不同 ε）...")

    gr_data = {}
    for eps in sorted(df['epsilon'].unique()):
        sim_dir = find_sim_dir(df, FIXED['gamma'], eps,
                                FIXED['T'], FIXED['rho'], rep_idx=0)
        if sim_dir is None:
            continue
        gr_path = os.path.join(sim_dir, 'gr.csv')
        if not os.path.exists(gr_path):
            continue
        gr_df = pd.read_csv(gr_path)
        gr_data[f'r_epsilon{eps}'] = gr_df['r'].values
        gr_data[f'gr_epsilon{eps}'] = gr_df['g_r'].values

    align_and_save(gr_data, 'fig18_gr_vs_epsilon.csv')

    # MSD curves for different T
    print("\n[13] MSD 曲线（不同 T）...")

    msd_data = {}
    for T in sorted(df['T'].unique()):
        sim_dir = find_sim_dir(df, FIXED['gamma'], FIXED['epsilon'],
                                T, FIXED['rho'], rep_idx=0)
        if sim_dir is None:
            continue
        msd_path = os.path.join(sim_dir, 'msd.csv')
        if not os.path.exists(msd_path):
            continue
        msd_df = pd.read_csv(msd_path)
        msd_data[f't_T{T}'] = msd_df['t_lag'].values
        msd_data[f'msd_T{T}'] = msd_df['msd'].values

    align_and_save(msd_data, 'fig19_msd_vs_T.csv')

    # g(r) for different T
    print("\n[14] g(r) 曲线（不同 T）...")

    gr_data = {}
    for T in sorted(df['T'].unique()):
        sim_dir = find_sim_dir(df, FIXED['gamma'], FIXED['epsilon'],
                                T, FIXED['rho'], rep_idx=0)
        if sim_dir is None:
            continue
        gr_path = os.path.join(sim_dir, 'gr.csv')
        if not os.path.exists(gr_path):
            continue
        gr_df = pd.read_csv(gr_path)
        gr_data[f'r_T{T}'] = gr_df['r'].values
        gr_data[f'gr_T{T}'] = gr_df['g_r'].values

    align_and_save(gr_data, 'fig20_gr_vs_T.csv')

    # files that were written
    print("\n" + "=" * 60)
    print(f"完成! 所有文件在 {OUTPUT_DIR}/")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.csv')))
    print(f"\n共生成 {len(files)} 个文件:")
    for f in files:
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f):50s} {size:>10,} bytes")


if __name__ == '__main__':
    main()