"""Kinetic Monte Carlo driver for the damping scan (multiprocess pools, Windows).

The scan covers 5 gamma x 4 epsilon x 4 T x 3 rho x 5 repeats = 1200 runs,
spread over three pools of 35 workers (105 workers in total). Runs that
already have a summary.json are skipped, so the driver can be restarted.

Each run writes trajectory.csv (positions plus times), analysis.csv, gr.csv,
msd.csv and summary.json; once everything finishes the results are collected
into output/summary.csv.

Usage:
    python damping_montecarlo.py

Progress:
    type output\progress.txt
    type output\simulation.log
"""

import numpy as np
from numba import njit
import os
import sys
import csv
import time
import json
import glob
import traceback
import subprocess
from datetime import datetime
from multiprocessing import Pool

# Numba-accelerated kMC kernels

@njit(cache=True)
def wca_potential(r, sigma, epsilon):
    r_cut = 2.0 ** (1.0 / 6.0) * sigma
    if r >= r_cut or r < 1e-10:
        return 0.0
    sr = sigma / r
    sr6 = sr ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6) + epsilon


@njit(cache=True)
def build_cell_list(positions, L, r_cut):
    N = positions.shape[0]
    n_cells = max(3, int(L / r_cut))
    cell_size = L / n_cells
    n_cells3 = n_cells * n_cells * n_cells

    cell_index = np.zeros(N, dtype=np.int64)
    for i in range(N):
        cx = int(positions[i, 0] / cell_size) % n_cells
        cy = int(positions[i, 1] / cell_size) % n_cells
        cz = int(positions[i, 2] / cell_size) % n_cells
        cell_index[i] = (cx * n_cells + cy) * n_cells + cz

    cell_counts = np.zeros(n_cells3, dtype=np.int64)
    for i in range(N):
        cell_counts[cell_index[i]] += 1

    cell_start = np.zeros(n_cells3 + 1, dtype=np.int64)
    for c in range(n_cells3):
        cell_start[c + 1] = cell_start[c] + cell_counts[c]

    cell_particles = np.zeros(N, dtype=np.int64)
    cell_pos = cell_start[:n_cells3].copy()
    for i in range(N):
        c = cell_index[i]
        cell_particles[cell_pos[c]] = i
        cell_pos[c] += 1

    return cell_particles, cell_start, cell_index, n_cells, cell_size


@njit(cache=True)
def build_neighbor_list(positions, L, r_verlet, max_neighbors):
    N = positions.shape[0]
    cell_particles, cell_start, cell_index, n_cells, cell_size = \
        build_cell_list(positions, L, r_verlet)

    neighbor_list = np.full((N, max_neighbors), -1, dtype=np.int64)
    n_neighbors = np.zeros(N, dtype=np.int64)

    for i in range(N):
        pi0, pi1, pi2 = positions[i, 0], positions[i, 1], positions[i, 2]
        cx = int(pi0 / cell_size) % n_cells
        cy = int(pi1 / cell_size) % n_cells
        cz = int(pi2 / cell_size) % n_cells

        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    nx = (cx + dx) % n_cells
                    ny = (cy + dy) % n_cells
                    nz = (cz + dz) % n_cells
                    c = (nx * n_cells + ny) * n_cells + nz
                    for k in range(cell_start[c], cell_start[c + 1]):
                        j = cell_particles[k]
                        if j == i:
                            continue
                        dr0 = positions[j, 0] - pi0
                        dr1 = positions[j, 1] - pi1
                        dr2 = positions[j, 2] - pi2
                        dr0 -= L * round(dr0 / L)
                        dr1 -= L * round(dr1 / L)
                        dr2 -= L * round(dr2 / L)
                        r2 = dr0 * dr0 + dr1 * dr1 + dr2 * dr2
                        if r2 < r_verlet * r_verlet:
                            if count < max_neighbors:
                                neighbor_list[i, count] = j
                                count += 1
        n_neighbors[i] = count

    return neighbor_list, n_neighbors


@njit(cache=True)
def compute_energy_change(positions, i, new_pos, L, sigma, epsilon,
                          neighbor_list, n_neighbors):
    old0, old1, old2 = positions[i, 0], positions[i, 1], positions[i, 2]
    new0, new1, new2 = new_pos[0], new_pos[1], new_pos[2]
    dU = 0.0
    for k in range(n_neighbors[i]):
        j = neighbor_list[i, k]
        pj0, pj1, pj2 = positions[j, 0], positions[j, 1], positions[j, 2]

        dr0 = pj0 - old0; dr1 = pj1 - old1; dr2 = pj2 - old2
        dr0 -= L * round(dr0 / L); dr1 -= L * round(dr1 / L); dr2 -= L * round(dr2 / L)
        r_old = np.sqrt(dr0 * dr0 + dr1 * dr1 + dr2 * dr2)

        dr0 = pj0 - new0; dr1 = pj1 - new1; dr2 = pj2 - new2
        dr0 -= L * round(dr0 / L); dr1 -= L * round(dr1 / L); dr2 -= L * round(dr2 / L)
        r_new = np.sqrt(dr0 * dr0 + dr1 * dr1 + dr2 * dr2)

        dU += wca_potential(r_new, sigma, epsilon) - \
              wca_potential(r_old, sigma, epsilon)
    return dU


@njit(cache=True)
def kmc_step(positions, L, sigma, epsilon, T, gamma, l,
             neighbor_list, n_neighbors):
    N = positions.shape[0]
    rates = np.zeros(N)
    dirs = np.zeros((N, 3))

    for i in range(N):
        d0 = np.random.randn(); d1 = np.random.randn(); d2 = np.random.randn()
        norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        d0 /= norm; d1 /= norm; d2 /= norm
        dirs[i, 0] = d0; dirs[i, 1] = d1; dirs[i, 2] = d2

        new_pos = np.empty(3)
        new_pos[0] = (positions[i, 0] + l * d0) % L
        new_pos[1] = (positions[i, 1] + l * d1) % L
        new_pos[2] = (positions[i, 2] + l * d2) % L

        dU = compute_energy_change(positions, i, new_pos, L, sigma, epsilon,
                                    neighbor_list, n_neighbors)
        if dU < 0:
            rates[i] = 1.0 / gamma
        else:
            rates[i] = np.exp(-dU / T) / gamma

    R_total = 0.0
    for i in range(N):
        R_total += rates[i]

    dt = -np.log(np.random.rand()) / R_total
    u = np.random.rand() * R_total
    cum = 0.0
    selected = N - 1
    for i in range(N):
        cum += rates[i]
        if cum >= u:
            selected = i
            break

    positions[selected, 0] = (positions[selected, 0] + l * dirs[selected, 0]) % L
    positions[selected, 1] = (positions[selected, 1] + l * dirs[selected, 1]) % L
    positions[selected, 2] = (positions[selected, 2] + l * dirs[selected, 2]) % L

    return dt, selected


@njit(cache=True)
def compute_total_energy(positions, L, sigma, epsilon, neighbor_list, n_neighbors):
    E = 0.0
    N = positions.shape[0]
    for i in range(N):
        for k in range(n_neighbors[i]):
            j = neighbor_list[i, k]
            if j <= i:
                continue
            dr0 = positions[j, 0] - positions[i, 0]
            dr1 = positions[j, 1] - positions[i, 1]
            dr2 = positions[j, 2] - positions[i, 2]
            dr0 -= L * round(dr0 / L); dr1 -= L * round(dr1 / L); dr2 -= L * round(dr2 / L)
            r = np.sqrt(dr0 * dr0 + dr1 * dr1 + dr2 * dr2)
            E += wca_potential(r, sigma, epsilon)
    return E


@njit(cache=True)
def compute_per_particle_energy(positions, L, sigma, epsilon,
                                 neighbor_list, n_neighbors):
    N = positions.shape[0]
    E = np.zeros(N)
    for i in range(N):
        e = 0.0
        for k in range(n_neighbors[i]):
            j = neighbor_list[i, k]
            dr0 = positions[j, 0] - positions[i, 0]
            dr1 = positions[j, 1] - positions[i, 1]
            dr2 = positions[j, 2] - positions[i, 2]
            dr0 -= L * round(dr0 / L); dr1 -= L * round(dr1 / L); dr2 -= L * round(dr2 / L)
            r = np.sqrt(dr0 * dr0 + dr1 * dr1 + dr2 * dr2)
            e += wca_potential(r, sigma, epsilon)
        E[i] = e
    return E


@njit(cache=True)
def compute_contact_numbers(positions, L, r_contact, neighbor_list, n_neighbors):
    N = positions.shape[0]
    z = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for k in range(n_neighbors[i]):
            j = neighbor_list[i, k]
            dr0 = positions[j, 0] - positions[i, 0]
            dr1 = positions[j, 1] - positions[i, 1]
            dr2 = positions[j, 2] - positions[i, 2]
            dr0 -= L * round(dr0 / L); dr1 -= L * round(dr1 / L); dr2 -= L * round(dr2 / L)
            r2 = dr0 * dr0 + dr1 * dr1 + dr2 * dr2
            if r2 < r_contact * r_contact:
                z[i] += 1
    return z


# Python helpers

def initialize_positions_lattice(N, L, sigma, rng):
    n = int(np.ceil(N ** (1.0 / 3.0)))
    a = L / n
    positions = np.zeros((N, 3))
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if count >= N:
                    break
                positions[count, 0] = (i + 0.5) * a
                positions[count, 1] = (j + 0.5) * a
                positions[count, 2] = (k + 0.5) * a
                count += 1
    positions += rng.normal(0, 0.05 * sigma, (N, 3))
    return positions % L


def compute_msd(trajectory, L):
    n_frames, N, _ = trajectory.shape
    msd = np.zeros(n_frames, dtype=np.float64)
    for lag in range(n_frames):
        disp = trajectory[lag:] - trajectory[:n_frames - lag]
        disp -= L * np.round(disp / L)
        r2 = np.sum(disp ** 2, axis=-1)
        msd[lag] = np.mean(r2)
    return msd


def compute_gr(trajectory, L, r_max, n_bins):
    N = trajectory.shape[1]
    dr = r_max / n_bins
    hist = np.zeros(n_bins, dtype=np.float64)
    n_frames_used = 0
    frame_step = max(1, len(trajectory) // 100)

    for frame in range(0, len(trajectory), frame_step):
        pos = trajectory[frame]
        n_frames_used += 1
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[j] - pos[i]
                d -= L * np.round(d / L)
                r = np.sqrt(np.sum(d ** 2))
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 2

    rho = N / L ** 3
    r_centers = (np.arange(n_bins) + 0.5) * dr
    shell_vol = 4 * np.pi * r_centers ** 2 * dr
    gr = hist / (n_frames_used * N * rho * shell_vol)
    return r_centers, gr


def extract_plateau(msd, times):
    if len(msd) < 10:
        return float(np.min(msd))
    log_msd = np.log(msd[1:] + 1e-10)
    log_t = np.log(times[1:] - times[0] + 1e-10)
    slopes = np.gradient(log_msd, log_t)
    mask = np.abs(slopes) < 0.15
    if np.any(mask):
        return float(np.median(msd[1:][mask]))
    return float(np.min(msd))


# single run driver

def run_one_simulation(args):
    sim_id, gamma, epsilon, T, rho, seed, output_dir, config = args
    sim_dir = os.path.join(output_dir, f"sim_{sim_id:04d}_rep{seed}")

    # resume: replay the stored summary if this run already finished
    summary_path = os.path.join(sim_dir, 'summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    try:
        N = config['N']
        sigma = config['sigma']
        l = config['l']
        r_contact = config['r_contact']
        n_equil_events = config['n_equil_events']
        n_prod_events = config['n_prod_events']
        n_samples = config['n_samples']
        max_neighbors = config['max_neighbors']
        r_skin = config['r_skin']
        r_max_gr = config['r_max_gr']
        n_bins_gr = config['n_bins_gr']

        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        L = (N / rho) ** (1.0 / 3.0)
        r_cut = 2.0 ** (1.0 / 6.0) * sigma
        r_verlet = r_cut + r_skin

        os.makedirs(sim_dir, exist_ok=True)
        log_file = os.path.join(sim_dir, "log.txt")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] 开始 sim_{sim_id:04d}_rep{seed}\n")
            f.write(f"  参数: γ={gamma}, ϵ={epsilon}, T={T}, ρ={rho}\n")

        positions = initialize_positions_lattice(N, L, sigma, rng)
        neighbor_list, n_neighbors = build_neighbor_list(
            positions, L, r_verlet, max_neighbors)
        last_rebuild = positions.copy()

        # equilibration
        t = 0.0
        t_equil_start = time.time()
        for step in range(n_equil_events):
            dt, _ = kmc_step(positions, L, sigma, epsilon, T, gamma, l,
                              neighbor_list, n_neighbors)
            t += dt
            if step % 500 == 0:
                disp = positions - last_rebuild
                disp -= L * np.round(disp / L)
                if np.max(np.linalg.norm(disp, axis=1)) > r_skin / 2:
                    neighbor_list, n_neighbors = build_neighbor_list(
                        positions, L, r_verlet, max_neighbors)
                    last_rebuild = positions.copy()
        t_equil = time.time() - t_equil_start

        # sampling interval, estimated from the production time
        r_bar = 0.5
        dt_per_event_est = gamma / (N * r_bar)
        t_total_est = n_prod_events * dt_per_event_est
        dt_sample = t_total_est / n_samples
        if dt_sample < 1e-8:
            dt_sample = 1e-8

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"  平衡完成, t={t:.4f}, 耗时 {t_equil:.1f}s\n")
            f.write(f"  dt_sample={dt_sample:.6f}\n")

        # production run with sampling
        sample_positions = []
        sample_times = []
        sample_energies = []
        sample_contact_means = []
        sample_contact_numbers = []
        sample_per_particle_energies = []

        next_sample_time = t
        n_collected = 0
        t_prod_start = time.time()

        for step in range(n_prod_events):
            dt, _ = kmc_step(positions, L, sigma, epsilon, T, gamma, l,
                              neighbor_list, n_neighbors)
            t += dt

            while t >= next_sample_time and n_collected < n_samples:
                sample_positions.append(positions.copy())
                sample_times.append(next_sample_time)

                E_total = compute_total_energy(positions, L, sigma, epsilon,
                                                neighbor_list, n_neighbors)
                sample_energies.append(E_total)

                z = compute_contact_numbers(positions, L, r_contact,
                                             neighbor_list, n_neighbors)
                sample_contact_means.append(float(np.mean(z)))
                sample_contact_numbers.append(z.copy())

                E_per = compute_per_particle_energy(positions, L, sigma, epsilon,
                                                     neighbor_list, n_neighbors)
                sample_per_particle_energies.append(E_per.copy())

                n_collected += 1
                next_sample_time += dt_sample

            if step % 500 == 0:
                disp = positions - last_rebuild
                disp -= L * np.round(disp / L)
                if np.max(np.linalg.norm(disp, axis=1)) > r_skin / 2:
                    neighbor_list, n_neighbors = build_neighbor_list(
                        positions, L, r_verlet, max_neighbors)
                    last_rebuild = positions.copy()

        t_prod = time.time() - t_prod_start

        # stack the samples into arrays
        sample_positions = np.array(sample_positions, dtype=np.float32)
        sample_times = np.array(sample_times, dtype=np.float64)
        sample_energies = np.array(sample_energies, dtype=np.float64)
        sample_contact_means = np.array(sample_contact_means, dtype=np.float64)
        sample_contact_numbers = np.array(sample_contact_numbers, dtype=np.int32)
        sample_per_particle_energies = np.array(sample_per_particle_energies,
                                                 dtype=np.float32)

        # MSD, plateau height and entanglement density
        msd = compute_msd(sample_positions, L)
        z_mean_global = float(np.mean(sample_contact_means))
        h_plateau = extract_plateau(msd, sample_times)
        Phi_e_raw = (z_mean_global / 1.0) * (sigma ** 2 / max(h_plateau, 1e-6))

        # run metadata
        metadata = {
            'sim_id': int(sim_id), 'seed': int(seed),
            'gamma': float(gamma), 'epsilon': float(epsilon),
            'T': float(T), 'rho': float(rho),
            'N': int(N), 'L': float(L), 'sigma': float(sigma), 'l': float(l),
            'r_cut': float(r_cut), 'r_contact': float(r_contact),
            'n_equil_events': int(n_equil_events),
            'n_prod_events': int(n_prod_events),
            'n_samples': int(n_collected), 'dt_sample': float(dt_sample),
            't_equil': float(t_equil), 't_prod': float(t_prod),
            't_final': float(t),
            'sample_times_first': float(sample_times[0]) if len(sample_times) > 0 else 0.0,
            'sample_times_last': float(sample_times[-1]) if len(sample_times) > 0 else 0.0,
        }
        with open(os.path.join(sim_dir, 'metadata.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # trajectory.csv: positions with time and parameters
        with open(os.path.join(sim_dir, 'trajectory.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['frame', 'time', 'gamma', 'epsilon', 'T', 'rho', 'Phi_e']
            for i in range(N):
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)
            for frame in range(n_collected):
                row = [frame, f"{sample_times[frame]:.6f}",
                       f"{gamma:.4f}", f"{epsilon:.4f}",
                       f"{T:.4f}", f"{rho:.4f}", f"{Phi_e_raw:.6f}"]
                for i in range(N):
                    row += [f"{sample_positions[frame, i, 0]:.4f}",
                            f"{sample_positions[frame, i, 1]:.4f}",
                            f"{sample_positions[frame, i, 2]:.4f}"]
                writer.writerow(row)

        # analysis.csv
        with open(os.path.join(sim_dir, 'analysis.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['frame', 'time', 'E_total', 'E_per_mean', 'z_mean', 'z_std']
            for i in range(N):
                header += [f'z_{i}', f'E_{i}']
            writer.writerow(header)
            for frame in range(n_collected):
                z = sample_contact_numbers[frame]
                E_per = sample_per_particle_energies[frame]
                row = [frame, f"{sample_times[frame]:.6f}",
                       f"{sample_energies[frame]:.6f}",
                       f"{np.mean(E_per):.6f}",
                       f"{np.mean(z):.6f}", f"{np.std(z):.6f}"]
                for i in range(N):
                    row.append(int(z[i]))
                    row.append(f"{E_per[i]:.6f}")
                writer.writerow(row)

        # gr.csv
        r_centers, gr = compute_gr(sample_positions, L, r_max_gr, n_bins_gr)
        with open(os.path.join(sim_dir, 'gr.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['r', 'g_r'])
            for r, g in zip(r_centers, gr):
                writer.writerow([f"{r:.6f}", f"{g:.6f}"])

        # msd.csv
        with open(os.path.join(sim_dir, 'msd.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['lag', 't_lag', 'msd'])
            for lag in range(len(msd)):
                t_lag = (sample_times[lag] - sample_times[0]) if lag < len(sample_times) else 0.0
                writer.writerow([lag, f"{t_lag:.6f}", f"{msd[lag]:.6f}"])

        # summary.json
        summary = {
            'sim_id': int(sim_id), 'seed': int(seed),
            'gamma': float(gamma), 'epsilon': float(epsilon),
            'T': float(T), 'rho': float(rho),
            'z_mean': float(z_mean_global),
            'h_plateau': float(h_plateau),
            'Phi_e_raw': float(Phi_e_raw),
            't_final': float(t), 'n_samples': int(n_collected),
            't_prod_walltime': float(t_prod),
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    except Exception as e:
        err_msg = f"sim_{sim_id}_rep{seed} 失败: {e}\n{traceback.format_exc()}"
        print(err_msg)
        try:
            os.makedirs(sim_dir, exist_ok=True)
            with open(os.path.join(sim_dir, 'error.txt'), 'w',
                      encoding='utf-8') as f:
                f.write(err_msg)
        except Exception:
            pass
        return {'sim_id': sim_id, 'seed': seed, 'error': str(e)}


# task list and progress helpers

def generate_all_tasks(config):
    """Build one task per (gamma, epsilon, T, rho, repeat) combination."""
    gammas = [0.5, 1.0, 2.0, 5.0, 10.0]
    epsilons = [0.5, 1.0, 1.5, 2.0]
    Ts = [0.5, 1.0, 1.5, 2.0]
    rhos = [0.5, 0.6, 0.7]   # 0.6 added for the final scan
    n_repeats = 5

    tasks = []
    sim_id = 0
    for gamma in gammas:
        for epsilon in epsilons:
            for T in Ts:
                for rho in rhos:
                    for rep in range(n_repeats):
                        seed = sim_id * 1000 + rep
                        tasks.append((sim_id, gamma, epsilon, T, rho, seed,
                                      'output', config))
                    sim_id += 1
    return tasks


def count_completed():
    return len(glob.glob('output/sim_*_rep*/summary.json'))


# worker process entry point

def run_worker_group(start, end, n_workers, config):
    os.makedirs('output', exist_ok=True)
    all_tasks = generate_all_tasks(config)
    my_tasks = all_tasks[start:end]

    log_file = f'output/worker_{start}_{end}.log'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"[{datetime.now()}] Worker [{start}, {end}), "
                f"{len(my_tasks)} 个任务, {n_workers} 个进程\n")

    with Pool(n_workers) as pool:
        pool.map(run_one_simulation, my_tasks)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now()}] Worker [{start}, {end}) 完成\n")


# main entry point

def main():
    config = {
        'N': 1000,
        'sigma': 1.0,
        'l': 0.05,
        'r_contact': 1.2,
        'r_skin': 0.3,
        'max_neighbors': 500,
        'n_equil_events': 100000,
        'n_prod_events': 1000000,
        'n_samples': 2000,
        'r_max_gr': 5.0,
        'n_bins_gr': 100,
    }

    start = end = n_workers_arg = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--start' and i + 1 < len(args):
            start = int(args[i + 1])
        elif arg == '--end' and i + 1 < len(args):
            end = int(args[i + 1])
        elif arg == '--workers' and i + 1 < len(args):
            n_workers_arg = int(args[i + 1])

    if start is not None and end is not None:
        n_workers = n_workers_arg or 35
        run_worker_group(start, end, n_workers, config)
        return

    all_tasks = generate_all_tasks(config)
    total_sims = len(all_tasks)

    n_workers_per_group = 35
    n_groups = 3
    total_workers = n_workers_per_group * n_groups

    print("=" * 60)
    print("kMC 模拟启动（多进程组版）")
    print("=" * 60)
    print(f"  参数扫描:")
    print(f"    γ: [0.5, 1.0, 2.0, 5.0, 10.0]  (5)")
    print(f"    ϵ: [0.5, 1.0, 1.5, 2.0]        (4)")
    print(f"    T: [0.5, 1.0, 1.5, 2.0]        (4)")
    print(f"    ρ: [0.5, 0.6, 0.7]             (3)")
    print(f"    重复次数: 5")
    print(f"  总任务数: {total_sims}")
    print(f"  并行组数: {n_groups}")
    print(f"  每组进程数: {n_workers_per_group}")
    print(f"  总并行进程: {total_workers}")
    print("=" * 60)

    os.makedirs('output', exist_ok=True)

    log_file = 'output/simulation.log'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"[{datetime.now()}] 开始模拟\n")
        f.write(f"总任务数: {total_sims}\n")
        f.write(f"并行组: {n_groups} × {n_workers_per_group} 进程\n")

    script_path = os.path.abspath(__file__)

    processes = []
    for i in range(n_groups):
        s = i * total_sims // n_groups
        e = (i + 1) * total_sims // n_groups

        cmd = [sys.executable, script_path,
               '--start', str(s),
               '--end', str(e),
               '--workers', str(n_workers_per_group)]

        p = subprocess.Popen(cmd)
        processes.append((i, p, s, e))
        print(f"  组 {i}: 任务 [{s}, {e}) — PID {p.pid}")
        time.sleep(1)

    t_start = time.time()
    try:
        while True:
            if all(p.poll() is not None for _, p, _, _ in processes):
                break

            n_done = count_completed()
            elapsed = time.time() - t_start
            eta = elapsed / max(1, n_done) * (total_sims - n_done) if n_done > 0 else 0

            msg = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                   f"进度 {n_done}/{total_sims} "
                   f"({100 * n_done / total_sims:.1f}%), "
                   f"已用 {elapsed / 60:.1f} 分钟, "
                   f"预计还需 {eta / 60:.1f} 分钟")
            print(msg)

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

            with open('output/progress.txt', 'w', encoding='utf-8') as f:
                f.write(f"总任务数: {total_sims}\n")
                f.write(f"已完成: {n_done} ({100 * n_done / total_sims:.1f}%)\n")
                f.write(f"已用: {elapsed / 60:.1f} 分钟\n")
                f.write(f"预计剩余: {eta / 60:.1f} 分钟\n")
                f.write(f"当前时间: {datetime.now()}\n")

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n收到中断，终止所有子进程...")
        for _, p, _, _ in processes:
            p.terminate()
        raise

    for _, p, _, _ in processes:
        p.wait()

    elapsed = time.time() - t_start
    print(f"\n全部完成! 总耗时 {elapsed / 60:.1f} 分钟")

    successful = []
    failed_dirs = []
    for sim_dir in sorted(glob.glob('output/sim_*_rep*')):
        summary_path = os.path.join(sim_dir, 'summary.json')
        error_path = os.path.join(sim_dir, 'error.txt')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    successful.append(json.load(f))
            except Exception:
                pass
        elif os.path.exists(error_path):
            failed_dirs.append(sim_dir)

    if successful:
        keys = ['sim_id', 'seed', 'gamma', 'epsilon', 'T', 'rho',
                'z_mean', 'h_plateau', 'Phi_e_raw',
                't_final', 'n_samples', 't_prod_walltime']
        with open('output/summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in successful:
                writer.writerow({k: r.get(k, '') for k in keys})
        print(f"汇总保存到 output/summary.csv ({len(successful)} 条)")

    print(f"成功: {len(successful)}, 失败: {len(failed_dirs)}")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n[{datetime.now()}] 全部完成\n")
        f.write(f"成功: {len(successful)}, 失败: {len(failed_dirs)}\n")
        f.write(f"总耗时: {elapsed / 60:.1f} 分钟\n")


if __name__ == '__main__':
    main()