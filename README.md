# kMC-repulsion-damping
Code for kMC simulation of repulsion-damping
# kMC Simulation of WCA Particulate Systems: Repulsion and Damping Effects

This repository contains the Python simulation codes and data processing scripts for the manuscript entitled **"Effects of Repulsion and Damping on the Structure and Dynamics of WCA Particulate Systems: A Kinetic Monte Carlo Study"** .

## 📖 Overview

The codes provide a complete kinetic Monte Carlo (kMC) framework to investigate the equilibrium structure and dynamic behavior of dense 3D particulate systems with Weeks-Chandler-Andersen (WCA) purely repulsive potentials. The framework allows the extraction of the radial distribution function \(g(r)\), mean squared displacement (MSD), diffusion coefficient \(D\), and the definition of an entanglement degree \(\Phi_e = D_0/D\).

## 📂 File Structure

*   `damping_montecarlo.py` : The core kMC simulation code. It implements the Metropolis acceptance criterion, variable time-step event selection, neighbor list optimization, and trajectory data output.
*   `reprocess_diffusion.py` : Post-processing script. It reads the MSD trajectories, performs linear fitting in the diffusive regime to extract the diffusion coefficient \(D\), and calculates the entanglement degree \(\Phi_e\).
*   `prepare_origin_data.py` : Data formatting script. It processes the simulation outputs to generate clean CSV files ready for plotting figures in Origin (e.g., \(g(r)\), MSD, phase diagrams).

## ⚙️ Requirements

The codes are written in Python 3 and require the following libraries:
*   `numpy`
*   `numba` (Highly recommended for accelerating the core kMC loops)
*   `pandas`
*   `multiprocessing` (Standard library)

You can install the dependencies via:
```bash
pip install numpy numba pandas




Usage
1. Running the Simulation
Run the main script to start the Monte Carlo simulations. The script is configured to run multiple parameter combinations using parallel worker groups (Windows compatible).

bash
python damping_montecarlo.py

Outputs will be saved in the output/ directory.
The code supports breakpoint resume. If the simulation is interrupted, it will continue from the last completed configuration.

2. Post-processing (Diffusion & Entanglement)
After the simulations are finished, run the post-processing script to extract diffusion coefficients and entanglement degrees.

bash
python reprocess_diffusion.py

This will generate summary_diffusion.csv containing D and Φe for all simulations.

3. Formatting Data for Origin
To generate high-quality plots, run the data preparation script to assemble clean CSVs.

bash
python prepare_origin_data.py

The processed data will be stored in output/origin_data/, ready to be imported into Origin.





Parameters
The simulations scan the following parameter space:

Damping coefficient γ∈[0.5,10.0]

Repulsive strength ϵ∈[0.5,2.0]

Temperature T∈[0.5,2.0]

Number density ρ∈[0.5,0.7]

Total independent simulations: 1200




Contact
For any questions or bug reports, please contact:
Geng He: 1504425480@qq.com
Hui Jin: jinhui@mail.xjtu.edu.cn





## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this code for academic and commercial purposes, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software. If you use this code in your research, please cite our paper as described above.
