# usage: python run_md.py <start.pdb> <out_dir> [--prod_ns 200] [--save_ps 1] [--seed 0]
import ctypes, glob, sys
_lib = glob.glob(f"{sys.prefix}/lib/python3.13/site-packages/nvidia/**/libnvrtc-builtins.so.13*", recursive=True)
assert _lib, "libnvrtc-builtins not found in the venv. Reinstall nvidia-cuda-nvrtc"
ctypes.CDLL(_lib[0], mode=ctypes.RTLD_GLOBAL)

import argparse, os
import openmm as mm, openmm.app as app, openmm.unit as u

from tqdm import tqdm 
from pathlib import Path

OUT_DIR = 'robin_md_trajectories'
PEPTIDE = ['AAA', 'AAAAAA']

args = {
    'out_dir': 'robin_md_trajectories',
    'prod_ns': 0.5,
    'save_ps': 0.01,
    'seed': 0.,
    'platform': 'CUDA',
}

def run_sim(start_file):
    sample = str(start_file).split('_')[-1][:5]
    peptide = str(start_file).split('/')[-1].split('_')[0]
    out_dir = f'{OUT_DIR}/{peptide}/{sample}'
    os.makedirs(out_dir, exist_ok=True)

    start = app.PDBFile(str(start_file))
    
    ff = app.ForceField("amber14-all.xml", "implicit/obc1.xml")
    system = ff.createSystem(start.topology, nonbondedMethod=app.CutoffNonPeriodic,
                            nonbondedCutoff=2.0 * u.nanometer, constraints=None)
    T = 310 * u.kelvin
    integrator = mm.LangevinMiddleIntegrator(T, 0.3 / u.picosecond, 1.0 * u.femtosecond)
    integrator.setRandomNumberSeed(args['seed'])
    sim = app.Simulation(start.topology, system, integrator, mm.Platform.getPlatformByName(args['platform']))
    sim.context.setPositions(start.positions)

    # 1) Energy minimisation (L-BFGS)
    sim.minimizeEnergy()
    with open(f"{out_dir}/minimized.pdb", "w") as f:
        app.PDBFile.writeFile(start.topology, sim.context.getState(getPositions=True).getPositions(), f)

    # 2) Equilibration: 50 ps burn-in, not saved
    sim.context.setVelocitiesToTemperature(T, args['seed'])
    sim.step(50_000)                                 # 50 ps at 1 fs

    # 3) Production
    every = int(args['save_ps'] * 1000)                      # steps per saved frame
    sim.reporters += [
        app.DCDReporter(f"{out_dir}/traj.dcd", every),
        app.StateDataReporter(f"{out_dir}/log.csv", every * 100, step=True, time=True,
                            potentialEnergy=True, temperature=True, speed=True),
        app.CheckpointReporter(f"{out_dir}/state.chk", every * 1000),
    ]
    sim.step(int(args['prod_ns'] * 1_000_000))               # 1 ns = 1e6 steps at 1 fs

def init_worker(gpu_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get())

if __name__ == '__main__':

    from tqdm import tqdm
    import os, glob
    import multiprocessing as mp
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from run_md import run_sim, init_worker
    NVRTC_DIR = _lib[0]
    GPUS = [0, 1, 2, 3, 4]
    JOBS_PER_GPU = 2        # 33–63 atoms barely load a GPU; benchmark 1, 2, 4

    # Set in the parent BEFORE the pool starts: spawned workers inherit it at process
    # start, which is when the loader reads it. That fixes the NVRTC error for workers.
    os.environ["LD_LIBRARY_PATH"] = f"{NVRTC_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    ctx = mp.get_context("spawn")                 # never fork with CUDA
    gpu_q = ctx.Queue()
    for g in GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_q.put(g)                         # one token per worker slot

    pdbs = sorted(glob.glob("robin_samples/*/*.pdb"))
    with ProcessPoolExecutor(max_workers=len(GPUS) * JOBS_PER_GPU, mp_context=ctx,
                            initializer=init_worker, initargs=(gpu_q,)) as ex:
        futures = {ex.submit(run_sim, p): p
                for i, p in enumerate(pdbs)}
        failed = []
        with tqdm(total=len(futures), desc="MD runs", unit="run") as bar:
            for f in as_completed(futures):
                pdb = futures[f]
                try:
                    f.result()
                except Exception as e:
                    failed.append((pdb, repr(e)))
                    tqdm.write(f"FAILED: {Path(pdb).name}: {e!r}")   # prints without breaking the bar
                bar.set_postfix(failed=len(failed))
                bar.update(1)