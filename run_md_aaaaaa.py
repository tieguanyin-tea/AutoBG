# Generate short MD trajectories for the first 200 AAAAAA Robin samples.
#
# The MD protocol is exactly the one in run_md.py (amber14-all + obc1, 310 K,
# 1 fs, 50 ps burn-in) - only the production length differs: 1000 saved frames
# (0.01 ps apart, i.e. 10 ps) instead of 0.5 ns.
#
# Speed comes purely from orchestration, not from changing simulation settings:
#   * each worker process keeps its CUDA kernels compiled and reuses them for all
#     of its samples (a fresh Simulation is created per sample, but OpenMM's
#     per-process kernel cache makes context creation ~0.15 s instead of ~1.4 s);
#   * several workers share each of the five GPUs already used by run_md.py.
#
# usage: python run_md_aaaaaa.py [--n-samples 200] [--jobs-per-gpu 8]

import ctypes
import glob
import sys

_lib = glob.glob(f"{sys.prefix}/lib/python3.13/site-packages/nvidia/**/libnvrtc-builtins.so.13*", recursive=True)
assert _lib, "libnvrtc-builtins not found in the venv. Reinstall nvidia-cuda-nvrtc"
ctypes.CDLL(_lib[0], mode=ctypes.RTLD_GLOBAL)

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import multiprocessing as mp
import openmm as mm
import openmm.app as app
import openmm.unit as u
from tqdm import tqdm

PEPTIDE = "AAAAAA"
OUT_DIR = "robin_md_trajectories"

N_SAMPLES = 200
N_FRAMES = 1000
SAVE_STEPS = 10          # 1 fs * 10 = 0.01 ps per frame
BURN_IN_STEPS = 50_000   # 50 ps, same as run_md.py

GPUS = [0, 1, 2, 3, 4]   # same GPUs as run_md.py, no more
JOBS_PER_GPU = 8

SEED = 0
TEMPERATURE = 310 * u.kelvin


def run_chunk(gpu, chunk, seed=SEED):
    """Run every (index, pdb_path) in `chunk` sequentially on one GPU."""
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"CudaDeviceIndex": str(gpu)}
    ff = app.ForceField("amber14-all.xml", "implicit/obc1.xml")

    system = None
    ref_names = None
    failures = []

    for index, pdb_path in chunk:
        out = Path(OUT_DIR) / PEPTIDE / f"{index:05d}"
        if (out / "traj.dcd").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)

        try:
            pdb = app.PDBFile(str(pdb_path))

            # The worker reuses one System, so every sample must share the topology.
            names = [(a.name, a.element.symbol) for a in pdb.topology.atoms()]
            if ref_names is None:
                ref_names = names
                system = ff.createSystem(
                    pdb.topology,
                    nonbondedMethod=app.CutoffNonPeriodic,
                    nonbondedCutoff=2.0 * u.nanometer,
                    constraints=None,
                )
            elif names != ref_names:
                raise ValueError(f"topology mismatch for {pdb_path}")

            integrator = mm.LangevinMiddleIntegrator(
                TEMPERATURE, 0.3 / u.picosecond, 1.0 * u.femtosecond
            )
            integrator.setRandomNumberSeed(seed)
            sim = app.Simulation(pdb.topology, system, integrator, platform, properties)
            sim.context.setPositions(pdb.positions)

            # 1) Energy minimisation (L-BFGS), same as run_md.py
            sim.minimizeEnergy()
            with open(out / "minimized.pdb", "w") as f:
                app.PDBFile.writeFile(
                    sim.topology,
                    sim.context.getState(getPositions=True).getPositions(),
                    f,
                )

            # 2) Equilibration: 50 ps burn-in, not saved
            sim.context.setVelocitiesToTemperature(TEMPERATURE, seed)
            sim.step(BURN_IN_STEPS)

            # 3) Production: 1000 saved frames, 10 steps apart
            sim.reporters += [
                app.DCDReporter(str(out / "traj.dcd"), SAVE_STEPS),
                app.StateDataReporter(
                    str(out / "log.csv"), SAVE_STEPS * 100, step=True, time=True,
                    potentialEnergy=True, temperature=True, speed=True,
                ),
            ]
            sim.step(N_FRAMES * SAVE_STEPS)
            del sim
        except Exception as e:
            failures.append((str(pdb_path), repr(e)))

    return gpu, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--jobs-per-gpu", type=int, default=JOBS_PER_GPU)
    args = parser.parse_args()

    pdbs = sorted(glob.glob(f"robin_samples/{PEPTIDE}/{PEPTIDE}_*.pdb"))[args.start : args.start + args.n_samples]
    samples = [(int(Path(p).stem.split("_")[-1]), p) for p in pdbs]
    print(f"{len(samples)} samples, {len(GPUS)} GPUs x {args.jobs_per_gpu} jobs/GPU")

    n_workers = len(GPUS) * args.jobs_per_gpu
    chunks = [[] for _ in range(n_workers)]
    for i, sample in enumerate(samples):
        chunks[i % n_workers].append(sample)
    work = [(GPUS[i % len(GPUS)], chunk) for i, chunk in enumerate(chunks) if chunk]

    ctx = mp.get_context("spawn")
    t0 = time.time()
    failed = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(run_chunk, gpu, chunk): (gpu, chunk) for gpu, chunk in work}
        with tqdm(total=len(samples), desc="MD runs", unit="sample") as bar:
            for f in as_completed(futures):
                gpu, chunk = futures[f]
                try:
                    _, failures = f.result()
                except Exception as e:
                    failures = [(str(p), repr(e)) for _, p in chunk]
                for pdb_path, err in failures:
                    failed.append((pdb_path, err))
                    tqdm.write(f"FAILED: {pdb_path}: {err}")
                done += len(chunk)
                bar.update(len(chunk))
                bar.set_postfix(failed=len(failed))

    print(f"finished in {time.time() - t0:.1f}s, {len(failed)} failures")
    for pdb_path, err in failed:
        print(f"  {pdb_path}: {err}")


if __name__ == "__main__":
    main()
