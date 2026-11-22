import torch
from types import SimpleNamespace
from huggingface_hub import hf_hub_download
from src.models.autoregressive_module import AutoregressiveLitModule
from src.data.single_peptide_datamodule import SinglePeptideDataModule
from src.data.preprocessing.encodings import get_encodings
from src.evaluation.plots.plot_energies import plot_energies

N_SAMPLES, BATCH = 10_000, 1000
STD = 0.35  # precomputed_std, configs/data/transferable/up_to_8aa.yaml

# 1. Load Robin. __init__ only reads datamodule.hparams.num_dimensions, so a stub is enough.
ckpt = hf_hub_download("danyalrehman17/robin-transferable", "robin.ckpt")
model = AutoregressiveLitModule.load_from_checkpoint(
    ckpt, datamodule=SimpleNamespace(hparams=SimpleNamespace(num_dimensions=3)),
    map_location="cuda", weights_only=False,
).eval()

# 2. Robin's per-residue atom order (alanine only)
ROBIN_ORDER = ["N", "H", "H1", "H2", "H3", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O", "OXT"]

def robin_index(top):
    idx = []
    for res in top.residues:
        by_name = {a.name: a.index for a in res.atoms}
        idx += [by_name[n] for n in ROBIN_ORDER if n in by_name]
    assert sorted(idx) == list(range(top.n_atoms))
    return torch.tensor(idx)

@torch.no_grad()
def sample(enc, n):
    n_atoms, out = len(enc["atom_type"]), []
    for _ in range(n // BATCH):
        e = {k: v[None].repeat(BATCH, 1).cuda() for k, v in enc.items()}
        tok = model.net.generate(batch_size=BATCH, num_atoms=n_atoms, encodings=e,
                                 temperature=model.hparams.temperature, use_cache=True)
        x = model.discretize.undiscretize_with_noise(tok).reshape(BATCH, n_atoms, 3)
        out.append(x.cpu())
    return torch.cat(out) * STD  # nm, Robin atom order

for seq in ["AAA", "AAAAAA"]:
    dm = SinglePeptideDataModule(
        repo_id="transferable-samplers/sequential-boltzmann-generators-data",
        data_dir="data", sequence=seq, temperature=310, num_dimensions=3, num_atoms=0,
    )
    dm.prepare_data(); dm.setup()
    pot = dm.setup_potential()

    # 3. Encodings in Robin order -> sample -> map back to PDB order
    idx = robin_index(dm.topology)
    enc = {k: (v if k == "seq_len" else v[idx]) for k, v in get_encodings(dm.topology).items()}
    x_robin = sample(enc, N_SAMPLES)
    x_pdb = torch.empty_like(x_robin); x_pdb[:, idx] = x_robin

    # 4. MD reference: strided 10k subsample of the test split, as in the paper's eval
    md_ref = dm.data_test.data
    md_ref = md_ref[:: len(md_ref) // N_SAMPLES][:N_SAMPLES]

    E_md, E_robin = pot.energy(md_ref).flatten(), pot.energy(x_pdb).flatten()  # units of kT
    plot_energies(lambda fig, name: fig.savefig(f"{seq}_{name}.png"),
                  E_md, E_robin, None, None, prefix="")
