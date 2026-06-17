"""Offline validation-loss eval: run the SE valid set through given model weights.
Mirrors train.py's construction; eval-only (no deepspeed).

Paths are taken from the environment (or CLI), so this is cluster-agnostic:
  ESPNET_ROOT  - espnet checkout root (added to sys.path). Default: inferred from this file.
  SE_DATA_DIR  - dir holding <name>/dataset.json manifests and a stats/ subdir.
                 Override per-run with --valid-spec / --stats-dir.
"""
import argparse, os, sys, yaml, torch
from pathlib import Path

# Repo root: env override, else infer (this file lives at egs2/opuslm_v2/speechlm1/se_caption/).
ESPNET_ROOT = os.environ.get(
    "ESPNET_ROOT", str(Path(__file__).resolve().parents[4])
)
sys.path.insert(0, ESPNET_ROOT)
from espnet2.speechlm.model import _all_job_types
from espnet2.speechlm.dataloader.iterator import DataIteratorFactory
from espnet2.speechlm.utils.data import to_device

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True, help="mp_rank_00_model_states.pt path")
p.add_argument("--tag", required=True)
p.add_argument("--max-batches", type=int, default=0)
p.add_argument("--train-config", required=True,
               help="train config used to build the model (e.g. "
                    "se_caption/conf/train_se_bagpiper_lora.yaml)")
p.add_argument("--valid-spec",
               default=os.environ.get("SE_VALID_SPEC", ""),
               help="iterator specifier 'task:name:/path/to/dataset.json'. "
                    "Default: $SE_VALID_SPEC.")
p.add_argument("--stats-dir",
               default=os.environ.get("SE_STATS_DIR", ""),
               help="length-stats dir produced by prepare_length_stats.py. "
                    "Default: $SE_STATS_DIR.")
a = p.parse_args()

if not a.valid_spec or not a.stats_dir:
    p.error("provide --valid-spec and --stats-dir (or set $SE_VALID_SPEC / $SE_STATS_DIR)")
cfg = yaml.safe_load(open(a.train_config))

job = _all_job_types[cfg["job_type"]](cfg, is_train=True)
pre = job.build_preprocessor()
model = job.build_model()

sd = torch.load(a.weights, map_location="cpu", weights_only=False)["module"]
inc = model.load_state_dict(sd, strict=False)
print(f"[{a.tag}] loaded {a.weights}: missing={len(inc.missing_keys)} "
      f"unexpected={len(inc.unexpected_keys)}", flush=True)
model = model.cuda().to(torch.bfloat16).eval()

fac = DataIteratorFactory(
    a.valid_spec, "",
    stats_dir=Path(a.stats_dir), collate_fn=pre.collate_fn,
    batchfy_method=cfg["data_loading"]["batchfy_method"],
    batch_size=cfg["data_loading"]["batch_size"],
    num_workers=0, rank=0, world_size=1, shuffle=False)

tot, n = 0.0, 0
with torch.no_grad():
    for i, batch in enumerate(fac.build_iter()):
        batch = to_device(batch, "cuda", dtype=torch.bfloat16)
        out = model(**batch)
        tot += float(out["loss"].float().cpu()); n += 1
        if i % 20 == 0:
            print(f"[{a.tag}] batch {i}: running mean {tot/n:.4f}", flush=True)
        if a.max_batches and n >= a.max_batches:
            break
print(f"[{a.tag}] FINAL valid loss over {n} batches: {tot/n:.4f}", flush=True)
