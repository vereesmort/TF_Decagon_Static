"""
train_simple_pykeen_disk_wandb.py
---------------------------------
Disk-optimised SimplE training (local checkpoints + Drive sync) with optional
Weights & Biases logging. Same training logic as train_simplE_pykeen_disk.py.

Weights & Biases: set --wandb_project (and optionally --wandb_entity,
--wandb_run_name). Use --no_wandb to disable. Requires `pip install wandb` and
`wandb login` (or WANDB_API_KEY).

KEY DIFFERENCE vs train_simplE_pykeen.py
-----------------------------------------
All checkpoint I/O goes to a fast local directory (--local_dir, e.g.
/content/ckpt) instead of Google Drive. This eliminates the ~5 min
per-epoch Drive write overhead that was causing 2.5 h for 4 epochs.

Outputs use fixed filenames (checkpoint_best.pt, training_losses.json).
Use a unique --out_dir / --local_dir per run, or pass --no_clobber on
fresh runs to abort if those paths already contain checkpoints.

Drive sync strategy:
  1. PyKEEN writes checkpoint_epoch.pt (and related state) under --local_dir;
     EarlyStopper writes checkpoint_best.pt. No custom checkpoint_last.
  2. Every --sync_every epochs, a background thread runs _sync_checkpoints
     (local checkpoint dir → Drive). Training thread only submits the job.
  3. A final synchronous _sync_checkpoints runs after training completes.

  Note: async full-dir copy can theoretically overlap PyKEEN writes to the
  same filenames; use a large --sync_every if you see rare corrupt copies.

Checkpoint files produced
-------------------------
  local_dir/checkpoints/checkpoint_epoch.pt  — PyKEEN (resume here)
  local_dir/checkpoints/checkpoint_best.pt   — best MRR (EarlyStopper)
  out_dir/checkpoints/*                      — mirrored periodically + at end

Usage (Colab):
    %cd /content/drive/MyDrive/TF_Decagon_Static/training

    # Fresh run (add --wandb_project <name> to log to Weights & Biases)
    !python train_simple_pykeen_disk_wandb.py \\
        --dataset_dir /content/drive/MyDrive/TF_Decagon_Static/data/pykeen/selfloops \\
        --out_dir     /content/drive/MyDrive/TF_Decagon_Static/pykeen_results/simplE_selfloops_fp \\
        --local_dir   /content/ckpt_simplE_fp \\
        --pretrained_entities /content/drive/MyDrive/TF_Decagon_Static/data/embeddings/fp_only/256.npy \\
        --embedding_dim 256 \\
        --wandb_project tf-decagon

    # Colab / Jupyter: avoid tqdm flooding the cell output
    !python train_simple_pykeen_disk_wandb.py ... --no_tqdm

    # Resume (local checkpoint_epoch.pt or copy from Drive)
    !python train_simple_pykeen_disk_wandb.py \\
        --dataset_dir /content/drive/MyDrive/TF_Decagon_Static/data/pykeen/selfloops \\
        --out_dir     /content/drive/MyDrive/TF_Decagon_Static/pykeen_results/simplE_selfloops_fp \\
        --local_dir   /content/ckpt_simplE_fp \\
        --pretrained_entities /content/drive/MyDrive/TF_Decagon_Static/data/embeddings/fp_only/256.npy \\
        --embedding_dim 256 \\
        --wandb_project tf-decagon \\
        --resume      /path/to/checkpoints/checkpoint_epoch.pt
"""

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pykeen.models import SimplE
from pykeen.losses import CrossEntropyLoss
from pykeen.regularizers import LpRegularizer
from pykeen.training import LCWATrainingLoop
from pykeen.training.callbacks import TrainingCallback
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------
class _ReduceLROnPlateauCallback(TrainingCallback):
    """Steps ReduceLROnPlateau using validation MRR from EarlyStopper."""

    def __init__(self, stopper: EarlyStopper, mode: str, factor: float,
                 patience: int, threshold: float):
        super().__init__()
        self._stopper = stopper
        self._sched_kwargs = dict(mode=mode, factor=factor,
                                  patience=patience, threshold=threshold)
        self._scheduler: "ReduceLROnPlateau | None" = None
        self._n_results_seen = 0

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        if self._scheduler is None:
            self._scheduler = ReduceLROnPlateau(
                self.optimizer, **self._sched_kwargs)
        results = getattr(self._stopper, "results", [])
        if len(results) > self._n_results_seen:
            self._scheduler.step(results[-1])
            self._n_results_seen = len(results)


class _DriveSyncCallback(TrainingCallback):
    """Queue full local checkpoint dir → Drive (_sync_checkpoints) on a worker.

    Mirrors PyKEEN checkpoint_epoch.pt, EarlyStopper checkpoint_best.pt, etc.
    Training thread only submits; see module docstring re: rare live-file overlap.
    """

    def __init__(
        self,
        local_ckpt_dir: str,
        drive_ckpt_dir: str,
        executor: ThreadPoolExecutor,
        sync_every: int = 5,
    ):
        super().__init__()
        self._local = local_ckpt_dir
        self._drive = drive_ckpt_dir
        self._executor = executor
        self._every = sync_every

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        if epoch % self._every != 0:
            return
        loc, drv = self._local, self._drive
        ep = int(epoch)

        def _job() -> None:
            _sync_checkpoints(loc, drv, silent=False)
            print(f"  Drive sync finished (queued at epoch {ep})")

        self._executor.submit(_job)
        print(f"  Drive sync queued (epoch {ep}): {loc} → {drv}")


class _WandbLoggingCallback(TrainingCallback):
    """Logs training loss, LR, validation MRR and Hits@k via a wandb run object.

    Logged every epoch:
      train/loss  — LCWA cross-entropy loss
      train/lr    — learning rate (reflects ReduceLROnPlateau steps)

    Logged every es_frequency epochs (when EarlyStopper runs evaluation):
      val/mrr        — Mean Reciprocal Rank on valid_tf
      val/hits_at_1  — Hits@1
      val/hits_at_3  — Hits@3
      val/hits_at_10 — Hits@10
    """

    def __init__(self, stopper: EarlyStopper, run):
        super().__init__()
        self._stopper = stopper
        self._run = run
        self._n_results_seen = 0

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        lr = float(self.optimizer.param_groups[0]["lr"])
        payload = {
            "epoch":      epoch,
            "train/loss": float(epoch_loss),
            "train/lr":   lr,
        }

        results = getattr(self._stopper, "results", [])
        if len(results) > self._n_results_seen:
            self._n_results_seen = len(results)
            payload["val/mrr"] = float(results[-1])

            # Hits@k — stopper.metric_results available in pykeen >= 1.9
            metric_results = getattr(self._stopper, "metric_results", [])
            if metric_results:
                mr = metric_results[-1]
                for k in (1, 3, 10):
                    try:
                        payload[f"val/hits_at_{k}"] = float(
                            mr.get_metric(f"hits_at_{k}")
                        )
                    except Exception:
                        pass

        self._run.log(payload)   # use run object, not module-level wandb.log


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _sync_checkpoints(src: str, dst: str, silent: bool = False) -> None:
    """Copy every file in src/ to dst/."""
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(src):
        shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
    if not silent:
        files = os.listdir(src)
        print(f"  Synced {len(files)} checkpoint(s): {src} → {dst}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset_dir", required=True,
                    help="Output of make_pykeen_datasets.py")
parser.add_argument("--out_dir",     required=True,
                    help="Final results directory (Google Drive). "
                         "Checkpoints synced here periodically.")
parser.add_argument("--local_dir",   required=True,
                    help="Fast local directory for checkpoints during training "
                         "(e.g. /content/ckpt_simplE_fp). "
                         "Must survive the training run — use /content/ on Colab.")
parser.add_argument("--sync_every",  type=int, default=5,
                    help="Sync local checkpoints to --out_dir every N epochs "
                         "(default 5). Lower = safer but more Drive writes.")
parser.add_argument("--seed",        type=int, default=42)

# Hyperparameters — defaults match best_traces.csv (selfloops/simple row)
parser.add_argument("--embedding_dim",    type=int,   default=256)
parser.add_argument("--lr",               type=float, default=0.0107609491076982)
parser.add_argument("--batch_size",       type=int,   default=256)
parser.add_argument("--max_epochs",       type=int,   default=500)
parser.add_argument("--entity_dropout",   type=float, default=0.0679910727776587)
parser.add_argument("--relation_dropout", type=float, default=0.1253491407260298)
parser.add_argument("--entity_reg_weight",   type=float, default=9.084280252352796e-05)
parser.add_argument("--relation_reg_weight", type=float, default=2.517173786064683e-07)
parser.add_argument("--lr_patience",     type=int,   default=6)
parser.add_argument("--lr_factor",       type=float, default=0.5)
parser.add_argument("--es_patience",     type=int,   default=10,
                    help="Early stopping patience (in eval checks)")
parser.add_argument("--es_frequency",    type=int,   default=5,
                    help="Evaluate every N epochs")
parser.add_argument("--es_min_epochs",   type=int,   default=50,
                    help="Don't allow early stop before this epoch")
parser.add_argument("--checkpoint_freq", type=int,   default=5,
                    help="Save checkpoint_epoch.pt every N *minutes* (PyKEEN unit).")
parser.add_argument("--resume",          type=str,   default=None,
                    help="Path to checkpoint to resume from. "
                         "Prefer checkpoint_epoch.pt for PyKEEN-native continuation; "
                         "checkpoint_best.pt loads weights (+ optimizer if compatible).")
parser.add_argument("--pretrained_entities", type=str, default=None,
                    help="Path to .npy file of shape (n_entities, embedding_dim) "
                         "for pretrained drug embedding initialisation. "
                         "Produced by build_pretrained_init.py or "
                         "process_non_naive_improved.py.")
parser.add_argument("--ses", nargs="+", default=None,
                    help="Optional subset of SE relation names to train on "
                         "(e.g. --ses C0034065 C0020473). "
                         "Full vocabulary is kept for assessment.")
parser.add_argument("--num_ses", type=int, default=None,
                    help="Train on the first N side effects by relation ID "
                         "(e.g. --num_ses 50 for a fast test run). "
                         "Ignored if --ses is also set.")
parser.add_argument("--wandb_project", type=str, default=None,
                    help="Weights & Biases project; enables logging when set.")
parser.add_argument("--wandb_entity", type=str, default=None,
                    help="Optional W&B entity (team or username).")
parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name; default: <out_dir_basename>_YYYYMMDD-HHMMSS.")
parser.add_argument("--no_wandb", action="store_true",
                    help="Disable W&B even if --wandb_project is set.")
parser.add_argument(
    "--no_tqdm",
    action="store_true",
    help="Disable PyKEEN tqdm bars (recommended in Colab/Jupyter to avoid "
         "one-line-per-update log spam and browser lag).",
)
parser.add_argument(
    "--no_clobber",
    action="store_true",
    help="Abort before any setup if outputs already exist under --out_dir or "
         "--local_dir (fresh runs only; omit when using --resume). "
         "Does not run after training.",
)
args = parser.parse_args()

checkpoint_dir = os.path.join(args.local_dir, "checkpoints")
drive_checkpoint_dir = os.path.join(args.out_dir, "checkpoints")

if args.no_clobber and not args.resume:

    def _dir_has_pt(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        return any(name.endswith(".pt") for name in os.listdir(d))

    losses_path = os.path.join(args.out_dir, "training_losses.json")
    if os.path.isfile(losses_path):
        raise SystemExit(
            f"--no_clobber: refusing to overwrite completed run artifact:\n  {losses_path}\n"
            "Use a new --out_dir (e.g. add a timestamp) or omit --no_clobber."
        )
    if _dir_has_pt(drive_checkpoint_dir):
        raise SystemExit(
            f"--no_clobber: --out_dir already has checkpoints:\n  {drive_checkpoint_dir}\n"
            "Use a new --out_dir or use --resume and omit --no_clobber."
        )
    if _dir_has_pt(checkpoint_dir):
        raise SystemExit(
            f"--no_clobber: --local_dir already has checkpoints:\n  {checkpoint_dir}\n"
            "Use a new --local_dir or use --resume and omit --no_clobber."
        )

# ---------------------------------------------------------------------------
# Seeding + device
# ---------------------------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  seed: {args.seed}")

# ---------------------------------------------------------------------------
# Checkpoint directories
# ---------------------------------------------------------------------------
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(drive_checkpoint_dir, exist_ok=True)
print(f"  Checkpoints (local):  {checkpoint_dir}")
print(f"  Checkpoints (Drive):  {drive_checkpoint_dir}")
print(f"  Drive sync every:     {args.sync_every} epochs")

drive_sync_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="drive_sync")

# ---------------------------------------------------------------------------
# 1. Load TriplesFactory objects
# ---------------------------------------------------------------------------
print("Loading datasets...")
train_tf = torch.load(os.path.join(args.dataset_dir, "train_tf.pt"),
                      weights_only=False)
valid_tf = torch.load(os.path.join(args.dataset_dir, "valid_tf.pt"),
                      weights_only=False)
test_tf  = torch.load(os.path.join(args.dataset_dir, "test_tf.pt"),
                      weights_only=False)
print(f"  train: {train_tf.num_triples:,}  "
      f"valid: {valid_tf.num_triples:,}  "
      f"test:  {test_tf.num_triples:,}")
print(f"  entities: {train_tf.num_entities:,}  "
      f"relations: {train_tf.num_relations:,}")

# ---------------------------------------------------------------------------
# 1b. Optional SE subset filter
#     --ses    : explicit list of SE relation names
#     --num_ses: shortcut — first N PSE relations by sorted integer ID
#     Both preserve full vocab so assessment against all 963 SEs still works.
# ---------------------------------------------------------------------------
if args.ses is not None or args.num_ses is not None:
    from pykeen.triples import CoreTriplesFactory
    with open(os.path.join(args.dataset_dir, "relation_to_id.json")) as _f:
        _rel2id = json.load(_f)

    if args.ses is not None:
        unknown = [s for s in args.ses if s not in _rel2id]
        if unknown:
            raise ValueError(f"Unknown SE names (not in relation_to_id.json): {unknown}")
        ses_ids = torch.tensor([_rel2id[s] for s in args.ses], dtype=torch.long)
        _subset_label = f"{len(args.ses)} named SEs"
    else:
        # Pick first N PSE relation IDs that appear in training triples
        # (sorted by integer ID so selection is deterministic).
        _id2rel = {v: k for k, v in _rel2id.items()}
        pse_ids_in_train = torch.unique(train_tf.mapped_triples[:, 1]).tolist()
        pse_ids_sorted   = sorted(int(i) for i in pse_ids_in_train)
        chosen_ids       = pse_ids_sorted[:args.num_ses]
        ses_ids          = torch.tensor(chosen_ids, dtype=torch.long)
        chosen_names     = [_id2rel[i] for i in chosen_ids]
        _subset_label    = f"first {len(chosen_ids)} SEs by relation ID"
        print(f"  --num_ses {args.num_ses}: IDs {chosen_ids[0]}–{chosen_ids[-1]} "
              f"({chosen_names[0]} … {chosen_names[-1]})")

    def _filter_tf(tf):
        mask = torch.isin(tf.mapped_triples[:, 1], ses_ids)
        return CoreTriplesFactory(
            mapped_triples=tf.mapped_triples[mask],
            num_entities=tf.num_entities,
            num_relations=tf.num_relations,
        )

    train_tf = _filter_tf(train_tf)
    valid_tf = _filter_tf(valid_tf)
    test_tf  = _filter_tf(test_tf)
    print(f"  SE subset ({_subset_label}) → "
          f"train: {train_tf.num_triples:,}  "
          f"valid: {valid_tf.num_triples:,}  "
          f"test:  {test_tf.num_triples:,}")

# ---------------------------------------------------------------------------
# 2. Regularizers
# ---------------------------------------------------------------------------
entity_regularizer = LpRegularizer(
    weight=args.entity_reg_weight, p=2, normalize=True)
relation_regularizer = LpRegularizer(
    weight=args.relation_reg_weight, p=2, normalize=True)

# ---------------------------------------------------------------------------
# 3. Entity initialiser
# ---------------------------------------------------------------------------
if args.pretrained_entities:
    from pykeen.nn.init import PretrainedInitializer
    pretrained_np = np.load(args.pretrained_entities)
    if pretrained_np.shape != (train_tf.num_entities, args.embedding_dim):
        raise ValueError(
            f"Pretrained matrix shape {pretrained_np.shape} does not match "
            f"(n_entities={train_tf.num_entities}, dim={args.embedding_dim}). "
            f"Regenerate with build_pretrained_init.py --dims {args.embedding_dim} "
            f"--vocab_dir <path/to/pykeen/dataset>."
        )
    entity_init = PretrainedInitializer(
        tensor=torch.from_numpy(pretrained_np).float())
    print(f"  Pretrained entity init: {args.pretrained_entities}  "
          f"shape={pretrained_np.shape}  "
          f"non-zero rows: {(pretrained_np.any(axis=1)).sum()}")
else:
    entity_init = "xavier_normal_"
    print("  Entity init: xavier_normal_")

# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
model = SimplE(
    triples_factory=train_tf,
    embedding_dim=args.embedding_dim,
    entity_initializer=entity_init,
    relation_initializer="xavier_normal_",
    regularizer=entity_regularizer,   # applied to entity representations
    regularizer_kwargs=None,
    loss=CrossEntropyLoss(),
).to(device)

# Apply dropout to entity and relation representations
for rep in model.entity_representations:
    rep.dropout = torch.nn.Dropout(p=args.entity_dropout)
for rep in model.relation_representations:
    rep.dropout = torch.nn.Dropout(p=args.relation_dropout)

# Apply relation_regularizer explicitly — SimplE's constructor only accepts
# one regularizer (applied to entities). The relation regularizer must be
# attached directly to relation representations, matching LibKGE's
# simple.relation_embedder.regularize_weight: 2.517e-07
for rep in model.relation_representations:
    rep.regularizer = relation_regularizer

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
training_loop = LCWATrainingLoop(
    model=model,
    triples_factory=train_tf,
    optimizer="Adam",
    optimizer_kwargs={"lr": args.lr},
)

# ---------------------------------------------------------------------------
# 5b. Resume
# ---------------------------------------------------------------------------
if args.resume:
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    if os.path.basename(args.resume) == "checkpoint_epoch.pt":
        # PyKEEN format — copy to local checkpoint_dir, PyKEEN auto-resumes
        target = os.path.join(checkpoint_dir, "checkpoint_epoch.pt")
        if os.path.abspath(args.resume) != os.path.abspath(target):
            shutil.copy2(args.resume, target)
            print(f"  Copied checkpoint_epoch.pt → {target}")
        print(f"  Resumed (PyKEEN ckpt): epoch/optimizer restored by PyKEEN")

    else:
        raw = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            model.load_state_dict(raw["model_state_dict"], strict=False)
            try:
                training_loop.optimizer.load_state_dict(
                    raw["optimizer_state_dict"])
                print(f"  Resumed (custom ckpt): epoch={raw.get('epoch','?')}  "
                      f"loss={raw.get('loss', float('nan')):.4f}")
            except Exception as e:
                print(f"  Loaded model weights only (optimizer mismatch: {e})")
            print(f"  Note: epoch counter resets to 1. "
                  f"Use checkpoint_epoch.pt for true continuation.")
        else:
            missing, unexpected = model.load_state_dict(raw, strict=False)
            print(f"  Loaded plain state dict from {args.resume}")
            if unexpected:
                print(f"  Ignored keys: {unexpected[:5]}")

# ---------------------------------------------------------------------------
# 6. Evaluator
# ---------------------------------------------------------------------------
evaluator = RankBasedEvaluator(filtered=True)

# ---------------------------------------------------------------------------
# 7. Early stopping — saves checkpoint_best.pt to local checkpoint_dir
# ---------------------------------------------------------------------------
stopper = EarlyStopper(
    model=model,
    evaluator=evaluator,
    training_triples_factory=train_tf,
    evaluation_triples_factory=valid_tf,
    frequency=args.es_frequency,
    patience=args.es_patience,
    metric="mean_reciprocal_rank",
    larger_is_better=True,
    best_model_path=Path(checkpoint_dir) / "checkpoint_best.pt",
    clean_up_checkpoint=False,
)

# ---------------------------------------------------------------------------
# 8. Callbacks
# ---------------------------------------------------------------------------
lr_callback = _ReduceLROnPlateauCallback(
    stopper=stopper,
    mode="max",
    factor=args.lr_factor,
    patience=args.lr_patience,
    threshold=1e-4,
)
sync_callback = _DriveSyncCallback(
    local_ckpt_dir=checkpoint_dir,
    drive_ckpt_dir=drive_checkpoint_dir,
    executor=drive_sync_executor,
    sync_every=args.sync_every,
)

_use_wandb = bool(args.wandb_project) and not args.no_wandb

# ---------------------------------------------------------------------------
# 9. Initialise W&B run (before appending _WandbLoggingCallback to the list)
# ---------------------------------------------------------------------------
_wb_run = None
if _use_wandb:
    try:
        import wandb
    except ImportError as err:
        raise ImportError(
            "W&B requested (--wandb_project) but the wandb package is not "
            "installed. Install with: pip install wandb"
        ) from err

    _wb_run_name = args.wandb_run_name
    if not _wb_run_name:
        _wb_run_name = (
            f"{Path(args.out_dir).name}_"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    _param_count = sum(p.numel() for p in model.parameters())

    # Store the run object — use run.log() everywhere instead of wandb.log()
    # to avoid silent failures in Colab / Jupyter notebook environments.
    _wb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=_wb_run_name,
        config={
            "seed": args.seed,
            "embedding_dim": args.embedding_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "entity_dropout": args.entity_dropout,
            "relation_dropout": args.relation_dropout,
            "entity_reg_weight": args.entity_reg_weight,
            "relation_reg_weight": args.relation_reg_weight,
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "es_patience": args.es_patience,
            "es_frequency": args.es_frequency,
            "es_min_epochs": args.es_min_epochs,
            "sync_every": args.sync_every,
            "checkpoint_freq": args.checkpoint_freq,
            "use_tqdm": not args.no_tqdm,
            "train_triples": train_tf.num_triples,
            "valid_triples": valid_tf.num_triples,
            "test_triples": test_tf.num_triples,
            "num_entities": train_tf.num_entities,
            "num_relations": train_tf.num_relations,
            "param_count": _param_count,
            "device": str(device),
            "dataset_dir": args.dataset_dir,
            "out_dir": args.out_dir,
            "local_dir": args.local_dir,
            "resume": args.resume,
            "pretrained_entities": args.pretrained_entities,
            "ses": args.ses,
            "num_ses": args.num_ses,
        },
    )

    # define_metric sets epoch as the x-axis for all train/* and val/* charts.
    _wb_run.define_metric("epoch")
    _wb_run.define_metric("train/*", step_metric="epoch")
    _wb_run.define_metric("val/*",   step_metric="epoch")
    print(f"  W&B: project={args.wandb_project!r}  run={_wb_run_name!r}")

_train_callbacks = [lr_callback, sync_callback]
if _use_wandb:
    _train_callbacks.append(_WandbLoggingCallback(stopper=stopper, run=_wb_run))

# ---------------------------------------------------------------------------
# 10. Train
# ---------------------------------------------------------------------------

print(f"\nStarting training (max {args.max_epochs} epochs)...")
print(f"  embedding_dim={args.embedding_dim}  lr={args.lr}  "
      f"batch_size={args.batch_size}")
print(f"  entity_dropout={args.entity_dropout}  "
      f"relation_dropout={args.relation_dropout}")
print(f"  Early stopping: patience={args.es_patience} checks × "
      f"{args.es_frequency} epochs = {args.es_patience * args.es_frequency} "
      f"epochs max wait")

losses = None
try:
    losses = training_loop.train(
        triples_factory=train_tf,
        num_epochs=args.max_epochs,
        batch_size=args.batch_size,
        stopper=stopper,
        callbacks=_train_callbacks,
        checkpoint_directory=checkpoint_dir,
        checkpoint_name="checkpoint_epoch.pt",
        checkpoint_frequency=args.checkpoint_freq,
        checkpoint_on_failure=True,
        use_tqdm=not args.no_tqdm,
    )
finally:
    print("Draining background Drive sync jobs...")
    drive_sync_executor.shutdown(wait=True)

try:
    if losses is not None:
        # -------------------------------------------------------------------
        # 11. Final sync + save results (live dir — catches checkpoints after
        #     the last async job was queued)
        # -------------------------------------------------------------------
        print("\nFinal sync: local → Drive...")
        _sync_checkpoints(checkpoint_dir, drive_checkpoint_dir, silent=False)

        results_path = os.path.join(args.out_dir, "training_losses.json")
        with open(results_path, "w") as f:
            json.dump(losses, f)
        print(f"Training losses saved to {results_path}")

        if _use_wandb:
            import wandb

            art = wandb.Artifact("training_losses", type="results")
            art.add_file(results_path)
            _wb_run.log_artifact(art)

        print(f"Best checkpoint (Drive): {drive_checkpoint_dir}/checkpoint_best.pt")
        print(f"Run assessment with:")
        print(f"  python assessment/assessment_pykeen.py \\")
        print(f"      --checkpoint {drive_checkpoint_dir}/checkpoint_best.pt \\")
        print(f"      --dataset_dir {args.dataset_dir} \\")
        print(f"      --out_dir {args.out_dir}/assessment")
finally:
    if _use_wandb:
        import wandb

        _wb_run.finish()
