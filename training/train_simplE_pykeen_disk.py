"""
train_simplE_pykeen_disk.py
---------------------------
Disk-optimised variant of train_simplE_pykeen.py for Google Colab.

KEY DIFFERENCE vs train_simplE_pykeen.py
-----------------------------------------
All checkpoint I/O goes to a fast local directory (--local_dir, e.g.
/content/ckpt) instead of Google Drive. This eliminates the ~5 min
per-epoch Drive write overhead that was causing 2.5 h for 4 epochs.

Drive sync strategy (three layers):
  1. Periodic sync every --sync_every epochs (default 5) via
     _DriveSyncCallback — protects against mid-run Colab disconnects.
  2. Final sync after training_loop.train() returns.
  3. checkpoint_best.pt is always synced immediately when EarlyStopper
     writes it (it lives in local_dir/checkpoints/ and is included in
     every periodic sync).

Usage (Colab):
    %cd /content/drive/MyDrive/TF_Decagon_Static/training

    # Fresh run
    !python train_simplE_pykeen_disk.py \\
        --dataset_dir /content/drive/MyDrive/TF_Decagon_Static/data/pykeen/selfloops \\
        --out_dir     /content/drive/MyDrive/TF_Decagon_Static/pykeen_results/simplE_selfloops_fp \\
        --local_dir   /content/ckpt_simplE_fp \\
        --pretrained_entities /content/drive/MyDrive/TF_Decagon_Static/data/embeddings/fp_only/256.npy \\
        --embedding_dim 256

    # Resume (use local checkpoint_epoch.pt if still in /content/, else Drive copy)
    !python train_simplE_pykeen_disk.py \\
        --dataset_dir /content/drive/MyDrive/TF_Decagon_Static/data/pykeen/selfloops \\
        --out_dir     /content/drive/MyDrive/TF_Decagon_Static/pykeen_results/simplE_selfloops_fp \\
        --local_dir   /content/ckpt_simplE_fp \\
        --pretrained_entities /content/drive/MyDrive/TF_Decagon_Static/data/embeddings/fp_only/256.npy \\
        --embedding_dim 256 \\
        --resume      /content/drive/MyDrive/TF_Decagon_Static/pykeen_results/simplE_selfloops_fp/checkpoints/checkpoint_epoch.pt

Checkpoint files produced
-------------------------
  local_dir/checkpoints/checkpoint_last.pt   — custom format, after every epoch
  local_dir/checkpoints/checkpoint_epoch.pt  — PyKEEN format, every N minutes + on crash
  local_dir/checkpoints/checkpoint_best.pt   — custom format, best MRR so far
  out_dir/checkpoints/*                      — synced from local_dir every --sync_every epochs
"""

import argparse
import json
import os
import random
import shutil
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


class _LastCheckpointCallback(TrainingCallback):
    """Saves checkpoint_last.pt to local disk after every epoch."""

    def __init__(self, path: Path):
        super().__init__()
        self._path = path

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": epoch_loss,
            },
            self._path,
        )


class _DriveSyncCallback(TrainingCallback):
    """
    Copies all checkpoints from local_dir → drive_dir every sync_every epochs.

    This is the crash-safety layer: if Colab disconnects between syncs you
    lose at most sync_every epochs of work. Keep sync_every low (5–10) if
    Drive speed is reasonable; raise it if Drive is the bottleneck.
    """

    def __init__(self, local_ckpt_dir: str, drive_ckpt_dir: str,
                 sync_every: int = 5):
        super().__init__()
        self._local  = local_ckpt_dir
        self._drive  = drive_ckpt_dir
        self._every  = sync_every

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        if epoch % self._every == 0:
            _sync_checkpoints(self._local, self._drive, silent=False)


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
                         "Use checkpoint_epoch.pt for true epoch continuation, "
                         "or checkpoint_last.pt / checkpoint_best.pt for "
                         "weight-only warm-start.")
parser.add_argument("--pretrained_entities", type=str, default=None,
                    help="Path to .npy file of shape (n_entities, embedding_dim) "
                         "for pretrained drug embedding initialisation. "
                         "Produced by build_pretrained_init.py or "
                         "process_non_naive_improved.py.")
parser.add_argument("--ses", nargs="+", default=None,
                    help="Optional subset of SE relation names to train on "
                         "(e.g. --ses C0034065 C0020473). "
                         "Full vocabulary is kept for assessment.")
args = parser.parse_args()

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
checkpoint_dir       = os.path.join(args.local_dir, "checkpoints")   # fast local
drive_checkpoint_dir = os.path.join(args.out_dir,   "checkpoints")   # Drive
os.makedirs(checkpoint_dir,       exist_ok=True)
os.makedirs(drive_checkpoint_dir, exist_ok=True)
print(f"  Checkpoints (local):  {checkpoint_dir}")
print(f"  Checkpoints (Drive):  {drive_checkpoint_dir}")
print(f"  Drive sync every:     {args.sync_every} epochs")

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
# ---------------------------------------------------------------------------
if args.ses is not None:
    from pykeen.triples import CoreTriplesFactory
    with open(os.path.join(args.dataset_dir, "relation_to_id.json")) as _f:
        _rel2id = json.load(_f)

    unknown = [s for s in args.ses if s not in _rel2id]
    if unknown:
        raise ValueError(f"Unknown SE names (not in relation_to_id.json): {unknown}")

    ses_ids = torch.tensor([_rel2id[s] for s in args.ses], dtype=torch.long)

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
    print(f"  SE subset ({len(args.ses)} types) → "
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
last_ckpt_callback = _LastCheckpointCallback(
    path=Path(checkpoint_dir) / "checkpoint_last.pt",
)
sync_callback = _DriveSyncCallback(
    local_ckpt_dir=checkpoint_dir,
    drive_ckpt_dir=drive_checkpoint_dir,
    sync_every=args.sync_every,
)

# ---------------------------------------------------------------------------
# 9. Train
# ---------------------------------------------------------------------------
print(f"\nStarting training (max {args.max_epochs} epochs)...")
print(f"  embedding_dim={args.embedding_dim}  lr={args.lr}  "
      f"batch_size={args.batch_size}")
print(f"  entity_dropout={args.entity_dropout}  "
      f"relation_dropout={args.relation_dropout}")
print(f"  Early stopping: patience={args.es_patience} checks × "
      f"{args.es_frequency} epochs = {args.es_patience * args.es_frequency} "
      f"epochs max wait")

losses = training_loop.train(
    triples_factory=train_tf,
    num_epochs=args.max_epochs,
    batch_size=args.batch_size,
    stopper=stopper,
    callbacks=[lr_callback, last_ckpt_callback, sync_callback],
    checkpoint_directory=checkpoint_dir,
    checkpoint_name="checkpoint_epoch.pt",
    checkpoint_frequency=args.checkpoint_freq,
    checkpoint_on_failure=True,
    use_tqdm=True,
)

# ---------------------------------------------------------------------------
# 10. Final sync + save results
# ---------------------------------------------------------------------------
print("\nFinal sync: local → Drive...")
_sync_checkpoints(checkpoint_dir, drive_checkpoint_dir, silent=False)

results_path = os.path.join(args.out_dir, "training_losses.json")
with open(results_path, "w") as f:
    json.dump(losses, f)
print(f"Training losses saved to {results_path}")
print(f"Best checkpoint (Drive): {drive_checkpoint_dir}/checkpoint_best.pt")
print(f"Run assessment with:")
print(f"  python assessment/assessment_pykeen.py \\")
print(f"      --checkpoint {drive_checkpoint_dir}/checkpoint_best.pt \\")
print(f"      --dataset_dir {args.dataset_dir} \\")
print(f"      --out_dir {args.out_dir}/assessment")
