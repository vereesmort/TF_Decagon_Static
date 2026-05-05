"""
train_simplE_pykeen.py
----------------------
Replaces:  kge start simplE_selfloops_best.yaml
Purpose:   Trains SimplE on the Selfloops graph using PyKEEN, reproducing
           the best configuration from Lloyd et al. (2024) best_traces.csv.

LIBKGE → PYKEEN TRANSLATION
-----------------------------
LibKGE concept          PyKEEN equivalent
-----------------------  --------------------------------------------------
model: simple            pykeen.models.SimplE
train.type: 1vsAll       LCWATrainingLoop  (Local Closed World Assumption)
train.loss: kl           CrossEntropyLoss  (PyKEEN's closest to KL 1vsAll)
train.optimizer: Adam    optimizer='Adam', optimizer_kwargs={'lr': 0.01076}
lookup_embedder.dim: 256 embedding_dim=256
xavier_normal_ init      entity_initializer='xavier_normal_'
entity dropout 0.068     entity_representations_kwargs dropout
relation dropout 0.125   relation_representations_kwargs dropout
lp regularize weighted   LpRegularizer(weight=..., p=2, normalize=True)
ReduceLROnPlateau        lr_scheduler='ReduceLROnPlateau'
early stopping           EarlyStopper(patience=10, frequency=5, ...)
checkpoint every 5 min   checkpoint_frequency=5  (unit is minutes, not epochs)

TRAINING TYPE NOTE
------------------
LibKGE's "1vsAll" = for each (h, r) pair, score against ALL entities as tails.
In PyKEEN this is LCWATrainingLoop (not SLCWA which does negative sampling).
LibKGE's "KvsAll" = same structure with label smoothing — also use LCWA.
LibKGE's "negative_sampling" = SLCWATrainingLoop.

LOSS NOTE
---------
LibKGE's "kl" loss with 1vsAll applies softmax over all entities then KL vs
one-hot label. PyKEEN's CrossEntropyLoss in LCWA mode applies log-softmax
over all entity scores — this is functionally equivalent (same gradient).

Usage:
    # Step 1: prepare dataset (once)
    python data/make_pykeen_datasets.py \\
        --edgelist data/graphs/selfloops/edgelist_selfloops.tsv \\
        --out_dir  data/pykeen/selfloops

    # Step 2: train
    python training/train_simplE_pykeen.py \\
        --dataset_dir data/pykeen/selfloops \\
        --out_dir     results/simplE_selfloops \\
        --seed        42

    # Resume from checkpoint
    python training/train_simplE_pykeen.py \\
        --dataset_dir data/pykeen/selfloops \\
        --out_dir     results/simplE_selfloops \\
        --resume      results/simplE_selfloops/checkpoints/checkpoint_best.pt
"""

import argparse
import json
import os
import random
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
    """
    Steps ReduceLROnPlateau with the validation metric recorded by the
    EarlyStopper after each evaluation window.

    PyKEEN's built-in LRSchedulerCallback calls scheduler.step(epoch=...).
    ReduceLROnPlateau requires step(metrics) instead, so it cannot be passed
    as lr_scheduler= to LCWATrainingLoop directly.
    """

    def __init__(self, stopper: EarlyStopper, mode: str, factor: float,
                 patience: int, threshold: float):
        super().__init__()   # sets self._training_loop = None (required by base class)
        self._stopper = stopper
        self._sched_kwargs = dict(mode=mode, factor=factor,
                                  patience=patience, threshold=threshold)
        self._scheduler: "ReduceLROnPlateau | None" = None
        self._n_results_seen = 0

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        if self._scheduler is None:
            # self.optimizer is a shortcut property on TrainingCallback
            # that returns self.training_loop.optimizer (set after registration)
            self._scheduler = ReduceLROnPlateau(
                self.optimizer, **self._sched_kwargs
            )
        # stopper.results grows by one after each validation run;
        # only step the scheduler when a fresh metric is available.
        results = getattr(self._stopper, "results", [])
        if len(results) > self._n_results_seen:
            self._scheduler.step(results[-1])
            self._n_results_seen = len(results)


class _LastCheckpointCallback(TrainingCallback):
    """
    Saves checkpoint_last.pt after every epoch, overwriting the previous one.

    Replicates LibKGE's per-epoch rolling checkpoint so a crash never loses
    more than one epoch of work.  This is separate from the milestone
    checkpoints (every N epochs) and the best-model checkpoint (EarlyStopper).
    """

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True,
                    help="Output of make_pykeen_datasets.py")
parser.add_argument("--out_dir",     required=True,
                    help="Where to save checkpoints and results")
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
                    help="Save checkpoint_epoch.pt every N *minutes* (PyKEEN unit). "
                         "Default 5 min ≈ every epoch at ~10 min/epoch.")
parser.add_argument("--resume",          type=str,   default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--pretrained_entities", type=str, default=None,
                    help="Path to .npy file of shape (n_entities, embedding_dim) "
                         "for non-naive pretrained drug embedding initialisation. "
                         "Produced by process_non_naive_improved.py. "
                         "When set, replaces xavier_normal_ for entity embeddings.")
parser.add_argument("--ses", nargs="+", default=None,
                    help="Optional subset of SE relation names to train on "
                         "(e.g. --ses C0034065 C0020473). "
                         "The full entity/relation vocabulary is kept so "
                         "embeddings are still indexable for all 963 SEs at "
                         "assessment time — only the training triples are filtered.")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  seed: {args.seed}")

checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load TriplesFactory objects saved by make_pykeen_datasets.py
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
#     Keeps the full vocabulary (all entity/relation IDs unchanged) so that
#     assessment against all 963 SEs still works after training.
#     Only the triples used for training/validation/testing are restricted.
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
# 2. Build regularizers
#    LibKGE lp regularize with weighted=True → LpRegularizer(normalize=True)
# ---------------------------------------------------------------------------
entity_regularizer = LpRegularizer(
    weight=args.entity_reg_weight,
    p=2,
    normalize=True,          # weighted=True in LibKGE
)
relation_regularizer = LpRegularizer(
    weight=args.relation_reg_weight,
    p=2,
    normalize=True,
)

# ---------------------------------------------------------------------------
# 3. Build entity initialiser
#    Selfloops:  xavier_normal_ (random, matches best_traces.csv)
#    Non-naive:  PretrainedInitializer from PCA/SVD compressed mono SE vectors
#                produced by process_non_naive_improved.py
# ---------------------------------------------------------------------------
if args.pretrained_entities:
    from pykeen.nn.init import PretrainedInitializer
    pretrained_np = np.load(args.pretrained_entities)   # (n_entities, dim)
    if pretrained_np.shape != (train_tf.num_entities, args.embedding_dim):
        raise ValueError(
            f"Pretrained matrix shape {pretrained_np.shape} does not match "
            f"(n_entities={train_tf.num_entities}, dim={args.embedding_dim}). "
            f"Regenerate with:\n"
            f"  python process_non-naive_improved.py \\\n"
            f"      --method pca --dims {args.embedding_dim} \\\n"
            f"      --vocab_dir <path/to/pykeen/non-naive>\n"
            f"where <path/to/pykeen/non-naive> contains entity_to_id.json "
            f"(produced by make_pykeen_datasets.py with --shared_vocab_dir)."
        )
    entity_init = PretrainedInitializer(
        tensor=torch.from_numpy(pretrained_np).float()
    )
    print(f"  Pretrained entity init: {args.pretrained_entities}  "
          f"shape={pretrained_np.shape}  "
          f"non-zero rows: {(pretrained_np.any(axis=1)).sum()}")
else:
    entity_init = "xavier_normal_"
    print("  Entity init: xavier_normal_ (selfloops mode)")

# ---------------------------------------------------------------------------
# 4. Build SimplE model
#    Dropout is applied to entity and relation representations post-init.
# ---------------------------------------------------------------------------
model = SimplE(
    triples_factory=train_tf,
    embedding_dim=args.embedding_dim,
    entity_initializer=entity_init,
    relation_initializer="xavier_normal_",
    regularizer=entity_regularizer,
    regularizer_kwargs=None,
    loss=CrossEntropyLoss(),
).to(device)

# Apply dropout to entity and relation representations
# PyKEEN SimplE has two entity embedding tables (h and t) and one relation table
for rep in model.entity_representations:
    rep.dropout = torch.nn.Dropout(p=args.entity_dropout)
for rep in model.relation_representations:
    rep.dropout = torch.nn.Dropout(p=args.relation_dropout)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# 5. Training loop: LCWA = 1vsAll
#    Loss is attached to the model (CrossEntropyLoss set above).
#    PyKEEN >= 1.8 removed the `loss` kwarg from TrainingLoop; it now lives
#    on the model itself.
# ---------------------------------------------------------------------------
training_loop = LCWATrainingLoop(
    model=model,
    triples_factory=train_tf,
    optimizer="Adam",
    optimizer_kwargs={"lr": args.lr},
)

# ---------------------------------------------------------------------------
# 5b. Resume from checkpoint (if --resume is set)
#
#  Two checkpoint formats exist in this repo:
#
#  A) checkpoint_epoch.pt  — PyKEEN's own internal format.
#     Contains full training state (epoch, model, optimizer, LR scheduler).
#     Pass resume_from_checkpoint=True to training_loop.train() and PyKEEN
#     restores everything and continues from the correct epoch.
#     Use this for true mid-run continuation.
#
#  B) checkpoint_last.pt / checkpoint_best.pt — our custom format.
#     Contains: {"epoch", "model_state_dict", "optimizer_state_dict", "loss"}.
#     We restore model + optimizer weights; epoch counter resets to 1.
#     Use this for fine-tuning or warm-starting from a known-good model.
# ---------------------------------------------------------------------------
if args.resume:
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    if os.path.basename(args.resume) == "checkpoint_epoch.pt":
        # Format A: PyKEEN's own checkpoint — MUST check filename first because
        # PyKEEN's format also contains a 'model_state_dict' key and would
        # otherwise be misdetected as Format B.
        # Copy to checkpoint_dir so PyKEEN auto-detects and resumes from it.
        target = os.path.join(checkpoint_dir, "checkpoint_epoch.pt")
        if os.path.abspath(args.resume) != os.path.abspath(target):
            import shutil
            shutil.copy2(args.resume, target)
            print(f"  Copied {args.resume} → {target}")
        print(f"  Resumed (PyKEEN ckpt): epoch and optimizer state will be "
              f"restored by PyKEEN → {args.resume}")

    else:
        raw = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(raw, dict) and "model_state_dict" in raw:
            # Format B: our custom checkpoint_last.pt / checkpoint_best.pt
            model.load_state_dict(raw["model_state_dict"], strict=False)
            try:
                training_loop.optimizer.load_state_dict(raw["optimizer_state_dict"])
                print(f"  Resumed (custom ckpt): epoch={raw.get('epoch','?')}  "
                      f"loss={raw.get('loss', float('nan')):.4f}  "
                      f"→ {args.resume}")
            except Exception as e:
                print(f"  Loaded model weights only (optimizer state mismatch: {e})")
            print(f"  Note: epoch counter resets to 1 (use checkpoint_epoch.pt "
                  f"for true continuation with correct epoch number).")

        else:
            # Unknown format: try loading as a plain state dict
            try:
                missing, unexpected = model.load_state_dict(raw, strict=False)
                print(f"  Loaded state dict from {args.resume}")
                if unexpected:
                    print(f"  Ignored keys: {unexpected[:5]}")
            except Exception as e:
                raise ValueError(
                    f"Cannot load checkpoint {args.resume}: {e}\n"
                    f"Expected either checkpoint_epoch.pt (PyKEEN format) or a "
                    f"dict with 'model_state_dict' key (custom format)."
                )

# ---------------------------------------------------------------------------
# 6. Evaluator: RankBasedEvaluator (reports MRR, Hits@k)
#    filter_with_test=True → filter known positives from rankings (same as LibKGE)
# ---------------------------------------------------------------------------
evaluator = RankBasedEvaluator(filtered=True)

# ---------------------------------------------------------------------------
# 7. Early stopping
#    patience=10 checks, frequency=5 epochs → stop after 50 epochs of no gain
#    min_epochs=50 → don't check before epoch 50 (matches LibKGE config)
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
    clean_up_checkpoint=False,   # keep checkpoint_best.pt after early stop
)

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

# ---------------------------------------------------------------------------
# 8. Train
# ---------------------------------------------------------------------------
print(f"\nStarting training (max {args.max_epochs} epochs)...")
print(f"  embedding_dim={args.embedding_dim}  lr={args.lr}  "
      f"batch_size={args.batch_size}")
print(f"  entity_dropout={args.entity_dropout}  "
      f"relation_dropout={args.relation_dropout}")
print(f"  Early stopping: patience={args.es_patience} checks × "
      f"{args.es_frequency} epochs = {args.es_patience * args.es_frequency} epochs max wait")

losses = training_loop.train(
    triples_factory=train_tf,
    num_epochs=args.max_epochs,
    batch_size=args.batch_size,
    stopper=stopper,
    callbacks=[lr_callback, last_ckpt_callback],
    checkpoint_directory=checkpoint_dir,
    checkpoint_name="checkpoint_epoch.pt",
    checkpoint_frequency=args.checkpoint_freq,
    checkpoint_on_failure=True,
    use_tqdm=True,
)

# ---------------------------------------------------------------------------
# 9. Save final results
# ---------------------------------------------------------------------------
results_path = os.path.join(args.out_dir, "training_losses.json")
with open(results_path, "w") as f:
    json.dump(losses, f)
print(f"\nTraining losses saved to {results_path}")
print(f"Best checkpoint: {checkpoint_dir}/checkpoint_best.pt")
print(f"Run assessment with:")
print(f"  python assessment/assessment_pykeen.py \\")
print(f"      --checkpoint {checkpoint_dir}/checkpoint_best.pt \\")
print(f"      --dataset_dir {args.dataset_dir} \\")
print(f"      --out_dir results/")
