"""GRAIL Protocol Constants — V1 Verifiable Inference.

Immutable values that all network participants must agree on.
No os.getenv() overrides. Changes require coordinated deployment.
"""

# ────────────────  GRAIL PROOF VERSION  ────────────────

GRAIL_PROOF_VERSION = "v5"

# ────────────────  CRYPTOGRAPHIC CONSTANTS  ────────────────

# Mersenne prime for modular sketch arithmetic.
PRIME_Q = 2_147_483_647

# Number of random challenge positions per completion.
CHALLENGE_K = 32

# PRF domain labels for different randomness derivations.
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# Transformer layer index for hidden state extraction (-1 = last layer).
LAYER_INDEX = -1

# Batch size for proof computation (log-softmax / GRAIL commitments).
# Fixed: changing causes numerical divergence between miner and validator.
PROOF_BATCH_SIZE = 16

# Top-K activation selection for sketch computation.
PROOF_TOPK = 16

# Logarithmic bucketing: buckets per sign (16 total = 8 positive + 8 negative).
PROOF_NUM_BUCKETS = 8

# Bounded coefficient range for sketch robustness: r in [-127, 127].
PROOF_COEFF_RANGE = 127

# Sketch tolerance at position 0. Covers cross-GPU drift.
# Empirical max diff across ~300M positions = 3979. Base of 6000 gives ~50% headroom.
PROOF_SKETCH_TOLERANCE_BASE = 6000

# Sketch tolerance sqrt growth factor per position.
# tolerance(P) = base + growth * sqrt(P).
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0

# Attention implementation forced across all model loading paths.
ATTN_IMPLEMENTATION = "flash_attention_2"

# ────────────────  TIMING (CONSENSUS)  ────────────────

# Blocks per window. All roles use this to determine window boundaries.
WINDOW_LENGTH = 30

# Bittensor block time target average (seconds).
BLOCK_TIME_SECONDS = 12

# Typical variance in block production time (seconds).
BLOCK_TIME_VARIANCE = 3

# Network latency allowance for file uploads (seconds).
NETWORK_UPLOAD_LATENCY = 30

# Grace period = block variance + upload latency.
UPLOAD_GRACE_PERIOD = BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# Buffer for future drand beacon (seconds).
DRAND_FUTURE_BUFFER = 30

# ────────────────  ROLLOUT GENERATION  ────────────────

# Network-wide protocol cap on completion length.
MAX_NEW_TOKENS_PROTOCOL_CAP = 8192

# ────────────────  ECONOMIC / INCENTIVE  ────────────────

# Superlinear weighting exponent for sybil resistance.
# w_i proportional to s_i^p. Splitting into k identities yields k^(1-p) * s^p < s^p.
SUPERLINEAR_EXPONENT = 4.0

# Maximum unique rollouts per miner per window that count toward weight allocation.
UNIQUE_ROLLOUTS_CAP = 5000
UNIQUE_ROLLOUTS_CAP_ENABLED = True

# ────────────────  VALIDATION RULES  ────────────────

# Miner sampling parameters.
MINER_SAMPLING_ENABLED = True
MINER_SAMPLE_RATE = 0.25
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35

# Failure lookback for exclusion from sampling.
FAILURE_LOOKBACK_WINDOWS = 14

# File size bounds for valid rollout window files.
MIN_ROLLOUT_FILE_SIZE_BYTES = 200
MAX_ROLLOUT_FILE_SIZE_BYTES = 350 * 1024 * 1024  # 350 MB

# Maximum number of rollouts per submission file.
MAX_ROLLOUTS_PER_FILE = 6000

# Maximum token sequence length in a single rollout.
MAX_TOKENS_PER_ROLLOUT = MAX_NEW_TOKENS_PROTOCOL_CAP + 4096  # prompt + completion

# Soft check threshold for stochastic failures.
STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.51

# ────────────────  CONTINUOUS VALIDATION  ────────────────

# How often the validator polls S3 for new miner files (seconds).
POLL_INTERVAL_SECONDS = 10

# Fraction of a miner's rollouts to verify (the rest are extrapolated).
ROLLOUT_SAMPLE_RATE = 0.10

# Minimum rollouts to verify per miner regardless of sample rate.
ROLLOUT_SAMPLE_MIN = 16

# Verification batch size — rollouts are checked in batches of this size.
# After each batch, the failure rate is evaluated for early gating.
VERIFICATION_BATCH_SIZE = 16

# If more than this fraction of checked rollouts fail within a batch,
# the miner is gated immediately and remaining rollouts are skipped.
BATCH_FAILURE_THRESHOLD = 0.30

# ────────────────  WEIGHT SUBMISSION  ────────────────

WEIGHT_SUBMISSION_INTERVAL = 360  # Blocks between weight submissions

# ────────────────  DATASET  ────────────────

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DATASET_SPLIT = "train"

# ────────────────  STORAGE  ────────────────

CHECKPOINT_PREFIX = "grail/checkpoints/"
