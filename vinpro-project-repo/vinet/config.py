"""
Configuration constants for ViNet.

Node types and branch types follow the Guyot pruning nomenclature
as described in Simonit (2014) and used in the 3D2cut dataset.
"""

# ── Node & Branch Taxonomies ──────────────────────────────────────────────────

NODE_TYPES = {
    "rootCrown": 0,
    "branchNode": 1,
    "growingTip": 2,
    "pruningCut": 3,
}

BRANCH_TYPES = {
    "mainTrunk": 0,
    "courson": 1,
    "cane": 2,
    "shoot": 3,
    "lateralShoot": 4,
}

NUM_NODE_TYPES = len(NODE_TYPES)
NUM_BRANCH_TYPES = len(BRANCH_TYPES)

# Total output channels: heatmaps + vector fields (x, y per branch type)
NUM_OUTPUT_CHANNELS = NUM_BRANCH_TYPES * NUM_NODE_TYPES + NUM_BRANCH_TYPES * 2  # 20 + 10 = 30

# ── Parent-Child Compatibility Rules ──────────────────────────────────────────

POSSIBLE_PARENTS = {
    "lateralShoot": ["lateralShoot", "shoot"],
    "shoot": ["shoot", "cane", "courson"],
    "cane": ["cane", "mainTrunk"],
    "courson": ["courson", "mainTrunk"],
    "mainTrunk": ["mainTrunk", "rootCrown"],
}

# ── Default Hyperparameters ───────────────────────────────────────────────────

DEFAULT_RESIZE = (256, 256)
DEFAULT_CROP_SIZE = 1024
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPOCHS = 300
DEFAULT_SIGMA_STAGE1 = 40   # Gaussian spread for first hourglass GT
DEFAULT_SIGMA_STAGE2 = 15   # Gaussian spread for second hourglass GT
DEFAULT_SIGMA_HEATMAP = 1.5 # Gaussian sigma for heatmap generation
DEFAULT_LIMB_WIDTH = 3      # Width of vector field around branches

# ── Model Architecture Defaults ───────────────────────────────────────────────

DEFAULT_FRONT_CHANNELS = 64
DEFAULT_HOURGLASS_CHANNELS = 256
DEFAULT_NUM_HOURGLASSES = 2
DEFAULT_HOURGLASS_DEPTH = 5

# ── Association Parameters ────────────────────────────────────────────────────

DEFAULT_ASSOCIATION_RADIUS = 0.2 * DEFAULT_RESIZE[0]  # r = 0.2 * W_h
DEFAULT_TAU_N = 0.5    # Threshold for low-confidence node filtering
DEFAULT_TAU_M = 0.97   # Threshold for local maxima detection
DEFAULT_ALPHA_LM = 0.1 # Factor for local maxima neighborhood distance
