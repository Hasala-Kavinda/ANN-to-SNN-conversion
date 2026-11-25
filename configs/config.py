"""
LOCAL DEV:
  Uncomment the "Local dev overrides" block below before running on
  the laptop. Re-comment it before pushing to the server.

SERVER :
  Default values below are tuned for the server.
"""

# Environment
SCENARIO       = "basic"         # VizDoom scenario name (matches scenarios/*.cfg)
FRAME_SKIP     = 4               # Repeat each action for this many frames
FRAME_STACK    = 4               # Number of stacked grayscale frames fed to CNN
IMG_H          = 84              # Resize height (pixels)
IMG_W          = 84              # Resize width  (pixels)
WINDOW_VISIBLE = False           # Set True only for local interactive debugging
N_ACTIONS      = 3               # move-left, move-right, attack

# DQN Training (server defaults)
GAMMA          = 0.99            # Discount factor
LR             = 1e-4            # Adam learning rate
BATCH_SIZE     = 256             # Minibatch size for gradient updates
REPLAY_SIZE    = 200_000         # Maximum replay buffer capacity
WARMUP_STEPS   = 10_000          # Random steps before learning begins
TARGET_UPDATE  = 1_000           # Hard target-network update interval (steps)
EPS_START      = 1.0             # Initial epsilon for ε-greedy exploration
EPS_END        = 0.05            # Final epsilon
EPS_DECAY      = 100_000         # Steps over which epsilon decays linearly
MAX_STEPS      = 500_000         # Total environment steps for full training
GRAD_CLIP      = 10.0            # Max gradient norm (0 = disabled)
REWARD_SCALE   = 0.01            # Scale rewards inside DQN TD target for Huber loss
SEED           = 42

# Local dev overrides (RTX 3050, 4 GB VRAM)
# Comment out this block before pushing to the server
FRAME_STACK  = 1
BATCH_SIZE   = 32
REPLAY_SIZE  = 50_000
WARMUP_STEPS = 2_000
EPS_DECAY    = 15_000
MAX_STEPS    = 30_000
EVAL_EVERY   = 5_000

# hSNN Conversion
CALIBRATION_EPISODES = 500       # Episodes of gameplay used to collect activations
PERCENTILE           = 99.9      # Activation percentile used as IF threshold
T_VALUES             = [8, 16, 32, 64, 128]  # Timestep values for T-sweep

# Checkpointing & Logging
LOGGER          = "tensorboard"  # "tensorboard" or "wandb"
CHECKPOINT_DIR  = "checkpoints"
VIDEO_DIR       = "videos"
LOG_DIR         = "runs"
SAVE_EVERY      = 10_000         # Save checkpoint every N steps
EVAL_EVERY      = 25_000         # Run evaluation every N steps during training
EVAL_EPISODES   = 20             # Episodes per mid-training evaluation
