# Main Performance Grid Smoke Report

## Baseline Readiness

- READY sources: third_party/PPO-PyTorch
- PARTIAL / unavailable: mfos: Official source is pinned, but full M-FOS meta-game training is not yet wrapped.; mbom: Official code is pinned but incompatible with the current lightweight matrix-game runtime.; simple_opponent_model: Official source is pinned but cannot run directly in this Python matrix-game stack.

## Smoke Onboarding

- Independent PPO: PASS (Official PPO.py is present and loaded directly by the adapter.)
- M-FOS: PASS (Official source is pinned, but full M-FOS meta-game training is not yet wrapped.)
- MBOM: PASS (Official code is pinned but incompatible with the current lightweight matrix-game runtime.)
- Simple Opponent Model: PASS (Official source is pinned but cannot run directly in this Python matrix-game stack.)

## Figure

- Output: `manuscript_bundle/main/figures/main_performance_grid.png`
- Layout: usable 3 x 3 task/regime grid with payoff bars and 95% CI error bars.
- ESL average minus external average across smoke cells: -0.1933

## Before Full Paper Runs

- Review the PARTIAL adapters and decide whether reduced implementations are acceptable.
- Wrap full official M-FOS and MBOM training stacks if they must be claimed as official baselines.
- Decide how to handle the Lua/Torch Simple Opponent Model source for camera-ready claims.
- Increase seeds and horizons only after this audit is accepted.
