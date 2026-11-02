# Third-party code

## PPO-PyTorch (independent PPO baseline)

- Upstream: https://github.com/nikhilbarhate99/PPO-PyTorch
- Vendored files: `PPO-PyTorch/PPO.py` and `PPO-PyTorch/LICENSE` (MIT)
- Pinned commit: `728cce83d7ab628fe2634eabcdf3239997eb81dd`
- Adapter: `esl/baselines/ppo_adapter.py` imports `PPO.py` only
- Optional dependency: `torch` (not required to recompute published tables)

M-FOS, MBOM, and the Lua/Torch opponent-modeling tree are **not** used in the
accepted twelve-game comparisons and are not shipped in this release.
