# Third-party baseline sources (PRD §8A, §14A)

## PPO-PyTorch (independent PPO)

- **Upstream:** [github.com/nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
- **This repo:** git submodule at `third_party/PPO-PyTorch` (shallow clone supported).
- **Initialize:** from repo root, `git submodule update --init --depth 1 third_party/PPO-PyTorch`
- **Adapter:** `esl/baselines/ppo_adapter.py` imports `PPO.py` only (no `train.py` roboschool dependency).
- **Pinned commit:** `728cce83d7ab628fe2634eabcdf3239997eb81dd`
- **License:** MIT.

Optional Python deps for the adapter: see `requirements-ppo.txt` (`torch`).

## Model-Free-Opponent-Shaping (M-FOS)

- **Upstream:** [github.com/luchris429/Model-Free-Opponent-Shaping](https://github.com/luchris429/Model-Free-Opponent-Shaping)
- **PRD source note:** ICML 2022 supplementary archive, `https://media.icml.cc/Conferences/ICML2022/supplementary/lu22d-supp.zip`.
- **This repo:** git submodule at `third_party/Model-Free-Opponent-Shaping`.
- **Initialize:** from repo root, `git submodule update --init --depth 1 third_party/Model-Free-Opponent-Shaping`
- **Pinned commit:** `3efb169c39ece7acdaa3f742c5618b7bb11e93d6`
- **License:** no license file found in pinned source.
- **Adapter:** `esl/baselines/mfos_adapter.py`; smoke milestone uses a reduced matrix-game policy-shaping adapter and records the deviation in provenance.

## MBOM

- **Upstream:** [github.com/PKU-RL/MBOM](https://github.com/PKU-RL/MBOM)
- **This repo:** git submodule at `third_party/MBOM`.
- **Initialize:** from repo root, `git submodule update --init --depth 1 third_party/MBOM`
- **Pinned commit:** `d79ba71af843a9073bce0501a0cd85230585fdc0`
- **License:** no license file found in pinned source.
- **Adapter:** `esl/baselines/mbom_adapter.py`; smoke milestone uses a reduced model-based opponent-frequency adapter because upstream targets Python 3.6, Gym, and gfootball.

## Simple Opponent Model / DRON

- **Upstream:** [github.com/hhexiy/opponent](https://github.com/hhexiy/opponent)
- **This repo:** git submodule at `third_party/opponent`.
- **Initialize:** from repo root, `git submodule update --init --depth 1 third_party/opponent`
- **Pinned commit:** `5ab544ca150013cbe5549227107eed64b9928da8`
- **License:** license file present in pinned source.
- **Adapter:** `esl/baselines/simple_opponent_adapter.py`; smoke milestone uses a reduced conditional opponent-action model because upstream is Lua/Torch and dataset-specific.
