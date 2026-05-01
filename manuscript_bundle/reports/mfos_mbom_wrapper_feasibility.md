# Official M-FOS and MBOM Wrapper Feasibility

## Decision

Do **not** add M-FOS or MBOM to the corrected main performance grid yet.

Neither pinned official source exposes a thin, faithful adapter into the current shared focal-agent protocol:

- focal learner is agent 0,
- environment emits one sampled 2-action interaction per step,
- all methods share the same opponent schedule,
- payoff is agent-0 realized payoff,
- no method may use a separate synthetic process.

## Status Summary

| Baseline | Status | Source | Pinned commit | Smoke run |
|---|---|---|---|---|
| M-FOS | **INCOMPATIBLE** with exact shared focal protocol | `third_party/Model-Free-Opponent-Shaping` | `3efb169c39ece7acdaa3f742c5618b7bb11e93d6` | Not run |
| MBOM | **INCOMPATIBLE** with exact shared focal protocol | `third_party/MBOM` | `d79ba71af843a9073bce0501a0cd85230585fdc0` | Not run |

No faithful adapter was implemented. Any adapter would require changing the evaluation protocol or reimplementing missing environment/model pieces, which would violate the “official code only if genuinely used” constraint.

## M-FOS Audit

### Official structure inspected

- Entry points:
  - `third_party/Model-Free-Opponent-Shaping/src/main_mfos_ppo.py`
  - `third_party/Model-Free-Opponent-Shaping/src/main_mfos_self.py`
  - `third_party/Model-Free-Opponent-Shaping/src/main_non_mfos.py`
- Policy/update:
  - `third_party/Model-Free-Opponent-Shaping/src/ppo.py`
  - `PPO`, `Memory`, `ActorCritic`
- Environment/meta-game:
  - `third_party/Model-Free-Opponent-Shaping/src/environments.py`
  - `MetaGames`, `SymmetricMetaGames`, `NonMfosMetaGames`

### Training loop

`src/main_mfos_ppo.py` constructs:

- `env = MetaGames(batch_size, opponent=args.opponent, game=args.game, ...)`
- `action_dim = env.d`
- `state_dim = env.d * 2`
- `ppo = PPO(state_dim, action_dim, ...)`

Then each step calls:

- `action = ppo.policy_old.act(state, memory)`
- `state, reward, info, M = env.step(action)`
- `ppo.update(memory)` once per meta-episode

This is official M-FOS logic, but its action is not a sampled matrix-game action. It is a continuous vector of policy parameters for the inner differentiable game.

### Policy object

`src/ppo.py` uses a continuous Gaussian actor:

- `ActorCritic.actor` outputs `action_dim * 2`
- action distribution is `torch.distributions.MultivariateNormal`
- `act()` returns a continuous action vector

For IPD, `env.d == 5`; M-FOS’s outer action is a 5-dimensional memory-one policy parameter vector, not C/D.

### Opponent-modeling / shaping logic

The opponent is embedded in `MetaGames.step()`:

- `NL` / `MAMAML`: opponent parameters update by differentiating through `l1`
- `LOLA`: opponent update includes higher-order gradient terms
- `BR`: opponent best response is computed by a 1000-step gradient loop

This is the paper’s differentiable meta-game setup, not a black-box focal-agent wrapper around sampled repeated interactions.

### Environment assumptions

Supported official games:

- `IPD`
- `IMP` / matching pennies-like zero-sum game
- `chicken`

Missing for our exact grid:

- no Stag Hunt task;
- no 3-opponent schedule with agent IDs/types;
- no focal agent 0 interacting with sampled opponents from a shared population;
- no per-step realized C/D action chosen by M-FOS;
- official main scripts use large vectorized batches and meta-episodes (`batch_size=4096`, `num_steps=100`, `max_episodes=1024`);
- official training scripts call `.cuda()` directly in logging tensors, which is not portable on this macOS CPU/MPS environment without patching.

### Can it run on 2-action repeated matrix games?

Partially, but not in the required sense.

M-FOS can run its own differentiable IPD/IMP/chicken meta-games. It does not expose a faithful direct policy object that, at each shared focal-agent step, consumes the same opponent schedule and returns a C/D action. Converting its continuous memory-one parameter vector into sampled C/D actions inside our environment would be a new evaluation bridge, not a thin official wrapper.

### Feasibility classification

**INCOMPATIBLE** for the main shared focal-agent grid.

Possible future appendix route:

- Run official M-FOS in its native `MetaGames` IPD/IMP/chicken protocol.
- Report it as a native-protocol appendix baseline, with clear caveats that it is not schedule-matched to ESL.

This should not be labeled MAIN_READY for the corrected main grid.

## MBOM Audit

### Official structure inspected

- Entry point:
  - `third_party/MBOM/main.py`
- Trainer:
  - `third_party/MBOM/trainer.py`
- Core policy:
  - `third_party/MBOM/policy/MBOM.py`
- Opponent model:
  - `third_party/MBOM/policy/Opponent_Model.py`
- PPO base:
  - `third_party/MBOM/baselines/PPO.py`
  - `third_party/MBOM/baselines/Base_ActorCritic.py`
- Trajectory helpers:
  - `third_party/MBOM/utils/rl_utils.py`
- Config:
  - `third_party/MBOM/config/gfootball_conf.py`
  - `third_party/MBOM/requirement.txt`

### Training loop

`trainer.py` expects:

- an external `env`,
- an external `env_model`,
- two agent configs (`confs`),
- multiprocessing workers,
- `collect_trajectory(agents, env, args, ...)`.

The checked-in `main.py` does not instantiate these objects. It contains placeholders:

- `#env`
- `#env_model`

and calls `trainer(args, logger)` even though `trainer` is defined as `trainer(args, logger, env, env_model, confs)`.

### Policy object

`policy/MBOM.py` defines `class MBOM(PPO)` and extends PPO with:

- `Opponent_Model`,
- `OM_Buffer`,
- recursive imagined opponent policies (`om_phis`),
- Bayesian/mixing logic,
- rollout through `self.env_model`.

`choose_action()` calls `_get_mixed_om_hidden_prob()`, which calls `_gen_om_phis()`, which calls `_rollout()`.

### Opponent-modeling logic

The key MBOM algorithm depends on:

- learning an opponent action model from `(state, opponent_action)`;
- recursively imagining opponent updates via `_rollout`;
- using `env_model.step(state, actions)` to evaluate imagined action branches;
- mixing imagined opponent policies through `_cal_mix_ratio`.

This means a faithful MBOM wrapper requires a compatible environment model object, not only a live environment.

### Environment assumptions

The repo is configured for football-style tasks:

- `requirement.txt`: `numpy==1.18.1`, `torch==1.4.0`, `gym==0.17.2`, `gfootball=2.0.5`
- `config/gfootball_conf.py`: state size `24`, action size `11`, opponent action size `11`
- README lists the same old dependencies.

The pinned repo does not include a ready 2-action matrix-game environment or env-model implementation. A faithful ESL wrapper would need to design and validate a new `env_model` with the exact semantics MBOM expects:

- `reset()`
- `step(state, actions)`
- vectorized branching over all opponent actions
- reward and done tensors compatible with `_rollout`

That is substantive environment-model engineering, not a thin adapter.

### Additional implementation blockers

Observed code-level issues in the pinned source:

- `main.py` has placeholder env/env_model creation and mismatched trainer/tester calls.
- `utils/rl_utils.py` checks `type(agent).__name__ == "MBAM"` rather than `"MBOM"`, so official collection logic would not identify MBOM as written.
- `policy/MBOM.py::_rollout()` calls `self.env_model.reset()` and `self.env_model.step(...)`; passing `env_model=None` would fail for `num_om_layers > 1`.
- Default configs target 11-action football, not 2-action matrix games.

### Can it run on 2-action repeated matrix games?

Not from the pinned official repo without implementing a new matrix-game `env` and `env_model`, changing configs, and likely patching collection logic. That would be a port/reimplementation effort, not the thinnest possible faithful wrapper.

### Feasibility classification

**INCOMPATIBLE** for the main shared focal-agent grid.

Possible future appendix route:

- Build a separate MBOM-port milestone that explicitly implements a 2-action matrix-game environment model and documents it as a port.
- Only after that, run smoke tests to decide whether it can be appendix or main-ready.

It is not MAIN_READY now.

## Final Recommendation

For the corrected v1 main performance grid:

- Keep **ESL**.
- Keep **Independent PPO** if the official `PPO.py` shared wrapper is accepted.
- Keep **Simple Opponent Model** only as APPENDIX_ONLY/reduced unless a faithful DRON wrapper is separately approved.
- Keep **M-FOS** and **MBOM** out of the main figure.

Do not run M-FOS/MBOM smoke runs under reduced approximations and do not silently include them in `main_performance_grid_protocol_fixed_smoke.png`.

If M-FOS or MBOM must appear in the main paper figure, the next milestone should be a dedicated porting milestone with explicit acceptance criteria for a matrix-game environment model and a method-faithfulness review.
