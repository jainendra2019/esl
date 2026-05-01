# Baseline Onboarding Status

| Method | Family | Status | Smoke result | Official/reduced | Reason |
|---|---|---|---|---|---|
| Independent PPO | independent_ppo | READY | PASS | official | Official PPO.py is present and loaded directly by the adapter. |
| M-FOS | mfos | PARTIAL | PASS | reduced | Official source is pinned, but full M-FOS meta-game training is not yet wrapped. |
| MBOM | mbom | PARTIAL | PASS | reduced | Official code is pinned but incompatible with the current lightweight matrix-game runtime. |
| Simple Opponent Model | simple_opponent_model | PARTIAL | PASS | reduced | Official source is pinned but cannot run directly in this Python matrix-game stack. |
