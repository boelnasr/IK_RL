# RL IK Project — Tune-up Checklist

> Tick the boxes ☑ as you complete each item.

---

## 0  Prep

- [ **X**] **Create a “dev-tune” branch** so *master* stays runnable  
- [ ] Run unit tests → ensure baseline still passes before editing  

---

## 1  Quick Wins (≤ 1 day)

| Done | Priority | Item | File / Function |
|------|----------|------|-----------------|
| [**X**] | 💡 | Switch Bullet to **DIRECT** mode during training | `InverseKinematicsEnv.__init__` → `p.connect(p.DIRECT)` |
| [ ] | 💡 | Call `get_actions(..., eval_mode=False)` after warm-up **or** gradually set `pd_weight → 0` | `MAPPOAgent.train()` |
| [**X** ] | 💡 | Clamp orientation error to **± π** before reward calc | `reward_function.compute_reward` |
| [ ] | 💡 | Replace inner `for`-loops in trajectory build with `torch.stack` / `torch.cat` | `MAPPOAgent.train()` |

---

## 2  RL-side Tweaks (1–3 days)

| Done | Priority | Item | File / Function |
|------|----------|------|-----------------|
| [ **x**] | 🔧 | Remove HER / Prioritized buffers from **on-policy** path | `MAPPOAgent.train()`, `update_policy_with_experiences` |
| [ ] | 🔧 | **Share actor parameters** across joints (add `agent_id` embedding) | `models.JointActor` |
| [ **X**] | 🔧 | Replace custom entropy adaptation with **target-KL schedule** | `MAPPOAgent.update_policy` |
| [ ] | 🔧 | Add **soft target critic** (τ ≈ 0.005) for advantages | `MAPPOAgent.__init__`, `update_policy` |
| [ ] | 🔧 | Insert gradient NaN / explode guard, skip update if triggered | `update_policy`, `update_policy_with_experiences` |

---

## 3  Environment & Curriculum (2–5 days)

| Done | Priority | Item | File / Function |
|------|----------|------|-----------------|
| [ ] | 🔧 | Sample 20 % targets on workspace boundary, 10 % near singularities | `sample_valid_target_position` |
| [**X** ] | 🔧 | Success threshold **per joint**: `σ_i = 0.1 rad · e^{-3·difficulty_i}` | `is_agent_success` |
| [ ] | 🔧 | Parallelise roll-outs with **SubprocVecEnv / Ray** | `envs/vec_env.py`, training launcher |
| [ ] | 🔧 | Add **domain randomisation** (mass ± 10 %, friction, sensor noise) | `InverseKinematicsEnv.reset` |
| [ ] | 🛠 | Swap `POSITION_CONTROL` for **TORQUE_CONTROL + inner PD @ 1 kHz** | `InverseKinematicsEnv.step` |

---

## 4  Network & Loss Surgery (~1 week)

| Done | Priority | Item | File / Function |
|------|----------|------|-----------------|
| [ ] | 🛠 | Introduce **GNN encoder** (nodes = joints) | `models/gnn_encoder.py` |
| [ ] | 🛠 | Add auxiliary heads: predict Cartesian Δ & Jacobian cond-# | `CentralizedCritic.forward` |
| [ ] | 🛠 | Implement adaptive-KL curriculum (target KL ↑ when success > 80 %) | `update_policy` |
| [ ] | 🛠 | Add λ-return look-ahead (`t+T+1`) in advantage calc | `compute_individual_gae` |

---

## 5  Diagnostics & CI

| Done | Priority | Item | File / Function |
|------|----------|------|-----------------|
| [ ] | 💡 | Enable TensorBoard/WandB: advantages, TD-error hist, PD vs RL magnitude | `training_metrics.log_episode` |
| [ ] | 💡 | Save **worst-10** trajectories each epoch for regression tests | new util in `training_metrics` |
| [ ] | 💡 | Unit-test analytic Jacobian vs numeric (< 1e-3 RMS) | `tests/test_jacobian.py` |
| [ ] | 💡 | Add CLI flags `--no-pd`, `--pd-anneal=exp`, `--vectorised N` | `args.py` |

---

### How to use

1. Copy this file as `CHECKLIST.md` in your repo.  
2. Check off items as your commits land.  
3. After finishing each section, rerun unit tests and a 500-episode benchmark.
