# JAX qpos-torso 100k 训练记录

## 概要

| 项目 | 值 |
|------|-----|
| 框架 | JAX 0.5.3 + Flax + Orbax |
| config name | `pi05_xingchen_qpos` |
| exp_name | `qpos_torso` |
| 训练步数 | 100,000 (完成到 step 99999) |
| 训练时长 | 约 12h 8min |
| 集群 | 百度 zhongwei / embody 队列, 2 nodes |
| 启动时间 | 2026-07-29 16:50 (UTC, 从 checkpoint metadata 推算) |
| 启动脚本 | `scripts/run_qpos_train.sh` (原在 stash 中，现已 commit) |
| 训练入口 | `python scripts/train.py pi05_xingchen_qpos --overwrite` |

## 数据

| 项目 | 值 |
|------|-----|
| 数据格式 | ARIO (OSS 直读) |
| s3_prefixes | `oss://shengshu-base2-test/xiaojun/新积木/` |
| 任务 | 新积木 (搭积木, prompt: "fold clothes") |
| episodes | 795 |
| 总帧数 | 72,526 (downsample 6x 后) |
| 视频视角 | cam_high, cam_left_wrist, cam_right_wrist (320x240) |
| video_downsample_rate | 6 |
| min_frames | 1 (未过滤) |
| load_instructions | True |

来源确认：job 日志显示 `ArioConfig(s3_prefixes='oss://shengshu-base2-test/xiaojun/新积木/', ...)`

## Action 表示

| 项目 | 值 |
|------|-----|
| 表示方式 | qpos (关节角) |
| ACTION_DIM | 26 (padded to 32 for model) |
| 拼接顺序 | torso_qpos(4) + head(2) + eef_left(9) + gripper_left(1) + eef_right(9) + gripper_right(1) |
| PT 文件 | torso.pt, head.pt, eef_left.pt, gripper_cmd.pt, eef_right.pt |
| delta_action_mask | `make_bool_mask(4, -2, 9, -1, 9, -1)` |
| delta 部分 | torso_qpos(4), eef_left(9), eef_right(9) |
| absolute 部分 | head(2), gripper_left(1), gripper_right(1) |

## 模型配置

| 项目 | 值 |
|------|-----|
| 模型 | pi0.5 (PaliGemma + Gemma expert + action projections) |
| pi05 | True |
| action_dim | 32 |
| action_horizon | 50 |
| base weights | `./checkpoints/pi05_base_jax/params` (JAX 原始权重) |

## 训练超参

| 项目 | 值 |
|------|-----|
| batch_size | 64 |
| num_workers | 8 |
| optimizer | AdamW (clip_gradient_norm=1.0) |
| ema_decay | 0.999 |
| lr_schedule | CosineDecay |
| warmup_steps | 1,000 |
| peak_lr | 1e-4 |
| decay_steps | 3,000 |
| decay_lr | 1e-5 |
| save_interval | 2,000 |
| keep_period | 10,000 |
| enable_async_checkpointing | False (因 multi-node Orbax bug 关闭) |

## Checkpoint 存储

| 位置 | 内容 |
|------|------|
| 本地 | `checkpoints/pi05_xingchen_qpos/qpos_torso/{step}/` |
| OSS | `oss://shengshu-base2-test/ali-checkpoint/mayuanbo/jax_qpos_torso/{step}/` |
| 已上传 OSS 的 steps | 6000, 8000, 10000, ..., 40000 |
| 仅本地的 steps | 50000, 60000, 70000, 80000, 90000, 99999 |

Checkpoint 格式为 Orbax（JAX 生态），包含 `params/`, `train_state/`, `assets/`。

## 训练结果

- Loss 在 step ~10000 收敛到 0.006-0.007 范围
- 训练正常完成，无中途崩溃（关闭 async checkpoint 后）

## Git 来源

训练代码来自 stash `stash@{0}` (message: "On jax/qpos-torso: WIP: qpos-torso changes")，基于 `jax/qpos-torso` 分支（tip: `5e1432f`，即 fanyiming 的 `fym/jax/multi-vision`）。stash 中的改动包括：

- `scripts/run_qpos_train.sh` — JAX 训练启动脚本（新文件）
- `scripts/train.py` — 添加 checkpoint save retry 逻辑
- `src/openpi/datasets/ario_dataset.py` — ACTION_DIM 31→26, PT_FILES 改用 torso.pt
- `src/openpi/policies/xingchen_policy.py` — ACTION_DIM 31→26
- `src/openpi/training/checkpoints.py` — 关闭 async checkpoint
- `src/openpi/training/config.py` — 新增 pi05_xingchen_qpos config, save_interval 改为 2000

注意：`origin/xingchen` 分支上的 commit `812ac9f` ("switch action representation from eef to qpos torso") 包含了 dataset/policy/config 的改动，但**不包含** `run_qpos_train.sh` 和 `train.py` 的修改。这些只在 stash 中存在。

## 与 origin/xingchen 上 pi05_xingchen (PyTorch) 的对比

| | JAX qpos (本次) | PyTorch xingchen (origin/xingchen) |
|--|--|--|
| 框架 | JAX | PyTorch DDP (torchrun) |
| 启动脚本 | run_qpos_train.sh | train_xingchen.sh |
| config name | pi05_xingchen_qpos | pi05_xingchen |
| base weights | pi05_base_jax | pi05_base_pytorch |
| warmup_steps | 1,000 | 5,000 |
| peak_lr | 1e-4 | 5e-5 |
| decay_steps | 3,000 | 500,000 |
| decay_lr | 1e-5 | 5e-5 |
| save_interval | 2,000 | 5,000 |
| checkpoint 格式 | Orbax | safetensors |
| 数据 | 新积木 (795 ep) | 新积木 |
