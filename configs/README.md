# 配置文件目录结构

配置文件按算法分类组织在不同的子文件夹中。

## 目录结构

```
configs/
├── ppo/          # PPO算法配置文件
├── smpe/         # SMPE算法配置文件
├── mappo/        # MAPPO算法配置文件（预留）
├── matrpo/       # MATRPO算法配置文件（预留）
├── happo/        # HAPPO算法配置文件（预留）
└── hatrpo/       # HATRPO算法配置文件（预留）
```

## 配置文件说明

### PPO 配置文件 (`configs/ppo/`)

- `ppo_magent2_selfplay.yaml` - PPO自博弈训练（MAgent2环境）
- `ppo_magent2_selfplay_optimized.yaml` - PPO自博弈训练（优化版本）
- `ppo_magent2_selfplay_test.yaml` - PPO自博弈训练（测试版本）
- `ppo_magent2_selfplay_12v12_8gb.yaml` - PPO自博弈训练（12v12，8GB显存优化）
- `ppo_penalty_magent2_selfplay.yaml` - PPO-Penalty自博弈训练（已弃用，使用PPOTrainer with use_penalty=True）
- `ppo_magent2.yaml` - PPO标准训练（MAgent2环境）
- `ppo_gfootball.yaml` - PPO训练（Google Football环境）

### SMPE 配置文件 (`configs/smpe/`)

- `smpe_magent2_selfplay.yaml` - SMPE自博弈训练（标准配置）
- `smpe_magent2_selfplay_12v12.yaml` - SMPE自博弈训练（12v12配置）
- `smpe_magent2_selfplay_12v12_fast_verify.yaml` - SMPE自博弈训练（快速验证版本）
- `smpe_magent2_selfplay_8gb.yaml` - SMPE自博弈训练（8GB显存优化版本）
- `smpe_magent2_selfplay_12v12_8gb.yaml` - SMPE自博弈训练（12v12，8GB显存优化）

### MAPPO 配置文件 (`configs/mappo/`)

- `mappo_magent2_selfplay_12v12_8gb.yaml` - MAPPO自博弈训练（12v12，8GB显存优化）

### MATRPO 配置文件 (`configs/matrpo/`)

- `matrpo_magent2_selfplay_12v12_8gb.yaml` - MATRPO自博弈训练（12v12，8GB显存优化）

### HAPPO 配置文件 (`configs/happo/`)

- `happo_magent2_selfplay_12v12_8gb.yaml` - HAPPO自博弈训练（12v12，8GB显存优化）

### HATRPO 配置文件 (`configs/hatrpo/`)

- `hatrpo_magent2_selfplay_12v12_8gb.yaml` - HATRPO自博弈训练（12v12，8GB显存优化）

## 使用方法

### 在训练脚本中使用（统一脚本）

```bash
# PPO自博弈训练（12v12，8GB显存优化）
python examples/train_selfplay_unified.py --algorithm ppo --config configs/ppo/ppo_magent2_selfplay_12v12_8gb.yaml

# MAPPO自博弈训练
python examples/train_selfplay_unified.py --algorithm mappo --config configs/mappo/mappo_magent2_selfplay_12v12_8gb.yaml

# MATRPO自博弈训练
python examples/train_selfplay_unified.py --algorithm matrpo --config configs/matrpo/matrpo_magent2_selfplay_12v12_8gb.yaml

# HAPPO自博弈训练
python examples/train_selfplay_unified.py --algorithm happo --config configs/happo/happo_magent2_selfplay_12v12_8gb.yaml

# HATRPO自博弈训练
python examples/train_selfplay_unified.py --algorithm hatrpo --config configs/hatrpo/hatrpo_magent2_selfplay_12v12_8gb.yaml

# SMPE自博弈训练
python examples/train_selfplay_unified.py --algorithm smpe --config configs/smpe/smpe_magent2_selfplay_12v12_8gb.yaml
```

### 在代码中加载

```python
from common.config import load_config

# 加载PPO配置
ppo_config = load_config("configs/ppo/ppo_magent2_selfplay.yaml")

# 加载SMPE配置
smpe_config = load_config("configs/smpe/smpe_magent2_selfplay_8gb.yaml")
```

## 注意事项

1. 所有配置文件路径都是相对于项目根目录的
2. 配置文件按算法分类，便于管理和查找
3. 新算法（MAPPO, MATRPO, HAPPO, HATRPO）的配置文件目录已创建，可以添加对应的配置文件
