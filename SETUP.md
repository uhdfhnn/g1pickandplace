# 从 Git clone 到运行

这份说明复现通过验收的公开 Unitree G1 / Isaac Lab 环境。默认目录结构为：

~~~text
workspace/
├── g1pickandplace/
├── unitree_sim_isaaclab/
├── unitree_ros/
├── unitree_sdk2_python/
├── IsaacLab/
└── cyclonedds/
~~~

setup 脚本会创建后五个同级依赖目录。不要把私人项目或资产加入这些路径。

## 1. 前置条件

- Ubuntu 22.04 或更高版本；当前验收主机使用 Ubuntu/RTX 5090。
- 符合 Isaac Sim 要求的 NVIDIA 驱动和 GPU。
- Git、Conda，以及可使用 `sudo apt-get` 的账号。
- 能访问 GitHub、NVIDIA Python index、PyTorch wheel index 和 Unitree 的
  Hugging Face 资产仓库。
- 足够的磁盘空间；Isaac Sim、Isaac Lab 和仿真资产远大于本仓库源码。

脚本会调用 Unitree 官方 `auto_setup_env.sh`，因此会安装系统包、下载大体积
资产，并可能要求接受 NVIDIA EULA 或生成本地证书。这些是首次安装动作，不适合
在已有的生产 Conda 环境上覆盖运行。

## 2. clone 并安装

~~~bash
mkdir -p workspace
cd workspace
git clone https://github.com/uhdfhnn/g1pickandplace.git
cd g1pickandplace
bash scripts/setup_environment.sh
~~~

默认创建 `unitree_sim_env`。如果机器上已经使用这个名字，请选择一个全新的环境名：

~~~bash
G1PICKPLACE_CONDA_ENV=g1_demo_env bash scripts/setup_environment.sh
~~~

脚本遇到同名 Conda 环境或被修改的依赖仓库会停止，不会自动删除或覆盖用户环境。

## 3. 已锁定的版本

| 依赖 | 已验证版本或 commit | 选择依据与变更风险 |
| --- | --- | --- |
| Isaac Sim | 5.0.0 | Unitree 为 RTX 50 系列推荐的路径；4.5 可能缺少 GPU 支持，5.1 尚未通过本项目可见门控 |
| Python | 3.11 | Unitree Isaac Sim 5.0 安装脚本的版本；其他版本可能没有兼容 wheel |
| PyTorch | 2.7.0 | 与 Isaac Sim 5.0 / Isaac Lab v2.2.0 验证；CUDA 本地后缀可由 NVIDIA 解析为兼容构建 |
| Isaac Lab | `46dff135f44683f031edf346e544fcfd8456b2bb` (`v2.2.0`) | 验收使用的任务和 API；升级可能改变场景、动作项或传感器 API |
| `unitree_sim_isaaclab` | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | 验收使用的公开场景、任务注册和资产布局 |
| `unitree_ros` | `7d6075f7f58588b189b940130e3edab3c839b2df` | 提供已验证的 G1 29-DoF URDF/mesh |
| `unitree_sdk2_python` | `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` | Unitree 官方安装所用 DDS Python 接口 |
| CycloneDDS | `5041f3560c088c99e5088b2b8520b69169621196` | 与 SDK 验证的 0.10.x 构建；升级可能改变 DDS ABI/初始化 |
| teleimager | `b81de448bca9c696d7ce145f4af71c66146d0b69` | `unitree_sim_isaaclab` 锁定的相机子模块 |
| Pinocchio (`pin`) | 2.7.0 | 验收 IK 与 cmeel/HPP-FCL ABI；3.x 未在当前轨迹上验证 |
| LeRobot | 0.4.4（dataset-only） | 验收数据集的原生 v3 写入、重开和视频验证 API；不安装无关的 policy/training 栈 |

这些值是复现锁，而不是声称所有机器只能使用这些版本。修改任一项后，至少需要重新
执行依赖检查、完整单元测试、可见 inspect、plan、rollout 和 LeRobot 重开验证。

## 4. 验证安装

~~~bash
conda activate unitree_sim_env
cd workspace/g1pickandplace
python scripts/check_install.py
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
~~~

如果使用了自定义环境名，把第一行替换为对应名称。`check_install.py` 会检查核心模块、
已验证的软件版本和默认同级 Unitree 仓库。

LeRobot 采用 [`requirements-recording.txt`](requirements-recording.txt) 中的精确
dataset-only 依赖，并使用 `--no-deps` 安装。原因是 LeRobot 的完整训练依赖解析会替换
Isaac Sim 自带的兼容包；本项目只调用 `LeRobotDataset` 的写入、finalize、重开和视频
验证接口。setup 最后会真实导入该类，缺少任何必要依赖都会让安装失败，而不会留下
一个表面成功但不能录制的环境。此环境不承诺支持 LeRobot policy 训练。

## 5. Assimp / HPP-FCL 兼容处理

Pinocchio 2.7.0 的 cmeel 依赖在验收主机上需要先加载
`cmeel.prefix/lib/libassimp.so.5`，否则可能在启动时出现 HPP-FCL 符号错误。
`scripts/run_demo.py` 会依次从以下位置查找：

1. `--assimp-preload` 显式路径；
2. `G1PICKPLACE_ASSIMP_LIB` 环境变量；
3. 当前 `CONDA_PREFIX`；
4. Conda 安装目录下的 `envs/<环境名>`。

自动发现不到时不会猜测系统 ABI。若本机复现符号错误，可以显式指定：

~~~bash
export G1PICKPLACE_ASSIMP_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so.5"
test -f "$G1PICKPLACE_ASSIMP_LIB"
~~~

这里的 Python 3.11 路径来自上表锁定的环境。若修改 Python 或 cmeel 版本，应先找到
实际库文件并重新验证，而不是复制这个路径。

## 6. 运行安全门控

激活环境并从仓库根目录运行。默认只执行可见 inspect 和 plan，不执行物理 rollout：

~~~bash
conda activate unitree_sim_env
cd workspace/g1pickandplace
python scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block."
~~~

inspect 和 plan 都通过后，才可以显式请求 rollout 和原生 LeRobot 录制：

~~~bash
python scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --rollout
~~~

更细的门控命令和验收标准见
[`docs/RUN_ENTRANCE_TEST_DEMO.md`](docs/RUN_ENTRANCE_TEST_DEMO.md)。

## 7. 生成证据不放入源码 Git

`outputs/`、`datasets/`、`videos/` 和 `deliverables/` 是生成产物并已被忽略。
其中现有 evaluator archive 超过 GitHub 普通单文件限制。需要分发时，应核对 manifest
和 SHA-256 后使用 GitHub Release asset、对象存储或显式配置 Git LFS；不要把它们混入
普通源码 commit。
