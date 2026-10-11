# 电力交易环境：Codex 开发入门

首次准备日期：2026-10-09。仓库来源：https://github.com/moke050113/power-trading-drl

## 1. 在 Codex 中打开项目

项目在原始单步版本基础上，增加24小时环境、B组独立负荷模型和C组策略及PPO运行框架。默认数据为合成教学数据。

当前开发分支为 `codex/bc-basic-modules`。该分支保留基础模块、自动化测试和已审阅的小型示例结果；PPO模型二进制和运行环境不纳入版本管理。

在 Codex 的项目区添加一个本地项目，选择下面的文件夹；如果项目已经建立，在项目菜单选择“编辑项目 / Edit project”，再“添加文件夹 / Add folder”，并设为主文件夹：

```text
D:\Project\power-trading-drl
```

选中这个项目后开始开发聊天。本次检查和策略比较均在 `D:\Project\power-trading-drl` 中运行。

本地项目操作依据：[OpenAI 官方项目说明](https://learn.chatgpt.com/docs/projects)。

其他成员首次下载时：先进入自己选定的存放目录，再执行 `git clone https://github.com/moke050113/power-trading-drl.git`，随后进入 `power-trading-drl`。私有仓库需要对应 GitHub 账号拥有访问权限。

## 2. 使用已有 Python 环境

本机已存在可用环境：

```text
D:\Miniconda\envs\power_trading\python.exe
```

基础环境已验证：Python 3.11.17、NumPy 2.4.6、Gymnasium 1.4.0。基础模块不依赖pytest或训练框架；PPO依赖单独列在 `requirements-ppo.txt`。

在 PowerShell 中依次复制下面的命令：

```powershell
cd "D:\Project\power-trading-drl"
& "D:\Miniconda\envs\power_trading\python.exe" -m unittest discover -s tests -v
& "D:\Miniconda\envs\power_trading\python.exe" test_env.py
& "D:\Miniconda\envs\power_trading\python.exe" test_strategy.py
& "D:\Miniconda\envs\power_trading\python.exe" demo_day.py --csv "../24小时交易示例.csv"
```

基础环境测试应显示30项中29项通过、1项PPO真实训练测试跳过；安装可选PPO依赖后30项全部通过。全天示例应执行小时00到23，最后显示1287425.00元。CSV可以用Excel打开。

文中的 `D:\Project\power-trading-drl` 和 `D:\Miniconda\envs\power_trading\python.exe` 是本机运行示例，不是源码依赖。其他机器进入自己的项目根目录，使用Python 3.11创建环境，例如Windows PowerShell：

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -X utf8 -B -m unittest discover -s tests -v
& ".\.venv\Scripts\python.exe" -X utf8 -B demo_day.py
```

先用 `python --version` 确认创建环境的解释器为3.11；Linux/macOS使用 `.venv/bin/python`。依赖版本记录的是本机已验证组合，其他机器尚未验证。

## 3. 这一阶段改变了什么

- 默认一轮为24小时，`step` 每次推进一小时，第24次交易返回 `terminated=True`。
- 状态由原来的3项变成4项：负荷、合同价格、现货价格、小时。前三项顺序不变。
- 动作仍为电量，例如 `[70.0]` 表示70 MWh，不是70%。超过负荷或低于0的动作沿用原版裁剪规则。
- 原有5个结算字段保留。`total_cost` 是本小时成本，新增 `episode_cost` 是当天累计成本。
- 奖励仍为本小时成本的负数；整轮奖励之和等于全天成本的负数。
- 返回的新状态指向下一小时；终止状态的小时标记为24，市场信息保留最后一小时。
- 结束后需再次 `reset`。未重置就交易、无效动作、错误数据长度会明确报错。
- 支持自定义等长负荷和现货价格曲线；默认曲线是固定合成数据。`horizon=1` 可用于单步核对。

原来的 `test.py`、`test_env.py`、`test_strategy.py` 原样保留。后两个是打印演示脚本，仍只展示第一小时；全天请运行 `demo_day.py`。市场模块也保留原样。

原第一小时动作70 MWh的结果仍为合同成本28000元、现货成本18000元、总成本46000元。新增测试覆盖全天电量平衡、逐小时状态、终止与重置、累计成本、零负荷、动作边界、输入校验及Gymnasium接口。

Gymnasium检查通过，同时建议未来训练时归一化动作空间；当前保留MWh动作便于理解。

## 4. 模型边界与下一步

这是单智能体教学环境。C组已增加PPO短训练运行验证，但尚未进行有意义的训练效果评估。70%覆盖率只是固定示例策略，1287425元不代表最优结果。

目前逐小时选择合同覆盖量、直接观察当小时现货价格，各小时没有跨时段约束；它还不能用于验证真实中长期策略或证明DRL优势。示例不含负电价、售电、预测误差、偏差罚金、EV和工业柔性负荷。

已增加下面的基础策略比较。下一步可明确合同提前锁定的时间和可见预测信息，再加入跨时段约束。EV、工业负荷和多智能体应在这些基础检查后逐步加入。

## 5. 三种基础策略比较

三种策略使用同一份24小时合成负荷、现货价格和400元/MWh合同价格，并通过现有 `PowerTradingEnv.step` 结算，环境接口和交易逻辑未改动。

| 策略 | 本小时合同购买量 | 全天累计成本（元） | 相比全现货节省（元） |
| --- | --- | ---: | ---: |
| 全现货购电 | 0 MWh，全部负荷由现货补足 | 1,514,750.00 | 0.00 |
| 固定70%合同覆盖率 | 本小时负荷 × 70% | 1,287,425.00 | 227,325.00 |
| 逐小时低价市场（简化模型参考策略） | 合同价格低于现货价格时覆盖全部负荷，否则为0 | 1,137,050.00 | 377,700.00 |

参考策略直接观察当小时价格，且允许逐小时选择市场；同价时选现货。当前各小时没有跨时段约束，因此每小时选择较低价格可得到这个简化模型的最低购电成本。这不表示真实中长期合同可以逐小时重签，也不代表DRL训练成果或真实市场的最优策略。

在PowerShell中运行：

```powershell
cd "D:\Project\power-trading-drl"
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B compare_strategies.py
```

脚本会打印全天成本比较，并在项目的 `outputs/strategy_comparison/` 下保存：

- `strategy_comparison.svg`：负荷曲线、三种策略的合同电量曲线、现货电量曲线，以及共用的合同和现货价格曲线；顶部显示全天成本比较。可用浏览器打开。
- `daily_costs.csv`：3行全天成本汇总，可用Excel打开。
- `hourly_details.csv`：3种策略 × 24小时，共72行结算明细，分别列出本小时成本和当天累计成本。

使用 `--output-dir "其他目录"` 可指定输出位置；重复运行会覆盖该目录中的上述3个结果文件。图表由Python标准库生成SVG，无需安装额外依赖。电量单位为MWh，价格单位为元/MWh，成本单位为元；小时索引为0～23。

验证命令：

```powershell
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B -m unittest discover -s tests -v
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B test_env.py
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B test_strategy.py
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B demo_day.py
```

策略比较测试核对相同输入曲线、电量平衡、逐小时及累计成本，并用独立公式检查高价、低价、同价和零负荷时的策略结果。这个比较脚本不涉及训练或柔性负荷。

## 6. B组：可独立测试的基础负荷模型

本次开发分支为 `codex/bc-basic-modules`，从本地现有24小时版本继续开发；保留此前未提交内容，未修改A组环境或市场模块，也未推送GitHub。B组模型暂未接入交易环境，不包含联合训练。

两个模型均以1小时为一步，窗口为 `[start_hour, end_hour)`（包含开始，不包含结束），小时0～23必须顺序调用，下一天先 `reset()`。功率单位为MW，电量单位为MWh。单车60 kWh应写为 `capacity_mwh=0.06`；7 kW写为 `max_power_mw=0.007`。

| 模块 | 接口 | 行为与模型假设 |
| --- | --- | --- |
| `load_models/ev_load.py` | `EVLoad(...).step(hour, requested_power_mw)` | 请求限制在最大功率及目标SOC范围内，窗口外实际用电为0；SOC增加量 = 电网电量 × 效率 / 电池容量。仅充电，不含出行、放电和电池损耗；不会强制保证离开前达到目标，返回 `target_reached` 供调用方检查。 |
| `load_models/industrial_load.py` | `IndustrialLoad(...).step(hour, requested_power_mw)` | 一个可分割、可暂停任务，窗口外电量为0；任务电量超过窗口容量时构造报错。保留未来供电能力，不足时自动提高本小时电量至可行下限，保证24小时完整调用后任务电量守恒。请求是偏好，实际功率可能高于请求；无生产顺序、最短开机时间等约束。 |

`step` 返回字典，两者共用 `hour`、`load_mwh`、`power_mw`；EV另有 `soc`、`battery_energy_mwh`、`target_reached`，工业模型另有 `remaining_energy_mwh`、`minimum_required_mwh`、`completed`。模型会裁剪过大请求，拒绝负数、非有限数值和跳小时调用；无效请求不推进状态。

独立使用示例（可保存到自己的脚本）：

```python
from load_models.ev_load import EVLoad
from load_models.industrial_load import IndustrialLoad

ev = EVLoad(start_hour=18, end_hour=24)
industry = IndustrialLoad(task_energy_mwh=12, max_power_mw=4, start_hour=8, end_hour=16)
ev_curve = [ev.step(hour, 0.007)['load_mwh'] for hour in range(24)]
industrial_curve = [industry.step(hour, 2)['load_mwh'] for hour in range(24)]
print(ev_curve, industrial_curve)
```

独立测试（在项目根目录PowerShell运行）：

```powershell
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B -m unittest discover -s tests -p test_ev_load.py -v
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B -m unittest discover -s tests -p test_industrial_load.py -v
```

## 7. C组：基准策略、评估和PPO运行框架

`agents/baseline.py` 的 `BaselinePolicy(strategy, seed)` 支持 `spot`（全部现货）、`fixed70`（70%合同）和 `random`（每小时独立均匀采样0～1合同比例）。`predict(observation, deterministic=True)` 返回 `(action, None)`；随机基准始终按自己的随机流采样，`deterministic` 参数只为兼容SB3接口。相同种子和调用顺序可复现结果；不在每一轮重置随机流。

`experiments/evaluate.py` 的 `evaluate_policy(policy, episodes, seed, env_factory)` 每轮创建新环境，完整运行至终止或截断。累加每步 `info['total_cost']` 及原始Reward，返回每轮记录和均值、总体标准差（ddof=0）。本小时成本与累计成本分别使用，避免重复累加 `episode_cost`。默认3轮固定合成曲线不是3个不同市场场景。

```powershell
cd "D:\Project\power-trading-drl"
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B -m experiments.evaluate --episodes 3 --seed 0
```

输出保存在 `outputs/baseline_evaluation.json`。本机种子0的3轮结果为：全部现货成本均值1,514,750.00元，70%合同1,287,425.00元，随机比例1,369,674.59元；对应Reward均值为成本的负数。固定策略成本标准差为0，随机比例为6,353.71元，这些只是教学示例。

`agents/train_ppo.py` 基于 [Stable-Baselines3的PPO接口](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) 构建单智能体框架。`ContractRatioAction` 包装器将训练动作 `[0,1]` 转换成“当小时负荷 × 比例”的MWh，A组的环境动作单位仍为MWh。仅训练Reward由 `VecNormalize` 缩放，观察值不缩放；评估回到原始元成本和Reward。重新加载模型必须使用同样的比例动作包装器，不能直接给原环境传比例。

本机已准备 `.venv-ppo`，继承Python 3.11解释器版本，依赖隔离，原Conda环境未安装新包。虚拟环境不随Git提交，其他成员需先按下方命令创建。已验证Python 3.11.17、Gymnasium 1.4.0、NumPy 2.4.6、Stable-Baselines3 2.9.0、CPU PyTorch 2.14.1。运行：

```powershell
& ".\.venv-ppo\Scripts\python.exe" -X utf8 -B -m agents.train_ppo --timesteps 96 --seed 0
& ".\.venv-ppo\Scripts\python.exe" -X utf8 -B -m experiments.evaluate --model "outputs/ppo_smoke/ppo_power_trading.zip" --episodes 3 --seed 0 --output "outputs/ppo_smoke/evaluation.json"
& ".\.venv-ppo\Scripts\python.exe" -X utf8 -B -m unittest discover -s tests -v
```

训练入口默认96步，`n_steps=24`，批大小24；SB3会按完整24步rollout向上取整实际训练步数，报告同时保存请求和实际步数。只验证训练、保存、加载及2轮评估能运行，不要求改善成本。`outputs/ppo_smoke/` 包含模型zip、训练Reward归一化状态pkl及 `smoke_report.json`。继续训练时须恢复归一化状态；当前入口只支持从头短训练。仅加载可信的模型和pkl文件。

其他机器初次准备PPO运行环境时（需联网）：

```powershell
python -m venv .venv-ppo
& ".\.venv-ppo\Scripts\python.exe" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
& ".\.venv-ppo\Scripts\python.exe" -m pip install -r requirements-ppo.txt
```

全部30项测试中，原有9项保留，B组6项，基准与评估6项，PPO3项（含真实24步训练、保存、加载测试），2026-10-11补充6项边界和接口检查（工业模型包含100组随机可行任务）。基础Conda环境仅跳过真实训练项，PPO环境全部运行。Gymnasium/SB3的动作空间建议属于警告，不是失败。原有两个演示及24小时示例也已回归验证。完整检查记录见 [2026-10-11测试报告](outputs/validation_2026-10-11/test_report.md)。

## 8. 后续与A组对接的接口

- **离线负荷曲线**：从B组收集24项 `load_mwh`，与基础负荷逐小时相加后传入现有 `PowerTradingEnv(load_profile=..., spot_price_profile=...)`。三种策略应使用同一负荷和价格曲线；A组当前总负荷上限1000 MWh、价格上限2000元/MWh，需在接入前核对。
- **在线柔性控制**：A组需要在每轮reset时同步重置B组模型，每小时在同一时钟先处理充电/工业功率请求、取得实际电量，再结算交易；加入SOC、剩余任务量、窗口和可行功率信息到观察值。当前环境观察/动作不含这些信息，暂未接入。
- **交易策略**：C组基准读取 `[负荷, 合同价, 现货价, 小时]`，输出形状 `(1,)` 的MWh数组。PPO经比例包装器适配，单步版本也有接口测试。后续若A组改变状态顺序、动作单位或时间步长，需要同步适配器。
- **评估结算**：继续提供Gymnasium五元组、每小时 `info['total_cost']` 和原始Reward；累计成本可保留在 `episode_cost`，评估只累加本小时成本。若增加SOC未达标罚金、工业任务罚金，应明确它们进入成本还是仅进入Reward；目前Reward = 负成本。

当前未实现多智能体、负荷与交易联合训练、真实数据泛化评估。B组EV不保证目标达成，工业模型按可行下限强制调度；这些假设需要A、B组在接入之前确认。

## 9. 示例结果的版本管理

`outputs/` 采用白名单，仅提交 [结果目录说明](outputs/README.md) 中列出的CSV、SVG、JSON和正式测试报告。其他运行输出默认忽略。模型ZIP、PyTorch权重、归一化PKL、虚拟环境、缓存和临时文件仅在本地保留；不要将示例成本或Smoke报告当成训练效果证明。

正式测试报告中的绝对路径记录当时的本机测试环境；跨电脑复现使用自己的项目目录及上述虚拟环境命令。本地模型未上传，README中的模型评估命令需先在本机运行训练入口生成模型。
