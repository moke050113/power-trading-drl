# A、B、C组基础模块测试报告

日期：2026-10-11（北京时间）。路径：`D:\Project\power-trading-drl`。分支：`codex/bc-basic-modules`。沿用现有代码，没有重新生成项目，没有commit、push或改动main。

## 1. 环境和依赖

- 指定基础解释器：`D:\Miniconda\envs\power_trading\python.exe`，Python 3.11.17、NumPy 2.4.6、Gymnasium 1.4.0。
- 检查了 `requirements.txt` 与 `requirements-ppo.txt`。基础解释器没有Stable-Baselines3和PyTorch，因此真实训练测试跳过；如要在该解释器训练，需要安装Stable-Baselines3 2.9.0和PyTorch 2.14.1及其依赖。
- 复用项目现有 `.venv-ppo\Scripts\python.exe`，Python 3.11.17、NumPy 2.4.6、Gymnasium 1.4.0、Stable-Baselines3 2.9.0、PyTorch 2.14.1+cpu。本次没有安装任何包，也没有修改原Conda环境。
- 两个解释器的 `pip check` 均输出 `No broken requirements found`。

## 2. 实际运行命令

所有命令在项目根目录运行。先运行现有24项测试，再增加6项针对性的检查并重新运行完整测试。以下为最终验证命令：

```powershell
cd "D:\Project\power-trading-drl"
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B -m unittest discover -s tests -v
& ".\.venv-ppo\Scripts\python.exe" -X utf8 -B -m unittest discover -s tests -v
& "D:\Miniconda\envs\power_trading\python.exe" -m pip check
& ".\.venv-ppo\Scripts\python.exe" -m pip check
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B test_env.py
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B test_strategy.py
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B demo_day.py
& "D:\Miniconda\envs\power_trading\python.exe" -X utf8 -B -m experiments.evaluate --episodes 3 --seed 0 --output outputs/validation_2026-10-11/baseline_evaluation.json
& ".\.venv-ppo\Scripts\python.exe" -X utf8 -B -m agents.train_ppo --timesteps 96 --seed 0 --output-dir outputs/validation_2026-10-11/ppo_smoke
& ".\.venv-ppo\Scripts\python.exe" -X utf8 -B -m experiments.evaluate --model outputs/validation_2026-10-11/ppo_smoke/ppo_power_trading.zip --episodes 3 --seed 0 --output outputs/validation_2026-10-11/ppo_evaluation.json
git diff --check
git branch --show-current
git status --short
```

另以指定解释器的 `-c` 检查独立市场模块正常结算、零电量、负电量拒绝，以及模型zip完整性、报告有限数值、24步评估和成本/Reward对应关系。

## 3. 测试结果

| 检查 | 通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: |
| 最初现有测试，基础解释器 | 23 | 0 | 1 |
| 最终全部测试，基础解释器 | 29 | 0 | 1 |
| 最终全部测试，PPO虚拟环境 | 30 | 0 | 0 |
| 96步PPO Smoke Test及独立加载评估 | 完成 | 0 | 0 |

计数为unittest测试方法数，不与pytest的subtests单独计数相加。基础环境跳过项为 `test_real_training_save_load_and_evaluate`，该项在PPO环境真实运行通过。基础测试输出1类Gymnasium动作空间警告；PPO环境另有SB3建议对称动作空间的警告，不影响测试通过。

新增离线接口测试首次失败：解析公式成本1,292,575.5元，float32基准动作结算1,292,575.5004119873元，相差约0.000412元。原因是新增测试容差过严；将成本误差容差设为0.01元后，两套环境重新运行全部通过。没有修改环境或基准策略精度定义。

原有 `test_env.py` 首小时仍为70 MWh合同、30 MWh现货，成本46,000元、Reward -46,000；`test_strategy.py` 成本60,000至40,000元；`demo_day.py` 完成00～23小时，全天成本1,287,425元。

## 4. A、B、C组完成情况和接口

| 分组 | 当前完成情况 | 本次验证范围 | 未完成范围 |
| --- | --- | --- | --- |
| A | 24小时逐步交易、状态推进、结算、累计成本、终止和重置可用 | 原有7项环境测试、2项策略比较测试、演示；独立市场模块正常与负电量输入 | 真实市场决策时序、中长期锁定、跨时段市场约束；独立market模块尚未被环境实际调用 |
| B EV | 独立充电、SOC和功率限制、窗口控制可用 | SOC=0/1、初始已满足目标、末小时窗口、零功率、效率、电网与电池电量区分、可满足与不可满足需求 | 离开前需求保证、出行放电、跨午夜窗口、多天SOC延续 |
| B 工业 | 独立可分割任务及可行性下限调度可用 | 零任务、不可行任务拒绝、电量守恒、窗口、功率限制、100组随机可行任务 | 不可分割生产、启停、工序先后、真实生产约束 |
| C | 三种基准、成本/Reward统计、PPO训练保存加载链路可用 | 随机种子复现、已知三轮成本的独立均值/总体标准差公式、24步结束、96步Smoke Test | 有意义的策略效果评估、训练/验证/测试数据划分、在线柔性负荷控制、断点续训入口 |

接口结论：已测试B组输出24项 `load_mwh` 与基础负荷相加，再传A组 `load_profile`，最后由C组全部现货、70%合同、随机策略完整评估。零负荷、小数负荷和高负荷曲线经过PPO比例包装器也能正常结算。这里是离线曲线兼容，不等于在线联合环境已经完成。

A/C动作单位保持MWh，B组输入MW，因时间步固定1小时，输出电网用电MWh。每轮必须分别reset负荷模型与环境，小时0～23顺序运行；窗口为左闭右开。评估只累加本小时 `total_cost`，不重复累加 `episode_cost`。PPO加载必须使用 `ContractRatioAction`，否则比例会被错误当作MWh。

## 5. 成本与Reward结果

固定合成曲线，种子0，3轮；标准差采用总体标准差（ddof=0）。Reward为原始成本的负数。

| 策略 | 成本均值（元） | 成本标准差（元） | Reward均值 | Reward标准差 |
| --- | ---: | ---: | ---: | ---: |
| 全现货 | 1,514,750.00 | 0.00 | -1,514,750.00 | 0.00 |
| 固定70%合同 | 1,287,425.00 | 0.00 | -1,287,425.00 | 0.00 |
| 随机比例 | 1,369,674.59 | 6,353.71 | -1,369,674.59 | 6,353.71 |
| PPO 96步后确定性评估 | 1,514,750.00 | 0.00 | -1,514,750.00 | 0.00 |

PPO实际训练96步，内部加载后2轮各24步，外部再加载3轮各24步；zip校验通过，归一化pkl非空，报告和成本/Reward均为有限数值。成本未优于全现货，只能说明框架可以运行，不能称为训练成果或最优策略。

## 6. 修复与新增文件

没有发现需要修改核心业务代码的明确错误。本次没有修改A/B/C实现、市场机制、State、Action或Reward。

- 新增 `tests/test_module_integration.py`：6项边界和接口测试，工业随机任务包含100个子场景；修正该新增测试的成本浮点容差。
- 更新 `README.md`：将测试计数更新为30项，并链接本报告。
- 新增本报告、`baseline_evaluation.json`、`ppo_evaluation.json`，以及本次 `ppo_smoke/` 的模型、归一化状态和运行报告。原有Smoke结果没有覆盖。

Git中原有 `.gitignore`、`env/power_trading_env.py` 的未提交改动在本次检查前就已存在；不是本次修复。

## 7. 记录的问题与后续方案

1. **动作空间建议**：A组保留MWh动作，PPO包装器使用[0,1]。Gymnasium/SB3建议归一化或对称[-1,1]，本次只记录，没有为消除警告改环境。
2. **EV默认需求不可满足**：默认电池0.06 MWh、SOC 0.2、最大0.007 MW、效率0.9、窗口18～24。最大电网电量0.042 MWh，电池增加0.0378 MWh，SOC最高0.83；距目标1.0还差电池电量0.0102 MWh。模型正确返回False，但不会主动保证需求，也不拒绝不可达目标。后续先确认需要可行性报错、未达标罚金还是允许缺电，再设计接口；当前保持不变。
3. **工业自动调度简化**：为了守恒，实际功率可能高于请求。算法适合可分割任务，不等同于真实工业生产控制；评估调度策略时必须同时记录请求、实际功率及强制下限。
4. **决策信息与潜在泄露**：状态提供当小时实际现货价格，再选择合同量。对当前教学决策时序这是显式假设；若声称合同在日前或更早锁定，这就是使用决策时尚不可知的信息。固定同一合成曲线训练评估也有记忆曲线、无法检验泛化的风险。后续方案应先确定合同锁定时刻，再区分预测与实现价格，划分不同日期的训练/验证/测试集，本次未改定义。
5. **多时间尺度简化**：按小时重选合同、无跨时段合同约束，EV与工业均按1小时且单日重置。尚不支持短时间充电控制与长期合同之间的真实联动，也不支持跨午夜窗口。
6. **尚未在线接入B组**：当前状态没有SOC、剩余任务量或窗口，动作只有合同电量。在线对接涉及State/Action/Reward重大设计，需要另行讨论，不可把本次离线接口测试当作联合训练完成。
7. **独立市场输入检查较弱**：`market/` 正常交易、零电量及负电量拒绝检查通过，但独立调用时NaN电量会传播到成本；它们目前未被A组环境调用，A组环境对动作有有限值检查。后续若要正式复用，建议单独补充有限价格/电量校验，本次未改旧模块。
8. **PPO限制**：仅短训练，未提供断点续训入口；续训需恢复训练Reward归一化状态。观察值量纲未归一化；优化策略和归一化方案应在真实实验设计前明确。模型依赖A组的hour/load_profile属性，需要在环境接口变更时同步适配。
9. **精度**：观察和基准动作使用float32，小数负荷会有小额舍入差。本次成本公式核对允许0.01元误差，实际发现误差约0.000412元；未更改单位或业务公式。

## 8. 是否建议提交GitHub

建议作为“基础模块可运行检查点”提交到当前功能分支并供团队审阅，不建议以训练有效或真实市场机制完成的名义提交，也不建议直接合并main。提交前应明确选择代码、依赖、README、测试和本报告，不要把 `.venv-ppo`、临时目录或本次Smoke二进制模型纳入源码提交。实际提交和推送留给用户，本次未执行。
