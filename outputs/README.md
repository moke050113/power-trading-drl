# 小型示例结果与测试记录

以下文件纳入版本管理，数据均来自固定合成教学曲线，不是训练成果：

| 文件 | 保留原因 |
| --- | --- |
| `strategy_comparison/daily_costs.csv` | 三种基础策略全天成本表 |
| `strategy_comparison/hourly_details.csv` | 72行逐小时电量与成本明细 |
| `strategy_comparison/strategy_comparison.svg` | 可复现的小型曲线图 |
| `validation_2026-10-11/test_report.md` | 正式A/B/C联合测试报告及模型限制 |
| `validation_2026-10-11/baseline_evaluation.json` | 指定种子下基准成本与Reward统计 |
| `validation_2026-10-11/ppo_evaluation.json` | 96步Smoke之后的原始成本评估记录 |
| `validation_2026-10-11/ppo_smoke/smoke_report.json` | 训练步数、依赖版本和保存加载验证记录 |

本目录说明也随Git保留。上述现有结果文件均小于20 KB。

忽略且保留在本地：

- `baseline_evaluation.json`：与日期目录中的基准结果重复。
- `ppo_smoke/`：早期未归档的Smoke运行输出。
- 日期目录下的 `ppo_smoke/ppo_power_trading.zip`、`reward_normalization.pkl`：模型及归一化状态二进制。
- 其他未列入白名单的输出、临时测试目录和未来实验文件。

`.gitignore` 已实现白名单及模型后缀忽略规则；不会删除本地文件。要新增正式结果，应先审核大小、敏感信息、数据来源和复现说明，再更新白名单。报告中的本机绝对路径是测试记录；源码使用项目相对路径或脚本位置，没有依赖这些本机路径。

生成策略图表：在项目根目录使用Python 3.11运行 `python compare_strategies.py`。模型二进制不在仓库中，若要重新评估PPO模型，先依README安装可选依赖并运行训练入口。随机或训练结果可能受依赖版本和硬件影响。
