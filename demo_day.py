"""固定70%合同覆盖率示例，输出24小时结算和整天成本。"""
import argparse
import csv
from pathlib import Path
from env.power_trading_env import PowerTradingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='可选：保存24小时结算表')
    args = parser.parse_args()
    env = PowerTradingEnv()
    state, _ = env.reset()
    rows = []
    for _ in range(env.horizon):
        state, reward, terminated, truncated, info = env.step([float(state[0]) * 0.7])
        rows.append(info)
        print(f"小时{info['hour']:02d} | 负荷 {info['load']:.1f} MWh | "
              f"合同 {info['long_term_quantity']:.1f} | 现货 {info['spot_quantity']:.1f} | "
              f"成本 {info['total_cost']:.2f} 元")
        if terminated or truncated:
            break
    print(f"全天成本：{rows[-1]['episode_cost']:.2f} 元")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"已保存：{args.csv.resolve()}")


if __name__ == '__main__':
    main()
