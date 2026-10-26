from env.power_trading_env import PowerTradingEnv


env = PowerTradingEnv()

actions = [0, 20, 40, 60, 80, 100]

print("中长期购买量 | 现货购买量 | 总成本")
print("--------------------------------")

for action in actions:
    env.reset()

    _, _, _, _, info = env.step([action])

    print(
        f"{info['long_term_quantity']:>10.0f} MWh | "
        f"{info['spot_quantity']:>9.0f} MWh | "
        f"{info['total_cost']:>8.0f} 元"
    )