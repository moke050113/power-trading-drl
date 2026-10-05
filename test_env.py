from env.power_trading_env import PowerTradingEnv


env = PowerTradingEnv()

state, info = env.reset()

print("初始状态：")
print(state)

action = [70.0]

next_state, reward, terminated, truncated, info = env.step(action)

print()
print("交易结果：")
print("中长期购买量：", info["long_term_quantity"])
print("现货购买量：", info["spot_quantity"])
print("中长期成本：", info["long_term_cost"])
print("现货成本：", info["spot_cost"])
print("总成本：", info["total_cost"])
print("Reward：", reward)