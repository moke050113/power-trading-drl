"""在相同合成曲线上比较三种基础策略；不包含训练。"""
import argparse
import csv
from html import escape
from pathlib import Path

from env.power_trading_env import PowerTradingEnv


STRATEGIES = (
    ('spot', '全现货购电'),
    ('fixed70', '固定70%合同覆盖率'),
    ('cheapest', '逐小时低价市场（简化模型参考策略）'),
)
COLORS = ('#2563eb', '#d97706', '#059669')


def compare_strategies(load_profile=None, spot_price_profile=None, long_term_price=400.0):
    """返回每种策略的逐小时结算；所有策略共用同一份输入曲线。"""
    source = PowerTradingEnv(load_profile=load_profile,
                             spot_price_profile=spot_price_profile,
                             long_term_price=long_term_price)
    results = {}
    for key, _ in STRATEGIES:
        env = PowerTradingEnv(load_profile=source.load_profile,
                              spot_price_profile=source.spot_price_profile,
                              long_term_price=source.long_term_price)
        env.reset()
        rows = []
        for hour in range(env.horizon):
            # 使用原始曲线精度结算，避免观察值的float32转换影响比较。
            load = float(source.load_profile[hour])
            if key == 'spot':
                quantity = 0.0
            elif key == 'fixed70':
                quantity = load * 0.7
            else:
                quantity = load if source.long_term_price < source.spot_price_profile[hour] else 0.0
            _, _, terminated, truncated, info = env.step([quantity])
            rows.append(info)
            if terminated or truncated:
                break
        results[key] = rows
    return results


def write_csv(results, directory):
    """分别保存全天汇总和逐小时明细，中文CSV可由Excel打开。"""
    with (directory / 'daily_costs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['策略', '全天累计成本（元）', '相比全现货节省（元）'])
        baseline = results['spot'][-1]['episode_cost']
        for key, label in STRATEGIES:
            cost = results[key][-1]['episode_cost']
            writer.writerow([label, cost, baseline - cost])
    with (directory / 'hourly_details.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['策略', '小时', '负荷（MWh）', '合同电量（MWh）', '现货电量（MWh）',
                         '合同价格（元/MWh）', '现货价格（元/MWh）', '本小时成本（元）', '累计成本（元）'])
        for key, label in STRATEGIES:
            for row in results[key]:
                writer.writerow([label] + [row[field] for field in (
                    'hour', 'load', 'long_term_quantity', 'spot_quantity',
                    'long_term_price', 'spot_price', 'total_cost', 'episode_cost')])


def write_chart(results, path):
    """用标准库生成独立SVG图，无需额外绘图依赖。"""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1310" viewBox="0 0 1200 1310">',
             '<rect width="1200" height="1310" fill="#f8fafc"/>',
             '<g font-family="Microsoft YaHei, SimHei, sans-serif" fill="#0f172a">']

    def text(x, y, value, size=16, **attrs):
        attributes = ' '.join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{escape(str(value))}</text>')

    text(50, 45, '24小时基础购电策略比较', 28, font_weight='bold')
    text(50, 76, '固定合成数据 · 三种策略使用相同负荷和价格曲线 · 未进行DRL训练')
    text(50, 105, '参考策略直接观察当小时价格并逐小时选市场；同价时选现货，不代表真实合同可逐小时重签。', 15)
    baseline = results['spot'][-1]['episode_cost']
    for index, (key, label) in enumerate(STRATEGIES):
        cost = results[key][-1]['episode_cost']
        text(50, 143 + index * 29, f'{label}：全天 {cost:,.2f} 元；相比全现货节省 {baseline - cost:,.2f} 元',
             fill=COLORS[index])

    def panel(top, title, series, unit):
        left, right, height = 100, 1150, 145
        plot_top = top + 58
        bottom = plot_top + height
        maximum = max(max(values) for _, _, values in series)
        maximum = max(1.0, maximum * 1.1)
        text(50, top + 20, title, 20, font_weight='bold')
        for index, (label, color, _) in enumerate(series):
            x = 100 + index * 340
            parts.append(f'<line x1="{x}" y1="{top + 40}" x2="{x + 25}" y2="{top + 40}" stroke="{color}" stroke-width="3"/>')
            text(x + 32, top + 45, label, 14)
        for tick in range(5):
            value = maximum * tick / 4
            y = bottom - height * tick / 4
            parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#cbd5e1"/>')
            text(left - 12, y + 5, f'{value:.0f}', 13, text_anchor='end')
        for hour in (0, 3, 6, 9, 12, 15, 18, 21, 23):
            x = left + (right - left) * hour / 23
            text(x, bottom + 21, str(hour), 13, text_anchor='middle')
        text(left, plot_top - 5, unit, 12)
        text(right, bottom + 40, '小时（0～23）', 13, text_anchor='end')
        for _, color, values in series:
            points = ' '.join(f'{left + (right - left) * hour / 23:.2f},{bottom - height * value / maximum:.2f}'
                              for hour, value in enumerate(values))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')

    rows = results['spot']
    panel(235, '负荷曲线（三种策略共用）', [('负荷', '#475569', [row['load'] for row in rows])], 'MWh')
    labels = ('全现货', '固定70%合同', '逐小时低价（参考）')
    for top, field, title in ((495, 'long_term_quantity', '合同电量曲线'),
                              (755, 'spot_quantity', '现货电量曲线')):
        panel(top, title, [(labels[index], COLORS[index], [row[field] for row in results[key]])
                           for index, (key, _) in enumerate(STRATEGIES)], 'MWh')
    panel(1015, '价格曲线（三种策略共用）', [
        ('合同价格', '#7c3aed', [row['long_term_price'] for row in rows]),
        ('现货价格', '#e11d48', [row['spot_price'] for row in rows])], '元/MWh')
    text(50, 1290, '全天成本为24个小时成本之和；各小时合同电量 + 现货电量 = 负荷。', 14)
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'strategy_comparison',
                        help='图表和CSV保存目录')
    args = parser.parse_args()
    results = compare_strategies()
    print('相同24小时合成曲线；成本单位：元；未进行训练。')
    baseline = results['spot'][-1]['episode_cost']
    for key, label in STRATEGIES:
        cost = results[key][-1]['episode_cost']
        print(f'{label} | 全天成本 {cost:.2f} | 相比全现货节省 {baseline - cost:.2f}')
    print('参考策略直接使用当小时价格，可逐小时选市场；同价时选现货。现实合同约束尚未建模。')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, args.output_dir)
    write_chart(results, args.output_dir / 'strategy_comparison.svg')
    print(f'图表与CSV已保存至：{args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
