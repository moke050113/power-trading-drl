import math
from numbers import Integral


def finite(value, name, minimum=0.0, maximum=math.inf):
    if isinstance(value, bool):
        raise ValueError(f'{name}必须是有限数值')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name}必须是有限数值') from None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f'{name}必须在[{minimum}, {maximum}]内')
    return result


def window(start_hour, end_hour):
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (start_hour, end_hour)):
        raise ValueError('时间窗口必须使用整数小时')
    if not 0 <= start_hour < end_hour <= 24:
        raise ValueError('要求0 <= start_hour < end_hour <= 24')


def check_hour(hour, expected):
    if isinstance(hour, bool) or not isinstance(hour, Integral) or hour != expected or not 0 <= hour < 24:
        raise ValueError('请从小时0开始按顺序调用step；24小时结束后先reset')
