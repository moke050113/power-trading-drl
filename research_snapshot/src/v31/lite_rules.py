"""Three frozen rules. No optimization against realized paths or evaluation."""
import numpy as np
from v3.baselines import fixed_contract,rolling_contract,zero_dispatch,forecast_dispatch

RULES={'B0':(fixed_contract,zero_dispatch),'B1':(rolling_contract,zero_dispatch),'B2':(rolling_contract,forecast_dispatch)}
PROTOCOL=dict(action_order=['active_spot_ratio','EV_power_ratio','industrial_power_ratio'],
    fixed_contract=[.3,.5,.7],rolling_contract='WAIT/SKIP if quote >= forecast_balancing_mean; otherwise Y .3, M/I .25 remaining',
    dispatch='original v3.baselines.forecast_dispatch, generic price explicitly mapped to historical balancing forecast',
    tie='original deterministic time/segment sort; quote equal means no buy',unknown_demand='soft only; fixed actual tasks')
