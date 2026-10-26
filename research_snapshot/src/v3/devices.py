"""每日到站会话桥接，复用冻结EV/工业语义；不把EV电池电量重复计入负荷。"""
from copy import deepcopy
import math
from load_models.ev_fleet import EVFleetExecutor, EVFleetConfig
from load_models.industrial_load import IndustrialLoad
from v3.core import vector,number


class DailyDevices:
    def __init__(self,limited=False):
        self.limited=bool(limited)
        self.ev=EVFleetExecutor(EVFleetConfig(start_hour=19 if limited else 18));self.ev.reset()
        self.industrial=IndustrialLoad(start_hour=12 if limited else 8)
        self.hour=0
        self.ev_total=self.ind_total=self.ev_forced=self.ind_forced=0.

    def state(self):
        e=self.ev.state_view();i=self.industrial;h=self.hour
        imin=max(0.,i.remaining_energy_mwh-(i.end_hour-h-1)*i.max_power_mw) if i.start_hour<=h<i.end_hour else 0.
        return dict(hour=h,ev_soc=e['soc'],ev_remaining=e['remaining_grid_mwh'],ev_min=e['minimum_power_mw'],
                    ev_start=e['start_hour'],ev_end=e['end_hour'],industrial_remaining=i.remaining_energy_mwh,
                    industrial_min=imin,industrial_start=i.start_hour,industrial_end=i.end_hour)

    def prepare(self,ev_request,industrial_request):
        er=number(ev_request,0,7);ir=number(industrial_request,0,4)
        if self.hour>=24 or self.ev.hour!=self.hour or self.industrial.hour!=self.hour:raise RuntimeError('设备时钟错位')
        candidate=deepcopy(self);h=self.hour
        i=candidate.industrial.step(h,ir)
        e=candidate.ev.step(h,er)
        v=self.state()
        expected_i=min(4,self.industrial.remaining_energy_mwh,max(ir,v['industrial_min'])) if v['industrial_start']<=h<v['industrial_end'] else 0.
        actual_i=number(i['load_mwh'],0,4);actual_e=number(e['grid_energy_mwh'],0,7)
        expected_e=min(7,v['ev_remaining'],max(er,v['ev_min'])) if v['ev_start']<=h<v['ev_end'] else 0.
        if (candidate.industrial.hour!=h+1 or candidate.ev.hour!=h+1 or
            not math.isclose(actual_i,expected_i,abs_tol=1e-10) or
            not math.isclose(candidate.industrial.remaining_energy_mwh,self.industrial.remaining_energy_mwh-actual_i,abs_tol=1e-10) or
            not math.isclose(actual_e*.9,e['battery_energy_mwh'],abs_tol=1e-10) or
            not math.isclose(actual_e,expected_e,abs_tol=1e-10) or
            not math.isclose(candidate.ev.soc,e['soc'],abs_tol=1e-10) or
            not math.isclose(candidate.ev.soc,v['ev_soc']+.9*actual_e/60,abs_tol=1e-10) or
            not math.isclose(e['remaining_grid_mwh'],v['ev_remaining']-actual_e,abs_tol=1e-10) or
            not math.isclose(e['forced_charging_mwh'],max(0.,actual_e-er),abs_tol=1e-10)):
            raise RuntimeError('设备输出破坏守恒')
        forced_i=max(0.,actual_i-ir)
        # 偏好是合法事前均匀曲线，不取事后最优；半L1小时成本可加。
        nominal_e=(100/3)/(24-v['ev_start']) if v['ev_start']<=h<24 else 0.
        nominal_i=12/(16-v['industrial_start']) if v['industrial_start']<=h<16 else 0.
        flex_cost=(abs(actual_e-nominal_e)+abs(actual_i-nominal_i)) # κ=2, 半L1。
        candidate.ev_total+=actual_e;candidate.ind_total+=actual_i
        candidate.ev_forced+=e['forced_charging_mwh'];candidate.ind_forced+=forced_i
        candidate.hour+=1
        if h==23:
            if not math.isclose(candidate.ev_total,100/3,abs_tol=1e-8) or not math.isclose(candidate.ind_total,12,abs_tol=1e-8) or not math.isclose(candidate.ev.soc,.7,abs_tol=1e-10):
                raise RuntimeError('日任务未完成')
        return candidate,dict(ev_request_mw=er,industrial_request_mw=ir,
                              ev_actual_mw=actual_e,industrial_actual_mw=actual_i,
                              ev_forced_mwh=e['forced_charging_mwh'],industrial_forced_mwh=forced_i,
                              ev_soc=e['soc'],ev_battery_mwh=e['battery_energy_mwh'],
                              ev_remaining=candidate.ev.state_view()['remaining_grid_mwh'],
                              industrial_remaining=candidate.industrial.remaining_energy_mwh,flex_cost=flex_cost)
