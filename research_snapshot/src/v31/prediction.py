"""Two deterministic historical baselines; no market loaders or model training."""
from dataclasses import dataclass, asdict
from datetime import timedelta
import json
import math

from .schema import (HourGrid, ForecastRecord, ModelProvenance, InputRef,
                     UNITS, canonical, digest, instant, stamp, text)
from .history import HistoryArchive

TARGETS = ('base_load', 'offer_price', 'balancing_price')
METHODS = ('seasonal_last', 'weekday_mean')


@dataclass(frozen=True)
class BaselineConfig:
    method: str
    history_source: str
    training_split_end: str
    window_days: int = 28

    def __post_init__(self):
        if self.method not in METHODS or type(self.window_days) is not int or self.window_days < 7 or self.window_days % 7:
            raise ValueError('method requires a whole number of weeks (at least seven days)')
        text(self.history_source)
        end = instant(self.training_split_end)
        if end.hour or end.minute or end.second or end.microsecond:
            raise ValueError('training split end must be a midnight boundary')
        object.__setattr__(self, 'training_split_end', end.isoformat())

    @property
    def content_sha256(self):
        return digest(asdict(self))


def _observations(snapshot, target, source):
    """Public quotes and other targets NEVER become realized training labels."""
    return tuple(r for r in snapshot if r.record_type == 'observation' and
                 r.target_variable == target and r.source == source)


class ForecastProvider:
    """Trusted builder: obtains its own P1 snapshot, never accepts unverified values."""
    def __init__(self, archive, config):
        if type(archive) is not HistoryArchive or type(config) is not BaselineConfig:
            raise ValueError('P1 HistoryArchive and frozen BaselineConfig required')
        self._archive = archive
        self.config = config
        self._audit = ()

    @property
    def audit(self):
        return tuple(json.loads(row) for row in self._audit)

    def issue(self, *, as_of, published_at, scope, target_variable, target_interval,
              forecast_id, vintage=0, valid_until=None, supersedes=None):
        a, p = instant(as_of), instant(published_at)
        if target_variable not in TARGETS or type(target_interval) is not HourGrid:
            raise ValueError('explicit supported target/grid required')
        target_interval.validate_scope(scope)
        expiry = target_interval.start if valid_until is None else instant(valid_until)
        if not a <= p < expiry <= target_interval.start:
            raise ValueError('illegal forecast timing')
        # Last fully completed calendar day, never a partial current-day profile.
        end = min(a.replace(hour=0, minute=0, second=0, microsecond=0),
                  instant(self.config.training_split_end))
        start = end-timedelta(days=self.config.window_days)
        snapshot = self._archive.snapshot(a.isoformat(), fields=(target_variable,))
        points = {}
        for r in _observations(snapshot, target_variable, self.config.history_source):
            # Whole referenced observations must be completed within the fit cutoff.
            if r.grid.end > end:
                continue
            for h, value in zip(r.grid.hours, r.values):
                t = instant(h)
                if start <= t < end:
                    if t in points:
                        raise ValueError('ambiguous overlapping historical series/hour')
                    points[t] = (value, r)
        expected = tuple(start+timedelta(hours=i) for i in range(self.config.window_days*24))
        if set(points) != set(expected):
            raise LookupError('insufficient complete legally visible history; no fallback')
        groups = {}
        for t in expected:
            groups.setdefault((t.weekday(), t.hour), []).append((t, *points[t]))
        selected = {}
        profile = {}
        for key, rows in groups.items():
            used = rows[-1:] if self.config.method == 'seasonal_last' else rows
            # Divide before summation to avoid overflow for finite extreme prices.
            value = math.fsum(row[1]/len(used) for row in used)
            if not math.isfinite(value):
                raise ValueError('nonfinite historical profile')
            profile[key] = value
            for t, _, r in used:
                selected[t] = r
        used_records = {r.record_id: r for r in selected.values()}
        refs = tuple(InputRef(k, r.content_sha256) for k, r in sorted(used_records.items()))
        # Dense-window validation also consumed the presence/availability of other rows.
        # Declare these eligibility dependencies separately in frozen configuration.
        eligibility_records = {r.record_id: r for _, r in points.values()}
        eligibility = tuple(InputRef(k, r.content_sha256) for k, r in sorted(eligibility_records.items()))
        configuration = {
            'algorithm_version': 'p2_historical_v1', **asdict(self.config),
            'target_variable': target_variable, 'window_start': start.isoformat(),
            'window_end': end.isoformat(), 'missing_policy': 'reject',
            'calendar_key': 'weekday_hour', 'eligibility_inputs': [asdict(x) for x in eligibility],
            'selected_hours': [t.isoformat() for t in sorted(selected)],
        }
        # Fit here means deterministic profile construction, not learned network fitting.
        labels_end = max(r.grid.end for r in eligibility_records.values())
        available = max(instant(r.available_at) for r in eligibility_records.values())
        model = ModelProvenance(
            model_version='p2_historical_v1/'+self.config.method,
            config_json=canonical(configuration), config_sha256=digest(configuration),
            fit_as_of=a.isoformat(), fit_completed_at=a.isoformat(),
            training_label_end=labels_end.isoformat(), training_labels_available_at=available.isoformat(),
            training_split_end=self.config.training_split_end, training_inputs=eligibility)
        output = tuple(profile[(instant(t).weekday(), instant(t).hour)] for t in target_interval.hours)
        record = ForecastRecord(
            forecast_id=forecast_id, vintage=vintage, scope=scope, target_variable=target_variable,
            grid=target_interval, as_of=a.isoformat(), published_at=p.isoformat(), valid_until=expiry.isoformat(),
            source='v31.ForecastProvider/p2_historical_v1', input_vintages=eligibility,
            model=model, values=output, unit=UNITS[target_variable], supersedes=supersedes)
        self._audit += (canonical({
            'operation': 'historical_prediction', 'as_of': a.isoformat(),
            'forecast_sha256': record.content_sha256, 'frozen_config_sha256': self.config.content_sha256,
            'input_vintages': [asdict(x) for x in eligibility],
            'value_inputs': [asdict(x) for x in refs],
            'selected_hours_sha256': digest(configuration['selected_hours']),
        }),)
        return record
