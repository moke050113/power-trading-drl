"""Immutable vintages with validated lineage; missing legal data always rejects."""
import json
from .schema import ForecastRecord, HourGrid, UNITS, digest, instant
from .history import HistoryArchive, event, save_new, load_records


class ForecastRepository:
    def __init__(self, archive):
        if type(archive) is not HistoryArchive:
            raise ValueError('HistoryArchive required')
        self._archive = archive
        self._records = {}
        self._audit = ()

    @property
    def audit(self):
        return tuple(json.loads(v) for v in self._audit)

    @property
    def content_sha256(self):
        return digest(sorted((k, r.content_sha256) for k, r in self._records.items()))

    def _validate_lineage(self, r):
        for ref in r.input_vintages:
            self._archive.resolve(ref, r.as_of)
        m = r.model
        for ref in m.training_inputs:
            h = self._archive.resolve(ref, m.fit_as_of)
            if (h.record_type != 'observation' or h.grid.end > instant(m.training_label_end) or
                    instant(h.available_at) > instant(m.training_labels_available_at)):
                raise ValueError('training label lineage exceeds declared availability/end')

    def publish(self, record):
        self.publish_many((record,))

    def publish_many(self, records):
        candidate = dict(self._records)
        events = []
        for r in tuple(records):
            if type(r) is not ForecastRecord or r.forecast_id in candidate:
                raise ValueError('invalid/duplicate forecast')
            self._validate_lineage(r)
            key = lambda v: (v.scope, v.target_variable, v.grid)
            siblings = [v for v in candidate.values() if key(v) == key(r)]
            if any(v.vintage == r.vintage for v in siblings):
                raise ValueError('duplicate vintage for target/scope')
            if r.supersedes is not None:
                old = candidate.get(r.supersedes)
                if (old is None or key(old) != key(r) or old.vintage >= r.vintage or
                        instant(old.published_at) > instant(r.published_at) or
                        instant(old.as_of) > instant(r.as_of)):
                    raise ValueError('invalid replacement relation')
            candidate[r.forecast_id] = r
            events.append(event('forecast_publish', r.published_at,
                                (r.content_sha256, *(v.sha256 for v in r.input_vintages),
                                 r.model.config_sha256)))
        self._records, self._audit = candidate, self._audit+tuple(events)

    def get_published(self, *, decision_at, scope, target_variable, target_interval):
        at = instant(decision_at)
        if type(target_interval) is not HourGrid or target_variable not in UNITS:
            raise ValueError('explicit target interval/variable required')
        target_interval.validate_scope(scope)
        if at >= target_interval.start:
            raise ValueError('forecast decision must precede delivery')
        released = [r for r in self._records.values() if instant(r.published_at) <= at]
        superseded = {r.supersedes for r in released if r.supersedes is not None}
        rows = [r for r in released if r.forecast_id not in superseded and
                r.scope == scope and r.target_variable == target_variable and
                r.grid == target_interval and at < instant(r.valid_until)]
        if not rows:
            raise LookupError('no legally published, unexpired forecast for this scope/target')
        r = max(rows, key=lambda v: (v.published_at, v.vintage, v.forecast_id))
        self._validate_lineage(r)
        self._audit += (event('forecast_query', decision_at, (r.content_sha256,)),)
        return ForecastRecord.from_dict(r.to_dict())

    def save(self, path):
        rows = sorted(self._records.values(), key=lambda r: (r.published_at, r.vintage, r.forecast_id))
        return save_new(path, 'forecasts', rows, self._audit)

    @classmethod
    def load(cls, path, archive):
        obj = cls(archive)
        rows, audit = load_records(path, 'forecasts', ForecastRecord)
        obj.publish_many(rows)
        obj._audit = audit
        return obj
