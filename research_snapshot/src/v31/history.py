"""Append-only archive, deterministic as-of projection and trusted persistence."""
from pathlib import Path
import json
from .schema import HistoryRecord, InputRef, canonical, digest, instant, sha


def event(operation, at, hashes):
    return canonical({'operation': operation, 'at': instant(at).isoformat(),
                      'content_sha256': tuple(hashes)})


def save_new(path, kind, records, audit):
    payload = {'schema_version': 1, 'kind': kind, 'records': [
        {'record': r.to_dict(), 'sha256': r.content_sha256} for r in records],
        'audit': [json.loads(x) for x in audit]}
    envelope = {'payload': payload, 'sha256': digest(payload)}
    # Encode/validate completely before exclusive creation; never overwrite a result.
    data = canonical(envelope)
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(data+'\n')
    return envelope['sha256']


def load_records(path, kind, cls):
    envelope = json.loads(Path(path).read_text(encoding='utf-8'))
    p = envelope['payload']
    if envelope['sha256'] != digest(p) or p['kind'] != kind or p['schema_version'] != 1:
        raise ValueError('invalid archive envelope/hash')
    records = []
    for row in p['records']:
        record = cls.from_dict(row['record'])
        if record.content_sha256 != row['sha256']:
            raise ValueError('record content hash mismatch')
        records.append(record)
    allowed = {'history_append', 'as_of_replay'} if kind == 'history' else {'forecast_publish', 'forecast_query'}
    audit = []
    for row in p['audit']:
        if set(row) != {'operation', 'at', 'content_sha256'} or row['operation'] not in allowed:
            raise ValueError('invalid audit operation')
        instant(row['at'])
        for value in row['content_sha256']:
            sha(value)
        audit.append(canonical(row))
    return tuple(records), tuple(audit)


class HistoryArchive:
    """Trusted owner API. Give policies only AsOfReplay results, never this object."""
    def __init__(self):
        self._records = {}
        self._audit = ()

    @property
    def audit(self):
        return tuple(json.loads(x) for x in self._audit)

    @property
    def content_sha256(self):
        return digest(sorted((k, v.content_sha256) for k, v in self._records.items()))

    def append(self, record):
        self.append_many((record,))

    def append_many(self, records):
        candidate = dict(self._records)
        events = []
        for r in tuple(records):
            if type(r) is not HistoryRecord or r.record_id in candidate:
                raise ValueError('invalid/duplicate history record')
            older = [v for v in candidate.values() if v.series_id == r.series_id]
            if not older:
                if r.revision != 0 or r.supersedes is not None:
                    raise ValueError('first release must be revision zero')
            else:
                old = max(older, key=lambda v: v.revision)
                identity = ('record_type', 'target_variable', 'grid', 'unit', 'source')
                if (r.revision != old.revision+1 or r.supersedes != old.record_id or
                        any(getattr(r, k) != getattr(old, k) for k in identity) or
                        instant(r.published_at) < instant(old.published_at) or
                        instant(r.available_at) < instant(old.available_at)):
                    raise ValueError('revision must extend the same series, without backdating')
            candidate[r.record_id] = r
            events.append(event('history_append', r.available_at, (r.content_sha256,)))
        self._records, self._audit = candidate, self._audit+tuple(events)

    def resolve(self, ref, as_of):
        """Time-gated lineage resolution, including exact revision SHA."""
        if type(ref) is not InputRef:
            raise ValueError('explicit input reference required')
        at = instant(as_of)
        r = self._records.get(ref.record_id)
        if r is None or instant(r.available_at) > at or r.content_sha256 != ref.sha256:
            raise ValueError('input missing, unavailable at cutoff or hash mismatch')
        return HistoryRecord.from_dict(r.to_dict())

    def snapshot(self, as_of, *, fields=None):
        at = instant(as_of)
        if fields is not None:
            from .schema import UNITS
            fields = tuple(fields)
            if any(f not in UNITS for f in fields):
                raise ValueError('unknown snapshot field')
        selected = {}
        for r in self._records.values():
            if instant(r.available_at) <= at and (fields is None or r.target_variable in fields):
                old = selected.get(r.series_id)
                if old is None or r.revision > old.revision:
                    selected[r.series_id] = r
        rows = tuple(sorted(selected.values(), key=lambda r: (r.available_at, r.series_id, r.revision, r.record_id)))
        self._audit += (event('as_of_replay', as_of, tuple(r.content_sha256 for r in rows)),)
        return tuple(HistoryRecord.from_dict(r.to_dict()) for r in rows)

    def save(self, path):
        """Administrative export includes all vintages; NOT a policy read interface."""
        rows = sorted(self._records.values(), key=lambda r: (r.series_id, r.revision))
        return save_new(path, 'history', rows, self._audit)

    @classmethod
    def load(cls, path):
        obj = cls()
        rows, audit = load_records(path, 'history', HistoryRecord)
        obj.append_many(rows)
        obj._audit = audit
        return obj


class AsOfReplay:
    def __init__(self, archive):
        if type(archive) is not HistoryArchive:
            raise ValueError('HistoryArchive required')
        self._archive = archive

    def at(self, as_of, *, fields=None):
        return self._archive.snapshot(as_of, fields=fields)
