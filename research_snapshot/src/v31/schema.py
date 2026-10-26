"""Immutable hourly records and strict, dependency-free information validators."""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import calendar
import hashlib
import json
import math

TZ = timezone(timedelta(hours=8))
UNITS = {'base_load': 'MWh', 'offer_price': 'CNY/MWh',
         'balancing_price': 'CNY/MWh', 'contract_quote': 'CNY/MWh'}
SCOPES = ('Y', 'M', 'I', 'DA')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('expected lowercase SHA256')
    return value


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected nonempty text')
    return value


def instant(value):
    """Require explicit UTC+08, not naive timestamps or silent timezone conversion."""
    if not isinstance(value, str):
        raise ValueError('timestamp must be ISO text')
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError('invalid ISO timestamp') from None
    if dt.utcoffset() != timedelta(hours=8):
        raise ValueError('explicit UTC+08 timestamp required')
    return dt


def stamp(value):
    return instant(value).isoformat()


@dataclass(frozen=True)
class HourGrid:
    """Explicit timestamps make duplicate/gap detection independent of values."""
    hours: tuple

    def __post_init__(self):
        hours = tuple(stamp(x) for x in self.hours)
        if not hours:
            raise ValueError('empty grid')
        ds = tuple(instant(x) for x in hours)
        if any(x.minute or x.second or x.microsecond for x in ds):
            raise ValueError('grid must align to hourly boundaries')
        if any(b-a != timedelta(hours=1) for a, b in zip(ds, ds[1:])):
            raise ValueError('duplicate or discontinuous hour grid')
        object.__setattr__(self, 'hours', hours)

    @classmethod
    def between(cls, start, end):
        a, z = instant(start), instant(end)
        n = (z-a).total_seconds()/3600
        if not 0 < n <= 8784 or not n.is_integer():
            raise ValueError('invalid hourly interval (maximum one leap year)')
        return cls(tuple((a+timedelta(hours=i)).isoformat() for i in range(int(n))))

    @property
    def start(self):
        return instant(self.hours[0])

    @property
    def end(self):
        return instant(self.hours[-1])+timedelta(hours=1)

    def validate_scope(self, scope):
        if scope not in SCOPES:
            raise ValueError('unknown forecast scope')
        a, z = self.start, self.end
        if a.hour:
            raise ValueError('scope must begin at midnight')
        if scope == 'Y':
            expected = datetime(a.year+1, 1, 1, tzinfo=TZ)
            valid = a.month == a.day == 1
        elif scope == 'M':
            expected = a.replace(day=1)+timedelta(days=calendar.monthrange(a.year, a.month)[1])
            valid = a.day == 1
        elif scope == 'I':
            days = calendar.monthrange(a.year, a.month)[1]
            end_day = {1: 11, 11: 21, 21: days+1}.get(a.day)
            valid = end_day is not None
            expected = a+timedelta(days=(end_day-a.day) if valid else 0)
        else:
            expected = a+timedelta(days=1)
            valid = True
        if not valid or z != expected:
            raise ValueError('grid does not match calendar scope')


def values_tuple(values, grid, variable, unit):
    if UNITS.get(variable) != unit:
        raise ValueError('unknown target or incompatible unit')
    raw = tuple(values)
    if len(raw) != len(grid.hours):
        raise ValueError('values/grid dimension mismatch')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
        raise ValueError('numeric scalar values required')
    result = tuple(float(v) for v in raw)
    if any(not math.isfinite(v) for v in result):
        raise ValueError('nonfinite data')
    if variable == 'base_load' and any(v < 0 for v in result):
        raise ValueError('negative base load')
    # Prices explicitly accept finite negatives, WITHOUT changing any V3 settlement.
    return result


def refs_tuple(refs):
    refs = tuple(refs)
    if any(type(r) is not InputRef for r in refs) or len({r.record_id for r in refs}) != len(refs):
        raise ValueError('invalid/duplicate input references')
    return tuple(sorted(refs, key=lambda r: r.record_id))


class ContentRecord:
    def to_dict(self):
        """New nested containers; modifying this detached export cannot alter records."""
        return asdict(self)

    @property
    def content_sha256(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class InputRef:
    record_id: str
    sha256: str

    def __post_init__(self):
        text(self.record_id)
        sha(self.sha256)


@dataclass(frozen=True)
class HistoryRecord(ContentRecord):
    record_id: str
    series_id: str
    revision: int
    record_type: str
    target_variable: str
    grid: HourGrid
    published_at: str
    available_at: str
    source: str
    source_sha256: str
    values: tuple
    unit: str
    supersedes: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported schema')
        if type(self.grid) is not HourGrid or type(self.revision) is not int or self.revision < 0:
            raise ValueError('invalid grid/revision')
        for x in (self.record_id, self.series_id, self.source):
            text(x)
        sha(self.source_sha256)
        if self.record_type not in ('observation', 'public_information'):
            raise ValueError('unknown history record type')
        if self.record_type == 'public_information' and self.target_variable not in ('offer_price', 'contract_quote'):
            raise ValueError('P1 public information supports published quotes only')
        p, a = instant(self.published_at), instant(self.available_at)
        if a < p or (self.record_type == 'observation' and p < self.grid.end):
            raise ValueError('observation unavailable before measurement completion')
        object.__setattr__(self, 'published_at', p.isoformat())
        object.__setattr__(self, 'available_at', a.isoformat())
        object.__setattr__(self, 'values', values_tuple(self.values, self.grid, self.target_variable, self.unit))
        if self.supersedes is not None:
            text(self.supersedes)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['grid'] = HourGrid(**data['grid'])
        return cls(**data)


@dataclass(frozen=True)
class ModelProvenance(ContentRecord):
    model_version: str
    config_json: str
    config_sha256: str
    fit_as_of: str
    fit_completed_at: str
    training_label_end: str
    training_labels_available_at: str
    training_split_end: str
    training_inputs: tuple

    def __post_init__(self):
        text(self.model_version)
        config = canonical(json.loads(self.config_json))
        if digest(json.loads(config)) != sha(self.config_sha256):
            raise ValueError('configuration hash mismatch')
        object.__setattr__(self, 'config_json', config)
        times = {k: instant(getattr(self, k)) for k in
                 ('fit_as_of', 'fit_completed_at', 'training_label_end', 'training_labels_available_at', 'training_split_end')}
        if not (times['training_label_end'] <= times['training_labels_available_at'] <=
                times['fit_as_of'] <= times['fit_completed_at']):
            raise ValueError('training labels not complete/observable at fit cutoff')
        if times['training_label_end'] > times['training_split_end']:
            raise ValueError('training label target crosses frozen split boundary')
        for k, v in times.items():
            object.__setattr__(self, k, v.isoformat())
        object.__setattr__(self, 'training_inputs', refs_tuple(self.training_inputs))
        if not self.training_inputs:
            raise ValueError('explicit training lineage required (P1 has no untracked model mode)')

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['training_inputs'] = tuple(InputRef(**v) for v in data['training_inputs'])
        return cls(**data)


@dataclass(frozen=True)
class ForecastRecord(ContentRecord):
    forecast_id: str
    vintage: int
    scope: str
    target_variable: str
    grid: HourGrid
    as_of: str
    published_at: str
    valid_until: str
    source: str
    input_vintages: tuple
    model: ModelProvenance
    values: tuple
    unit: str
    supersedes: str | None = None
    uncertainty: tuple | None = None
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported schema')
        if type(self.vintage) is not int or self.vintage < 0 or type(self.grid) is not HourGrid:
            raise ValueError('invalid vintage/grid')
        self.grid.validate_scope(self.scope)
        for x in (self.forecast_id, self.source):
            text(x)
        a, p, z = map(instant, (self.as_of, self.published_at, self.valid_until))
        if not a <= p < z <= self.grid.start:
            raise ValueError('forecast cutoff/publication/expiry must precede target delivery')
        if type(self.model) is not ModelProvenance or instant(self.model.fit_completed_at) > a:
            raise ValueError('model fit occurs after forecast cutoff')
        for k, v in [('as_of', a), ('published_at', p), ('valid_until', z)]:
            object.__setattr__(self, k, v.isoformat())
        object.__setattr__(self, 'input_vintages', refs_tuple(self.input_vintages))
        if not self.input_vintages:
            raise ValueError('explicit prediction input lineage required')
        object.__setattr__(self, 'values', values_tuple(self.values, self.grid, self.target_variable, self.unit))
        if self.supersedes is not None:
            text(self.supersedes)
        if self.uncertainty is not None:
            # P1 supports only explicit lower/upper arrays, no unspecified nested mutable payload.
            u = tuple(tuple(v) for v in self.uncertainty)
            if len(u) != 2:
                raise ValueError('uncertainty must be (lower, upper) in the same unit')
            lo, hi = (values_tuple(v, self.grid, self.target_variable, self.unit) for v in u)
            if any(not l <= v <= h for l, v, h in zip(lo, self.values, hi)):
                raise ValueError('invalid uncertainty bounds')
            object.__setattr__(self, 'uncertainty', (lo, hi))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['grid'] = HourGrid(**data['grid'])
        data['model'] = ModelProvenance.from_dict(data['model'])
        data['input_vintages'] = tuple(InputRef(**v) for v in data['input_vintages'])
        return cls(**data)
