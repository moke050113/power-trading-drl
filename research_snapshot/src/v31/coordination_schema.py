"""Actor-safe immutable value objects. No archive, query or execution capabilities."""
from dataclasses import dataclass
from .schema import ContentRecord


@dataclass(frozen=True)
class LotView(ContentRecord):
    stage: str
    signed_at: str
    unit_price_cny_per_mwh: float
    delivery_plan_mwh: tuple


@dataclass(frozen=True)
class PositionView(ContentRecord):
    hours: tuple
    version: int
    reference_sha256: str
    reference_mwh: tuple
    contract_mwh: tuple
    contract_fee_cny: tuple
    remaining_c1_capacity_mwh: tuple
    lots: tuple


@dataclass(frozen=True)
class ForecastView(ContentRecord):
    target: str
    unit: str
    as_of: str
    published_at: str
    usage: str
    values: tuple


@dataclass(frozen=True)
class ConstraintView(ContentRecord):
    boundaries: tuple
    cumulative_min_mwh: tuple
    cumulative_max_mwh: tuple
    minimum_first_slot_mw: float
    feasible: bool


@dataclass(frozen=True)
class TaskView(ContentRecord):
    slot: int
    resource: str
    certainty: str
    window_start: str
    deadline: str
    max_power_mw: float
    grid_demand_mwh: float
    battery_capacity_mwh: float
    efficiency: float
    soc_initial_estimate: float | None
    soc_actual: float | None
    state_observed_at: str | None
    constraint: ConstraintView | None


@dataclass(frozen=True)
class Observation(ContentRecord):
    decision_at: str
    level: str
    opportunity: int | None
    contract_quote_cny_per_mwh: float | None
    spot_quote_cny_per_mwh: float | None
    spot_capacity_mwh: float | None
    position: PositionView
    forecasts: tuple
    known_tasks: tuple
    unknown_demand_forecast: tuple
    resource_forecast_usage: str


@dataclass(frozen=True)
class MarketToLoad(ContentRecord):
    issued_at: str
    position: PositionView
    exposure_hours: tuple
    base_forecast_signed_gap_mwh: tuple
    known_flexible_grid_mwh: float
    unknown_flexible_grid_mwh: float
    total_point_demand_minus_contract_mwh: float
    c1_future_hour_capacity_mask: tuple
    unresolved_unknown_prior_count: int
    meaning: str = 'point_exposure_not_CVaR_or_execution_plan'


@dataclass(frozen=True)
class LoadToMarket(ContentRecord):
    issued_at: str
    position_version: int
    reference_sha256: str
    position_snapshot_sha256: str
    visible_resource_sha256: str
    future_hours: tuple
    remaining_known_tasks: tuple
    predicted_future_tasks: tuple
    meaning: str = 'advisory_only_no_recontract_or_task_rewrite_permission'
