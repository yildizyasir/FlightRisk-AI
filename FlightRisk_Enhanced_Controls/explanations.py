"""Explanations of predictions, with no ungrounded causal claims."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def holiday_name_on(travel_date) -> str | None:
    observed = USFederalHolidayCalendar().holidays(
        start=pd.Timestamp(travel_date), end=pd.Timestamp(travel_date), return_name=True
    )
    return str(observed.iloc[0]) if len(observed) else None

def prediction_explanation(late_probability: float, delay_minutes: float) -> str:
    p = round(100 * late_probability)
    if delay_minutes < -0.5:
        num = abs(delay_minutes)
        difference = f"{num:.0f} minute{'s' if round(num) != 1 else ''} before schedule"
    elif delay_minutes > 0.5:
        difference = f"{delay_minutes:.0f} minute{'s' if round(delay_minutes) != 1 else ''} after schedule"
    else:
        difference = "approximately on schedule"
    return (f"{p}% is the estimated chance of arriving 15+ minutes late. "
            f"The estimated arrival ({difference}) comes from a separate model. "
            "Both can be true: some flights arrive early, while others are substantially delayed.")


def condition_explanations(weather: dict, is_holiday: bool, near_holiday: bool) -> list[str]:
    """Display *possible* disruption factors, not trained per-feature attribution.

    Thresholds are descriptive display flags, not learned decision boundaries.
    """
    notes = []
    flagged = []
    available = False
    for side, location in [('origin', 'departure'), ('dest', 'arrival')]:
        values = (weather or {}).get(side, {}) or {}
        wind = values.get('wind_speed_10m')
        rain = values.get('precipitation')
        if wind is not None or rain is not None:
            available = True
        try:
            if wind is not None and float(wind) >= 35:
                flagged.append(f"wind near {location} ({float(wind):.0f} km/h)")
            if rain is not None and float(rain) >= 2:
                flagged.append(f"rain near {location} ({float(rain):.1f} mm/h)")
        except (TypeError, ValueError):
            pass
    if flagged:
        notes.append('Weather: Forecast '+', '.join(flagged)+'. These conditions could contribute to disruption, but do not prove why this flight might be late.')
    # When weather is absent or unremarkable, avoid a redundant second box.

    if is_holiday:
        notes.append('Holiday: The flight is scheduled on a US federal holiday. Travel patterns may differ, but this is not a confirmed cause of delay.')
    elif near_holiday:
        notes.append('Holiday: The trip is close to a US federal holiday. Demand may differ; this does not prove an increased delay risk.')

    return notes
