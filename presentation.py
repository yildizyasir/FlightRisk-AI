"""Customer-facing airport and pickup planning times.

Departure airport recommendation is a general US domestic travel guideline,
not an ML prediction of check-in or security lines.
Pickup meeting time is a heuristic based on typical arrival and delay regression.
It is NOT a reliable flight-specific ETA or confidence bound.
"""
from datetime import datetime, timedelta
import math


def _display(dt):
    return dt.strftime('%I:%M %p').lstrip('0')


def traveller_planning_times(scheduled_departure: datetime,
                             typical_scheduled_arrival: datetime,
                             predicted_arrival_delay_minutes: float,
                             typical_absolute_error_minutes: float=28.6,
                             preflight_arrival_minutes: int=120,
                             pickup_exit_minutes: int=30) -> dict:
    if scheduled_departure.tzinfo is None or typical_scheduled_arrival.tzinfo is None:
        raise ValueError('Travel planning requires local timezone-aware datetimes.')
    delay=float(predicted_arrival_delay_minutes)
    error=float(typical_absolute_error_minutes)
    if not (math.isfinite(delay) and math.isfinite(error) and error >= 0):
        raise ValueError('Invalid model prediction or error value.')
    if preflight_arrival_minutes <= 0 or pickup_exit_minutes < 0:
        raise ValueError('Invalid planning buffer.')

    # Avoid displaying small minute-level differences as precise ETAs when
    # the regression MAE is much larger than those differences.
    arrival_reference = (typical_scheduled_arrival
                         if abs(delay) <= error
                         else typical_scheduled_arrival + timedelta(minutes=delay))
    pickup_target=arrival_reference + timedelta(minutes=pickup_exit_minutes)
    # Round the pickup recommendation to nearest 10 minutes.
    pickup_target=(pickup_target + timedelta(minutes=5)).replace(second=0,microsecond=0)
    pickup_target=pickup_target.replace(minute=(pickup_target.minute//10)*10)
    airport_target=scheduled_departure - timedelta(minutes=preflight_arrival_minutes)
    return {
        'airport_arrival_time':_display(airport_target),
        'departure_time':_display(scheduled_departure),
        'departure_date_label':scheduled_departure.strftime('%a %d %b · %Z'),
        'airport_date_label':airport_target.strftime('%a %d %b · %Z'),
        'usual_arrival':_display(typical_scheduled_arrival),
        'arrival_date_label':typical_scheduled_arrival.strftime('%a %d %b · %Z'),
        'pickup_time':_display(pickup_target),
        'pickup_date_label':pickup_target.strftime('%a %d %b · %Z'),
        'used_typical_arrival':abs(delay)<=error,
    }
