"""Conservative customer-facing presentation of the existing, independent models.

This does not modify model predictions or turn average absolute error into a confidence interval.
"""
from datetime import datetime, timedelta
import math


def arrival_planning_summary(scheduled: datetime, predicted_delay_minutes: float,
                             typical_absolute_error_minutes: float = 28.6) -> dict:
    """When a small change is less than historical typical error, prefer schedule
    as practical pickup planning reference instead of a misleading exact early ETA.
    """
    delta = float(predicted_delay_minutes)
    mae = float(typical_absolute_error_minutes)
    if not all(math.isfinite(x) for x in (delta, mae)) or mae < 0:
        raise ValueError('Invalid arrival-delay prediction or model error estimate.')
    near_schedule = abs(delta) <= mae
    target = scheduled if near_schedule else scheduled + timedelta(minutes=delta)
    rounded = target + timedelta(minutes=5)
    rounded = rounded.replace(minute=(rounded.minute//10)*10,second=0,microsecond=0)
    if near_schedule:
        explanation = (
            'Plan around the scheduled arrival. The predicted shift is smaller than '
            'the model\'s typical historical error (about 29 minutes), so we do not '
            'recommend relying on a few minutes early or late.'
        )
    else:
        explanation = (
            'A rough planning reference based on historical delays, not a live ETA. '
            'The model\'s average historical error is about 29 minutes; leave flexibility.'
        )
    return {'display_time':rounded.strftime('%I:%M %p').lstrip('0'),
            'date':rounded.date(), 'uses_schedule':near_schedule,
            'explanation':explanation}
