"""Production inference for the exact, user-trained FlightRisk Layer 2 CatBoost models.

Important: all model features are derived from a future flight's *scheduled*
information, frozen training-period history, public airport metadata, US holidays,
and optionally live Open-Meteo *forecasts*. No post-departure fields are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'models'
WX_VARS = ('temperature_2m','precipitation','wind_speed_10m','cloud_cover')
WX_AIRPORT_LIMIT_NOTE = 'Weather forecasts are only used for airports represented in the training weather subset.'

@lru_cache(maxsize=1)
def load_bundle():
    with (ASSETS / 'flightrisk_layer2_metadata.json').open(encoding='utf-8') as f:
        metadata = json.load(f)
    routes = pd.read_csv(ASSETS / 'supported_routes.csv')
    classification = CatBoostClassifier()
    classification.load_model(str(ASSETS / 'flightrisk_layer2_delay_classifier.cbm'))
    regression = CatBoostRegressor()
    regression.load_model(str(ASSETS / 'flightrisk_layer2_arrival_delay_regressor.cbm'))
    expected = metadata['features']
    for model in (classification, regression):
        model_features = model.feature_names_
        if model_features and list(model_features) != expected:
            raise ValueError('Exported model features differ from metadata: refusing to predict.')
    routes['airline'] = routes['airline'].astype(str)
    routes['origin'] = routes['origin'].astype(str)
    routes['dest'] = routes['dest'].astype(str)
    return metadata, routes, classification, regression


def airport_details(code, metadata):
    record=metadata['origin_airport_metadata'].get(code,{})
    return {'code':code, 'city':record.get('city') or code,
            'name': record.get('name') or f'{code} airport',
            'lat':record.get('lat'),'lon':record.get('lon'),
            'timezone':record.get('timezone') or 'UTC'}


def airport_label(code, meta):
    loc=airport_details(code,meta)
    return f"{loc['city']} — {loc['name']} ({code})"


def route_entry(routes, airline, origin, dest):
    found=routes.loc[(routes['airline']==airline)&(routes['origin']==origin)&(routes['dest']==dest)]
    if found.empty:
        raise ValueError('This combination was not sufficiently represented in the training data. Choose a listed route.')
    return found.iloc[0]


def holiday_features(flight_date: date):
    start = pd.Timestamp(flight_date)-pd.Timedelta(days=12)
    end = pd.Timestamp(flight_date)+pd.Timedelta(days=12)
    holiday = USFederalHolidayCalendar().holidays(start=start,end=end)
    if len(holiday)==0:
        return 99,0,0
    days = int(min(abs((pd.Timestamp(flight_date)-h).days) for h in holiday))
    return days,int(days==0),int(days<=3)


def scheduled_times(flight_date:date, departure_time:time, elapsed_min:float, origin:str, dest:str, meta:dict):
    a=airport_details(origin,meta)
    b=airport_details(dest,meta)
    start=datetime.combine(flight_date,departure_time).replace(tzinfo=ZoneInfo(a['timezone']))
    end=(start.astimezone(timezone.utc)+timedelta(minutes=float(elapsed_min))).astimezone(ZoneInfo(b['timezone']))
    return start,end


def build_features(airline:str, origin:str, dest:str, flight_date:date, departure_time:time,
                   meta:dict, routes:pd.DataFrame, wx:Optional[dict]=None):
    """Return one feature row and the reference scheduled arrival, including time zones.

    Historical summaries are explicitly frozen to the train period; the live app
    does not pretend that these are fresh operational measures.
    """
    reference=route_entry(routes,airline,origin,dest)
    distance=float(reference['distance'])
    elapsed=float(reference['sched_elapsed_time'])
    dmin=departure_time.hour*60+departure_time.minute
    weekday=flight_date.weekday()
    day_of_year=flight_date.timetuple().tm_yday
    origin_loc=airport_details(origin,meta)
    dest_loc=airport_details(dest,meta)
    days_to_holiday,is_holiday,near_holiday=holiday_features(flight_date)
    hist=meta['lagged_reference_train_only']
    global_prior=float(meta['global_training_delay_rate'])
    values={'airline':airline,'origin':origin,'dest':dest,
            'month':flight_date.month,'day_of_week':weekday,'is_weekend':int(weekday>=5),
            'dep_time_minutes':dmin,'distance':distance,'sched_elapsed_time':elapsed,
            'route':f'{origin} → {dest}',
            'origin_lat':origin_loc['lat'],'origin_lon':origin_loc['lon'],
            'dest_lat':dest_loc['lat'],'dest_lon':dest_loc['lon'],
            'day_of_year_sin':math.sin(2*math.pi*day_of_year/365.25),
            'day_of_year_cos':math.cos(2*math.pi*day_of_year/365.25),
            'dep_time_minutes_sin':math.sin(2*math.pi*dmin/1440),
            'dep_time_minutes_cos':math.cos(2*math.pi*dmin/1440),
            'is_morning_peak':int(6<=departure_time.hour<=9),
            'is_evening_peak':int(16<=departure_time.hour<=20),
            'days_to_nearest_us_holiday':days_to_holiday,
            'is_us_federal_holiday':is_holiday,
            'within_3_days_us_holiday':near_holiday}
    for name,key in (('route',values['route']),('airline',airline),('origin',origin),('dest',dest),
                     ('origin_hour',f'{origin}||{departure_time.hour}')):
        item=hist.get(name,{}).get(key,{})
        values[f'hist_{name}_flights']=float(item.get('n',0))
        values[f'hist_{name}_late_rate']=float(item.get('late_rate',global_prior))
    # Approximate typical historical sampled flight volume per month (not real-time traffic).
    train_start=pd.Timestamp(meta['train_date_start']).to_period('M')
    train_end=pd.Timestamp(meta['train_date_end']).to_period('M')
    observed_months=max(1,(train_end.year-train_start.year)*12+train_end.month-train_start.month+1)
    values['historic_origin_hour_flights_per_month']=values['hist_origin_hour_flights']/observed_months
    wx=wx or {}
    for prefix in ('origin','dest'):
        found=wx.get(prefix,{})
        values[f'{prefix}_wx_available']=int(all(found.get(v) is not None for v in WX_VARS))
        for v in WX_VARS:
            values[f'{prefix}_wx_{v}']=found.get(v)
    for feature in meta['features']:
        if feature not in values:
            raise ValueError(f'Unable to construct model feature {feature}')
        if feature not in meta['feature_categorical']:
            n=pd.to_numeric(values[feature],errors='coerce')
            values[feature]=float(n) if pd.notna(n) and math.isfinite(float(n)) else float(meta['medians'][feature])
        else:
            values[feature]=str(values[feature])
    start,end=scheduled_times(flight_date,departure_time,elapsed,origin,dest,meta)
    X=pd.DataFrame([{f:values[f] for f in meta['features']}],columns=meta['features'])
    return X, start, end


@dataclass
class Prediction:
    late_probability:float
    arrival_delay_minutes:float
    scheduled_departure:datetime
    typical_scheduled_arrival:datetime
    estimated_arrival:datetime
    weather_origin:bool
    weather_dest:bool
    typical_duration_minutes:float


def predict_flight(airline,origin,dest,flight_date,departure_time,weather:Optional[dict]=None):
    meta,routes,classification,regression=load_bundle()
    X,start,end=build_features(airline,origin,dest,flight_date,departure_time,meta,routes,weather)
    probability=float(classification.predict_proba(X)[0,1])
    delay=float(regression.predict(X)[0])
    if not 0.0<=probability<=1.0 or not math.isfinite(delay):
        raise ValueError('Model output is invalid')
    return Prediction(probability,delay,start,end,end+timedelta(minutes=delay),
                      bool(X['origin_wx_available'].iloc[0]),bool(X['dest_wx_available'].iloc[0]),
                      float(X['sched_elapsed_time'].iloc[0]))


# Calling the public weather API is optional. If unavailable, trained missing-value
# behavior (imputation + available=0) is used, not fabricated weather readings.
@lru_cache(maxsize=200)
def _forecast_for_airport(code:str, forecast_day:str):
    meta,_,_,_=load_bundle()
    if code not in meta.get('weather_airports',[]):
        return None
    loc=airport_details(code,meta)
    response=requests.get('https://api.open-meteo.com/v1/forecast',params={
        'latitude':loc['lat'],'longitude':loc['lon'],
        'hourly':','.join(WX_VARS),'timezone':'UTC',
        'start_date':forecast_day,'end_date':forecast_day,
        'temperature_unit':'celsius','wind_speed_unit':'kmh',
        'precipitation_unit':'mm'},timeout=12,
        headers={'User-Agent':'FlightRiskStudentDemo/1.0'})
    response.raise_for_status()
    return response.json().get('hourly',{})


def forecast_at(code:str, when:datetime, now_utc:Optional[datetime]=None):
    """Use live forecast only near the model's 12–24h pre-departure scenario.

    Forecast lead differs slightly from the historical 24h-lead data; this is an
    educational approximation, NOT a calibrated operational weather model.
    """
    now_utc=now_utc or datetime.now(timezone.utc)
    if not (0<= (when.astimezone(timezone.utc)-now_utc).total_seconds()/3600<=48):
        return {}
    hour=when.astimezone(timezone.utc).replace(minute=0,second=0,microsecond=0)
    try:
        data=_forecast_for_airport(code,hour.date().isoformat())
        if not data or 'time' not in data:return {}
        target=hour.strftime('%Y-%m-%dT%H:%M')
        if target not in data['time']:return {}
        pos=data['time'].index(target)
        out={k:(None if data.get(k,[None]*len(data['time']))[pos] is None
                 else float(data[k][pos])) for k in WX_VARS}
        if all(v is not None for v in out.values()):return out
    except (requests.RequestException,ValueError,TypeError,KeyError,IndexError):
        return {}
    return {}


def load_live_weather(origin:str,dest:str,start:datetime,planned_arrival:datetime):
    # The model was trained on 24h-ahead forecasts, so avoid making a claim
    # that a multi-week forecast is an equivalent observed input.
    ahead=(start.astimezone(timezone.utc)-datetime.now(timezone.utc)).total_seconds()/3600
    if not (8<=ahead<=36):return {},'Weather is not included for this departure window.'
    weather={'origin':forecast_at(origin,start),'dest':forecast_at(dest,planned_arrival)}
    matched=sum(bool(x) for x in weather.values())
    return weather, ('Forecast context used at both airports.' if matched==2 else
                     'Forecast context used at one airport.' if matched==1 else
                     'Forecast unavailable; using historical patterns.')
