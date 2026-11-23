"""FlightRisk — polished, customer-facing Streamlit application.
Run with: streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from html import escape

import streamlit as st
import streamlit.components.v1 as components

from inference import (load_bundle, airport_details, airport_label, load_live_weather,
                       predict_flight, route_entry, scheduled_times, holiday_features)
from map_visual import route_map_html
from explanations import condition_explanations, holiday_name_on

st.set_page_config(page_title='FlightRisk | Know before you go', page_icon='✈',
                   layout='wide', initial_sidebar_state='collapsed')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--navy:#081321;--panel:#0d2031;--panel2:#12293b;--line:#264052;--mint:#75eed4;--text:#edf7fb;--muted:#9bb2c0}
html,body,[data-testid="stApp"],.stApp{background:#081321!important;color:#edf7fb!important;font-family:'DM Sans',sans-serif!important}
[data-testid="stApp"]{background:radial-gradient(ellipse 58% 39% at 78% 4%,#1643549c,transparent 70%),#081321!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stDecoration"], [data-testid="stToolbar"], #MainMenu,footer{visibility:hidden!important}
.block-container{max-width:1350px!important;padding:2.0rem 2.7rem 3.5rem!important}
@media(max-width:700px){.block-container{padding:1.4rem 1rem 2rem!important}}
.nav{display:flex;justify-content:space-between;align-items:center;padding-bottom:20px;margin-bottom:34px;border-bottom:1px solid #ffffff1c}
.brand{display:flex;align-items:center;gap:11px;color:#f1f9fc;font-family:Manrope,sans-serif;font-size:22px;font-weight:800;letter-spacing:-.9px}
.brand-mark{border-radius:13px;background:#75eed4;color:#0b2430;width:43px;height:43px;display:inline-flex;align-items:center;justify-content:center;font-size:26px;line-height:1}
.brand .accent{color:#75eed4}
.tag{border:1px solid #2a4c5a;border-radius:40px;color:#c4e7e6;font-size:11px;letter-spacing:1px;padding:10px 13px;font-weight:700}
.tagdot{display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:#75eed4}
.hero-eyebrow{font-size:11px;letter-spacing:2.3px;color:#75eed4;font-weight:800;text-transform:uppercase;margin:4px 0 13px}
.hero-heading{font-family:Manrope,sans-serif;font-size:clamp(37px,4.5vw,64px);font-weight:800;line-height:1.12;letter-spacing:-2.8px;margin:0 0 15px;color:#f2f9fb}
.hero-heading .mint{color:#75eed4}
.hero-sub{color:#a9bdc8;line-height:1.75;font-size:15px;max-width:625px;margin:0 0 34px}
.panelhead{font-family:Manrope,sans-serif;font-size:18px;font-weight:800;letter-spacing:-.4px;color:#f2fbff;margin:0 0 6px}
.panelsub{font-size:12px;color:#a0b7c4;line-height:1.55;margin-bottom:12px}
[data-testid="stVerticalBlockBorderWrapper"]{background:#0d2031!important;border:1px solid #284154!important;border-radius:18px!important;box-shadow:0 12px 34px #0002!important}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]{gap:.75rem!important}
[data-testid="stWidgetLabel"] p{font-size:12px!important;font-weight:700!important;color:#a7bfca!important}
[data-baseweb="select"] > div,[data-baseweb="input"] > div,[data-baseweb="input"] input,[data-testid="stDateInput"] input,[data-testid="stTimeInput"] input{
  background:#132b3e!important;border-color:#355366!important;color:#effbfc!important;border-radius:10px!important;}
[data-baseweb="select"] span{color:#f2fbfc!important}
[data-testid="stRadio"] label{color:#e9f5f9!important}
[data-testid="stButton"] button[kind="primary"]{width:100%;border-radius:11px!important;padding:.72rem 1rem!important;background:#75eed4!important;color:#06212b!important;font-weight:800!important;border:0!important;box-shadow:0 13px 26px #75eed414!important}
[data-testid="stButton"] button[kind="primary"] p{color:#06212b!important;font-size:15px!important;font-weight:800!important}
[data-testid="stButton"] button[kind="primary"]:hover{background:#a2ffe9!important}
.small-note{font-size:11px;color:#85a1b0;line-height:1.65;margin-top:6px}
.routebar{background:#0e2638;border:1px solid #294758;border-radius:14px;display:grid;grid-template-columns:1fr 55px 1fr;gap:10px;align-items:center;padding:17px 20px;margin-top:12px}
.routebar .iata{font-family:Manrope,sans-serif;font-size:24px;font-weight:800;letter-spacing:-.6px;color:#f4fbfc}
.routebar .city{font-size:12px;color:#adc4cf;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.routebar .arrow{color:#75eed4;font-size:22px;text-align:center}
.routebar .right{text-align:right}
.results{margin-top:23px;border-radius:18px;border:1px solid #35565e;background:linear-gradient(124deg,#112f3e,#102337 67%);padding:26px 29px;box-shadow:0 17px 48px #0002}
.results .eyebrow{color:#75eed4;text-transform:uppercase;letter-spacing:1.65px;font-size:11px;font-weight:800;margin-bottom:14px}
.result-flex{display:flex;align-items:center;gap:30px;flex-wrap:wrap}
.risk-number{font-family:Manrope,sans-serif;color:#f7fdfd;font-size:clamp(65px,8vw,105px);font-weight:800;letter-spacing:-5px;line-height:1;min-width:245px;white-space:nowrap}
.risk-number span{font-size:.54em;color:#75eed4;letter-spacing:-2px}
.risk-label{font-family:Manrope,sans-serif;color:#e9f9f9;font-size:20px;font-weight:700;max-width:570px;line-height:1.45}
.risk-caption{color:#aac3ce;font-size:12px;line-height:1.75;margin-top:9px;max-width:610px}
.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px;margin-top:23px}
.detailbox{border:1px solid #345365;background:#102a3b;border-radius:12px;padding:16px 18px;min-width:0}
.detailbox .ey{font-size:10px;color:#9bbac6;letter-spacing:1.1px;text-transform:uppercase;font-weight:800;margin-bottom:8px}
.detailbox .big{font-family:Manrope,sans-serif;font-weight:800;font-size:23px;color:#effafd;letter-spacing:-.6px}
.detailbox .tiny{font-size:11px;color:#8caebc;margin-top:6px}
.results-foot{font-size:11px;line-height:1.65;color:#86a7b3;margin-top:17px}
.footerline{display:flex;flex-wrap:wrap;justify-content:space-between;gap:15px;border-top:1px solid #203948;padding-top:18px;margin-top:27px;color:#7d9ba9;font-size:11px}
@media(max-width:700px){.tag{display:none}.routebar{padding:12px;grid-template-columns:1fr 35px 1fr}.routebar .iata{font-size:20px}.result-flex{gap:13px}.risk-number{min-width:0}.detail-grid{grid-template-columns:1fr}.results{padding:19px}}

.flight-context{margin-top:15px;background:#0d2333;border:1px solid #2d485b;border-radius:15px;padding:19px 22px}
.context-title{font-family:Manrope,sans-serif;font-weight:800;font-size:14px;color:#eaf9fb;margin-bottom:11px}
.context-pills{display:flex;gap:8px;flex-wrap:wrap}
.context-pill{border:1px solid #36596a;background:#122f40;border-radius:999px;padding:9px 12px;font-size:11px;color:#c4e2e7;font-weight:700}
.context-weather{margin-top:12px;font-size:12px;color:#a9e8dc;line-height:1.6}
.context-note{font-size:11px;line-height:1.7;color:#8eabb8;margin-top:12px}

.intent-hint{margin-top:11px;border-radius:11px;border:1px solid #3a6470;background:#122e3f;padding:12px 15px;color:#c6e4eb;font-size:12px;line-height:1.6}
.condition-heading{font-family:Manrope,sans-serif;color:#f1fcfc;font-weight:800;font-size:17px;margin:21px 0 12px}
.condition-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:12px}
.condition-card{background:#10283a;border:1px solid #315064;border-radius:13px;padding:17px 17px;min-width:0}
.condition-card .kicker{font-size:11px;color:#9fc1cc;font-weight:800;letter-spacing:.65px;margin-bottom:8px}
.condition-card .primary{font-family:Manrope,sans-serif;font-weight:800;color:#f0fafb;font-size:14px;line-height:1.45}
.condition-card .secondary{font-size:11px;color:#9cbac7;margin-top:7px;line-height:1.7}
@media(max-width:800px){.condition-grid{grid-template-columns:1fr}}
.explainer{margin-top:14px;background:#0d2c3c;border:1px solid #355d68;border-radius:13px;padding:15px 19px}
.explainer .head{font-size:13px;font-weight:800;color:#c7fff0;margin-bottom:7px}
.explainer .copy{font-size:12px;line-height:1.7;color:#c2dbe3}
.reasons{margin-top:14px;background:#102739;border:1px solid #355465;border-radius:13px;padding:17px 19px}
.reasons .head{font-size:14px;font-weight:800;color:#effafc;margin-bottom:8px}
.reasons .item{font-size:12px;line-height:1.7;color:#c1d9e1;margin-top:8px}
.reasons .note{font-size:11px;color:#95b1be;margin-top:11px;line-height:1.6}

.view-switch [data-testid="stRadio"]{background:#112b3c;border:1px solid #345363;border-radius:13px;padding:5px 11px}
.primary-summary{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.risk-pill{display:inline-block;padding:7px 13px;border:1px solid #47716f;border-radius:999px;font-size:12px;font-weight:800;color:#96f9dc;background:#173c42}
.summary-title{font-family:Manrope,sans-serif;font-size:clamp(19px,2.6vw,27px);font-weight:800;color:#f1fbfd;line-height:1.4}
.summary-strap{font-size:13px;color:#c4d9df;line-height:1.7;margin-top:10px}
.summary-fine{font-size:12px;color:#a9c3cb;line-height:1.75;margin-top:8px}
.mini-meta{margin:13px 0 0;display:flex;gap:10px;flex-wrap:wrap}
.mini-meta span{color:#b8d2da;font-size:11px;padding:7px 10px;border:1px solid #2c5160;border-radius:9px;background:#123142}
.status-line{color:#a5c8cb;font-size:12px;line-height:1.75;margin-top:10px}

</style>''',unsafe_allow_html=True)



meta, routes, model, arrival_model = load_bundle()

st.markdown('''<div class="nav"><div class="brand"><span class="brand-mark">✈</span> FlightRisk<span class="accent">.</span></div>
<div class="tag"><span class="tagdot"></span>ARRIVAL INTELLIGENCE</div></div>
<div class="hero-eyebrow">Travel with more confidence</div>
<h1 class="hero-heading">Know your route.<br><span class="mint">Plan your arrival.</span></h1>
<p class="hero-sub">Understand the chance of arriving late and see an approximate arrival time — without confusing flight statistics or technical jargon.</p>''',unsafe_allow_html=True)

left, right = st.columns([.95,1.45],gap='large')
with left:
    with st.container(border=True):
        st.markdown('<div class="panelhead">Check an arrival</div><div class="panelsub">Select a route and scheduled departure. We’ll estimate arrival delay based on historical flights.</div>',unsafe_allow_html=True)
        journey = st.radio('I want to…',options=['Check my flight','Pick someone up'],horizontal=True,key='journey')
        airlines=sorted(routes['airline'].unique())
        default_airline='Delta Air Lines' if 'Delta Air Lines' in airlines else airlines[0]
        airline=st.selectbox('Airline',options=airlines,index=airlines.index(default_airline),key='airline')
        r1=routes.loc[routes['airline']==airline]
        origin_codes=sorted(r1['origin'].unique(),key=lambda c:airport_label(c,meta))
        default_origin='ATL' if 'ATL' in origin_codes else origin_codes[0]
        origin=st.selectbox('Departure airport',origin_codes,index=origin_codes.index(default_origin),
                            format_func=lambda c:airport_label(c,meta),key='origin')
        r2=r1.loc[r1['origin']==origin]
        dest_codes=sorted(r2['dest'].unique(),key=lambda c:airport_label(c,meta))
        default_dest='MCO' if 'MCO' in dest_codes else dest_codes[0]
        dest=st.selectbox('Arrival airport',dest_codes,index=dest_codes.index(default_dest),
                          format_func=lambda c:airport_label(c,meta),key='dest')
        a,b=st.columns(2,gap='small')
        with a:
            travel_date=st.date_input('Travel date',value=date.today()+timedelta(days=1),
                min_value=date(2025,1,1),max_value=date(2030,12,31),
                format='DD/MM/YYYY',key='travel_date',help='Dates from 2025 through 2030. Later years are extrapolations of historical patterns.')
        with b:
            departure_time=st.time_input('Scheduled departure',value=time(14,0),
                step=timedelta(minutes=15),key='departure_time')

        set_ticket=st.checkbox('Use the scheduled arrival time from my ticket (more accurate reference)',
                               value=False,key='set_ticket')
        ticket_arrival_date=None;ticket_arrival_time=None
        if set_ticket:
            ref=route_entry(routes,airline,origin,dest)
            _,typical_arrival=scheduled_times(travel_date,departure_time,float(ref['sched_elapsed_time']),origin,dest,meta)
            ac,bc=st.columns(2,gap='small')
            with ac:
                ticket_arrival_date=st.date_input('Ticket arrival date',value=typical_arrival.date(),
                                                 format='DD/MM/YYYY',key='ticket_arrival_date')
            with bc:
                ticket_arrival_time=st.time_input('Ticket arrival time (destination local)',
                                                 value=typical_arrival.time().replace(second=0,microsecond=0),
                                                 step=timedelta(minutes=5),key='ticket_arrival_time')
        check=st.button('Estimate arrival  →',type='primary',use_container_width=True)
        st.markdown('<div class="small-note">US domestic routes only · Historical estimates, not live flight tracking · Flight schedules are not checked.</div>',unsafe_allow_html=True)

with right:
    components.html(route_map_html(origin,dest,meta),height=479,scrolling=False)
    o=airport_details(origin,meta);d=airport_details(dest,meta)
    st.markdown(f'''<div class="routebar"><div><div class="iata">{escape(origin)}</div><div class="city">{escape(o['city'])}</div></div>
        <div class="arrow">✈</div><div class="right"><div class="iata">{escape(dest)}</div><div class="city">{escape(d['city'])}</div></div></div>''',unsafe_allow_html=True)

query=(journey,airline,origin,dest,travel_date.isoformat(),departure_time.isoformat(),
       bool(set_ticket),str(ticket_arrival_date) if set_ticket else None,
       str(ticket_arrival_time) if set_ticket else None)
if check:
    try:
        ref=route_entry(routes,airline,origin,dest)
        dep_dt, arr_dt=scheduled_times(travel_date,departure_time,
            float(ref['sched_elapsed_time']),origin,dest,meta)
        with st.spinner('Estimating arrival…'):
            weather,wx_note=load_live_weather(origin,dest,dep_dt,arr_dt)
            result=predict_flight(airline,origin,dest,travel_date,departure_time,weather=weather)
        st.session_state['result']=(query,result,wx_note,weather)
    except Exception:
        st.error('We couldn’t complete this estimate. Please choose a different route or time.')
        st.session_state.pop('result',None)

if 'result' in st.session_state:
    last_query,res,wx_note,weather=st.session_state['result']
    if last_query == query:
        scheduled=res.typical_scheduled_arrival
        source_note='Arrival time is based on a typical historical flight duration, not a verified timetable.'
        if set_ticket:
            # Let the timezone of the destination determine the local ticket time.
            booked=datetime.combine(ticket_arrival_date,ticket_arrival_time).replace(tzinfo=scheduled.tzinfo)
            scheduled=booked
            source_note='Arrival time is calculated using the ticket schedule you entered.'

        predicted=scheduled+timedelta(minutes=res.arrival_delay_minutes)
        # Do not imply minute-level precision when the held-out MAE is ~29 min.
        rounded=predicted+timedelta(minutes=5)
        rounded=rounded.replace(minute=(rounded.minute//10)*10,second=0,microsecond=0)
        pct=int(round(res.late_probability*100))
        if pct<25:
            band='Lower estimated delay risk'
        elif pct<50:
            band='Moderate estimated delay risk'
        else:
            band='Higher estimated delay risk'
        arrival_time=rounded.strftime('%I:%M %p').lstrip('0')
        schedule_time=scheduled.strftime('%I:%M %p').lstrip('0')
        full_date=rounded.strftime('%a %d %b %Y · %Z')
        arrival_sentence='A rough arrival-time estimate, not a live update.'

        if journey=='Pick someone up':
            lead_title='Approximate arrival time'
            lead_value=f'{escape(arrival_time)}'
            lead_note='Allow extra flexibility when planning pickup. Historical prediction error averages about 29 minutes.'
            side_title='Chance of arriving 15+ min late'
            side_value=f'{pct}%'
            side_note=band
        else:
            lead_title='Chance of arriving 15+ minutes late'
            lead_value=f'{pct}%'
            lead_note=f'{band}. This is the risk of a substantial arrival delay, not a promise about the exact arrival minute.'
            side_title='Approximate arrival time'
            side_value=escape(arrival_time)
            side_note='Based on an independently trained arrival-time model.'

        st.markdown(f'''<div class="results" id="results">
          <div class="eyebrow">{'Pickup outlook' if journey=='Pick someone up' else 'Arrival outlook'}</div>
          <div class="primary-summary"><div class="risk-number" style="font-size:clamp(48px,6.2vw,75px);letter-spacing:-3px">{lead_value}</div>
            <div><div class="summary-title">{escape(lead_title)}</div><div class="summary-strap">{escape(lead_note)}</div></div></div>
          <div class="detail-grid">
            <div class="detailbox"><div class="ey">{escape(side_title)}</div>
              <div class="big">{side_value}</div><div class="tiny">{escape(side_note)}</div></div>
            <div class="detailbox"><div class="ey">Scheduled arrival · destination local time</div>
              <div class="big">{escape(schedule_time)}</div><div class="tiny">{escape(scheduled.strftime('%a %d %b %Y · %Z'))}</div></div>
          </div><div class="mini-meta"><span>Estimated arrival date: {escape(full_date)}</span>
          <span>Historical average arrival-time error: ~29 min</span></div>
          <div class="results-foot">{escape(source_note)} The delay chance and the approximate arrival time are produced by different models and therefore describe different aspects of the flight. Confirm actual flight status with the airline or airport.</div>
        </div>''',unsafe_allow_html=True)

        days,is_holiday,near_holiday=holiday_features(travel_date)
        if is_holiday:
            holiday_primary=holiday_name_on(travel_date) or 'US federal holiday'
            holiday_secondary='Travel conditions may differ on holidays; this is not a proven cause of a particular delay.'
        elif near_holiday:
            holiday_primary='Near a US federal holiday'
            holiday_secondary='Holiday-related travel patterns may differ from typical days.'
        else:
            holiday_primary='No nearby federal holiday'
            holiday_secondary='No US federal holiday within three days.'

        wx_parts=[]
        for side,name in [('origin','Departure'),('dest','Arrival')]:
            w=weather.get(side,{})
            if all(w.get(k) is not None for k in ('temperature_2m','wind_speed_10m','precipitation')):
                wx_parts.append(f"{name}: {float(w['temperature_2m']):.0f}°C · wind {float(w['wind_speed_10m']):.0f} km/h · rain {float(w['precipitation']):.1f} mm")
        wx_primary='Available forecast' if wx_parts else 'Forecast not available'
        wx_secondary=(' | '.join(wx_parts) if wx_parts else
            'Weather data are used only for supported airports when a suitable near-term forecast exists. Otherwise the model falls back to historical patterns.')
        st.markdown(f'''<div class="condition-heading">Conditions to be aware of</div>
          <div class="condition-grid">
            <div class="condition-card"><div class="kicker">☁ WEATHER</div><div class="primary">{escape(wx_primary)}</div><div class="secondary">{escape(wx_secondary)}</div></div>
            <div class="condition-card"><div class="kicker">▦ HOLIDAY</div><div class="primary">{escape(holiday_primary)}</div><div class="secondary">{escape(holiday_secondary)}</div></div>
            <div class="condition-card"><div class="kicker">✈ ROUTE HISTORY</div><div class="primary">{escape(origin)} → {escape(dest)}</div><div class="secondary">The model uses past patterns for this airline, route, departure time and airports.</div></div>
          </div>''',unsafe_allow_html=True)

        possible=condition_explanations(weather,bool(is_holiday),bool(near_holiday))
        if possible:
            notes_html=''.join(f'<div class="item">• {escape(note)}</div>' for note in possible)
            st.markdown('<div class="reasons"><div class="head">Possible travel disruption factors</div>'+notes_html+
                '<div class="note">These factors can be associated with delays, but they do not prove why an individual flight will be delayed.</div></div>',unsafe_allow_html=True)

        # Model trained on a limited historical date range; do not imply validity into the far future.
        if travel_date>date(2026,12,31):
            st.info('This date is beyond the period represented in the training data. Treat this estimate as exploratory; airline schedules and travel conditions may change.')

st.markdown('''<div class="footerline"><span>FlightRisk © 2026 · Know before you go.</span>
<span>Historical US domestic flight data · No live status or verified flight schedule</span></div>''',unsafe_allow_html=True)
