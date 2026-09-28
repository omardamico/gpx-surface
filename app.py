"""
Analisi del fondo di un percorso tramite BRouter. Avvio: streamlit run app.py
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date, time, timedelta

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import brouter_core as bc
import gemini_coach as gc
import weather as wx

st.set_page_config(page_title="BRouter · Analisi percorso", page_icon="🥾", layout="wide")

CSS = """
<style>
header[data-testid="stHeader"], #MainMenu, footer { display: none; }
.block-container { padding: 0 1.25rem 2rem; max-width: 100%; }
.br-navbar {
  background: #343a40; color: #fff; margin: 0 -1.25rem .9rem; padding: .7rem 1.25rem;
  display: flex; justify-content: space-between; align-items: center;
}
.br-navbar .brand { font-size: 1.3rem; font-weight: 400; letter-spacing: .01em; }
.br-navbar .brand sup { font-size: .65rem; opacity: .8; margin-left: .15rem; }
.br-navbar .hint { font-size: .9rem; color: #ced4da; }
.br-title { color: #007bff; font-size: 1.35rem; margin: .1rem 0 .4rem; }
.br-stats {
  display: flex; flex-wrap: wrap; justify-content: space-around; gap: .5rem;
  border-top: 1px solid #dee2e6; border-bottom: 1px solid #dee2e6; padding: .45rem 0; margin: .4rem 0 1rem;
}
.br-stat { text-align: center; min-width: 120px; }
.br-stat .label { color: #6c757d; font-size: .85rem; }
.br-stat .value { font-weight: 700; font-size: 1.15rem; }
.br-legend { width: 100%; border-collapse: collapse; font-size: .92rem; }
.br-legend td { padding: .28rem .3rem; border-bottom: 1px solid #f1f3f5; vertical-align: middle; }
.br-legend .sw { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
.br-legend .bar { background: #e9ecef; height: 6px; border-radius: 3px; width: 100%; }
.br-legend .bar > div { height: 6px; border-radius: 3px; }
.br-legend .num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.br-legend .muted { color: #868e96; font-size: .8rem; }
.br-empty { border: 1px dashed #adb5bd; border-radius: 6px; padding: 2.5rem 1.5rem; text-align: center; color: #495057; }
</style>
"""

NAVBAR = """
<div class="br-navbar">
  <span class="brand">🥾 Analisi percorso</span>
  <span class="hint">BRouter<sup>API</sup> · profilo a piedi (hiking-mountain) · OpenStreetMap</span>
</div>
"""


@dataclass(frozen=True)
class Options:
    with_forest: bool
    climb_threshold_m: float


@st.cache_data(show_spinner=False, ttl=3600)
def run_analysis(req: bc.RouteRequest, with_forest: bool) -> bc.RouteAnalysis:
    return bc.analyze(req, with_forest)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_advice(_api_key: str, context_json: str, activity: str, level: str) -> gc.CoachAdvice:
    # La chiave è esclusa dalla cache key (prefisso _): stessa richiesta, stessa risposta
    return gc.ask_coach(_api_key, context_json, activity, level)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_weather(lat: float, lon: float, day: date, start: time, duration_s: float, today: date) -> wx.WeatherInfo:
    return wx.fetch_weather(lat, lon, day, start, duration_s, today)


def render_toolbar() -> tuple[bc.RouteRequest | None, Options]:
    """Barra comandi; restituisce la richiesta solo quando si preme Analizza."""
    c_src, c_input, c_opts, c_go = st.columns([1.2, 4.4, 0.9, 0.9], vertical_alignment="bottom")

    source = c_src.radio("Sorgente", ["File GPX", "Link BRouter-Web"], label_visibility="collapsed")
    is_gpx = source == "File GPX"

    if is_gpx:
        upload = c_input.file_uploader("Traccia GPX", type=["gpx"], label_visibility="collapsed")
        url = ""
    else:
        upload = None
        url = c_input.text_input("Link BRouter-Web", label_visibility="collapsed",
                                 placeholder="https://brouter.de/brouter-web/#...&lonlats=...&profile=...")

    with c_opts.popover("Opzioni"):
        spacing = st.slider("Punti di aggancio GPX ogni (m)", 50, 500, 150, 25, disabled=not is_gpx,
                            help="Più è basso, più il percorso ricalcolato segue fedelmente la traccia.")
        with_forest = st.checkbox("Rileva tratti nel bosco", value=True,
                                  help="Interroga OpenStreetMap (Overpass) per landuse=forest e natural=wood.")
        climb_threshold = st.slider("Dislivello minimo di una salita (m)", 5, 50, int(bc.CLIMB_THRESHOLD_M), 5,
                                    help="Saliscendi più piccoli non spezzano una salita. Si applica subito, "
                                         "senza ricalcolare il percorso.")
    options = Options(with_forest=with_forest, climb_threshold_m=float(climb_threshold))

    if not c_go.button("Analizza", type="primary"):
        return None, options

    if is_gpx:
        if upload is None:
            raise ValueError("Carica un file GPX.")
        return bc.request_from_gpx(upload.getvalue(), spacing), options
    return bc.parse_brouter_url(url), options


def render_stats(s: bc.RouteSummary) -> None:
    stats = (
        ("Distanza", f"{bc.fmt_num(s.length_m / 1000, 1)} km"),
        ("Tempo", bc.fmt_duration(s.total_time_s)),
        ("Salita | Salita piana", f"{bc.fmt_num(s.ascend_m)} m | {bc.fmt_num(s.plain_ascend_m)} m"),
        ("Discesa", f"{bc.fmt_num(s.descend_m)} m"),
        ("Quota min – max", f"{bc.fmt_num(s.ele_min)} – {bc.fmt_num(s.ele_max)} m"),
        ("Energia", f"{bc.fmt_num(s.energy_kwh, 2)} kWh"),
        ("Costo | Fattore medio", f"{bc.fmt_num(s.cost)} | {bc.fmt_num(s.mean_cost_factor, 2)}"),
    )
    cells = "".join(
        f'<div class="br-stat"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in stats
    )
    st.markdown(f'<div class="br-stats">{cells}</div>', unsafe_allow_html=True)


def build_map(a: bc.RouteAnalysis) -> folium.Map:
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)

    def coords(seg) -> list[list[float]]:
        return bc.track_slice(a.track, seg.km_start, seg.km_end)[["lat", "lon"]].to_numpy().tolist()

    # Alone bianco sotto la traccia per leggibilità su mappa OSM
    folium.PolyLine(a.track[["lat", "lon"]].to_numpy().tolist(), color="#ffffff", weight=9, opacity=0.9).add_to(fmap)
    for seg in a.segments.itertuples():
        folium.PolyLine(
            coords(seg), color=bc.SURFACE_COLORS[seg.fondo], weight=5, opacity=0.95,
            tooltip=f"km {bc.fmt_num(seg.km_start, 2)}–{bc.fmt_num(seg.km_end, 2)} · {seg.fondo} · {seg.via}",
        ).add_to(fmap)

    start, end = a.track.iloc[0], a.track.iloc[-1]
    folium.Marker([start.lat, start.lon], tooltip="Partenza", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([end.lat, end.lon], tooltip="Arrivo", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    fmap.fit_bounds([[a.track.lat.min(), a.track.lon.min()], [a.track.lat.max(), a.track.lon.max()]])
    return fmap


def render_surface_legend(surfaces: pd.DataFrame) -> None:
    rows = "".join(
        f"""<tr>
          <td style="width:18px"><span class="sw" style="background:{bc.SURFACE_COLORS[r.fondo]}"></span></td>
          <td>{html.escape(r.fondo)}{
              f'<div class="muted">{bc.fmt_num(r.dedotto_m / 1000, 1)} km dedotti</div>' if r.dedotto_m > 0 else ''}</td>
          <td style="width:35%"><div class="bar"><div style="width:{r.pct:.1f}%;background:{bc.SURFACE_COLORS[r.fondo]}"></div></div></td>
          <td class="num"><b>{bc.fmt_num(r.pct)}%</b></td>
          <td class="num">{bc.fmt_num(r.km, 2)} km</td>
        </tr>"""
        for r in surfaces.itertuples()
    )
    st.markdown(f'<table class="br-legend">{rows}</table>', unsafe_allow_html=True)


def build_elevation_chart(a: bc.RouteAnalysis, climbs: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    seen: set[str] = set()

    def add_segment(seg) -> None:
        part = bc.track_slice(a.track, seg.km_start, seg.km_end)
        color = bc.SURFACE_COLORS[seg.fondo]
        fig.add_trace(go.Scatter(
            x=part.km, y=part.ele, mode="lines", fill="tozeroy", line=dict(color=color, width=2),
            fillcolor=color, opacity=0.85, name=seg.fondo, legendgroup=seg.fondo,
            showlegend=seg.fondo not in seen,
            hovertemplate=f"km %{{x:.2f}}<br>%{{y:.0f}} m<br>{seg.fondo} · {seg.via}<extra></extra>",
        ))
        seen.add(seg.fondo)

    for seg in a.segments.itertuples():
        add_segment(seg)

    if a.forest is not None and a.forest.ranges:
        for x0, x1 in a.forest.ranges:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=bc.FOREST_COLOR, opacity=0.14, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Bosco",
                                 marker=dict(symbol="square", size=12, color=bc.FOREST_COLOR, opacity=0.35)))

    # Etichetta sopra ogni salita: dislivello e pendenza media
    for climb in climbs.query("tipo == 'salita'").itertuples():
        fig.add_annotation(
            x=(climb.km_start + climb.km_end) / 2, y=climb.ele_max, yshift=12, showarrow=False,
            text=f"+{bc.fmt_num(climb.dz)} m · {bc.fmt_num(climb.pendenza_media, 1)}%",
            font=dict(size=11, color="#212529"), bgcolor="rgba(255,255,255,0.75)",
        )

    s = a.summary
    pad = max(25.0, (s.ele_max - s.ele_min) * 0.12)
    fig.update_layout(
        template="plotly_white", height=300, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="km", range=[0, s.length_m / 1000]),
        yaxis=dict(title="m", range=[s.ele_min - pad, s.ele_max + pad]),
    )
    return fig


def render_segments(a: bc.RouteAnalysis) -> None:
    columns = ["km_start", "km_end", "lunghezza_m", "fondo", "via", "pendenza_pct",
               "salita_m", "discesa_m", "quota_m", *(["bosco_pct"] if "bosco_pct" in a.segments else []),
               "difficolta", "dedotto"]
    table = a.segments[columns]
    st.dataframe(
        table, hide_index=True, height=420,
        column_config={
            "km_start": st.column_config.NumberColumn("Da km", format="%.2f"),
            "km_end": st.column_config.NumberColumn("A km", format="%.2f"),
            "lunghezza_m": st.column_config.NumberColumn("Lunghezza (m)", format="%.0f"),
            "fondo": "Fondo",
            "via": "Tipo di via",
            "pendenza_pct": st.column_config.NumberColumn("Pendenza %", format="%.1f"),
            "salita_m": st.column_config.NumberColumn("D+ (m)", format="%.0f"),
            "discesa_m": st.column_config.NumberColumn("D− (m)", format="%.0f"),
            "quota_m": st.column_config.NumberColumn("Quota fine (m)", format="%.0f"),
            "bosco_pct": st.column_config.ProgressColumn("Bosco", format="%.0f%%", min_value=0, max_value=100),
            "difficolta": "Difficoltà",
            "dedotto": st.column_config.CheckboxColumn("Fondo dedotto"),
        },
    )
    st.download_button("Scarica CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name="tratti_percorso.csv", mime="text/csv")


def render_ways(a: bc.RouteAnalysis) -> None:
    st.dataframe(
        bc.breakdown(a.steps, "via")[["via", "km", "pct"]], hide_index=True,
        column_config={
            "via": "Tipo di via",
            "km": st.column_config.NumberColumn("km", format="%.2f"),
            "pct": st.column_config.ProgressColumn("%", format="%.0f%%", min_value=0, max_value=100),
        },
    )


def render_climbs(climbs: pd.DataFrame) -> None:
    if climbs.empty:
        st.info("Nessun dato altimetrico disponibile per questo percorso.")
        return
    st.markdown("\n\n".join(bc.describe_climbs(climbs)))


def _secret_key() -> str | None:
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:  # nessun secrets.toml presente
        return None


def render_weather(a: bc.RouteAnalysis) -> wx.WeatherInfo | None:
    """Meteo alla partenza per data e ora scelte; None se non disponibile."""
    today = date.today()
    c_day, c_time, c_info = st.columns([1.2, 1, 3], vertical_alignment="bottom")
    day = c_day.date_input("Giorno", value=today, min_value=date(1950, 1, 1),
                           max_value=today + timedelta(days=wx.FORECAST_MAX_DAYS), format="DD/MM/YYYY")
    start = c_time.time_input("Partenza", value=time(9, 0), step=900)

    lat, lon = float(a.track["lat"].iloc[0]), float(a.track["lon"].iloc[0])
    try:
        with st.spinner("Recupero il meteo…"):
            info = cached_weather(round(lat, 3), round(lon, 3), day, start, a.summary.total_time_s, today)
    except wx.WeatherError as exc:
        st.warning(str(exc))
        return None

    c_info.caption(f"{info.source}, al punto di partenza (quota modello {bc.fmt_num(info.elevation_m)} m). "
                   f"Finestra {info.start:%H:%M}–{info.end:%H:%M} stimata sul tempo a piedi di BRouter.")

    rain_prob = f" · {bc.fmt_num(info.rain_prob_max)}%" if pd.notna(info.rain_prob_max) else ""
    stats = (
        ("Condizioni", info.conditions),
        ("Temperatura", f"{bc.fmt_num(info.temp_min)}–{bc.fmt_num(info.temp_max)} °C"),
        ("Percepita", f"{bc.fmt_num(info.feels_min)}–{bc.fmt_num(info.feels_max)} °C"),
        ("Pioggia", f"{bc.fmt_num(info.rain_mm, 1)} mm{rain_prob}"),
        ("Vento | Raffiche", f"{bc.fmt_num(info.wind_max)} | {bc.fmt_num(info.gust_max)} km/h"),
        ("Terreno", info.ground),
        ("Tramonto", f"{info.sunset:%H:%M}" if info.sunset else "–"),
    )
    cells = "".join(f'<div class="br-stat"><div class="label">{label}</div><div class="value">{html.escape(value)}</div></div>'
                    for label, value in stats)
    st.markdown(f'<div class="br-stats">{cells}</div>', unsafe_allow_html=True)

    alerts = info.warnings()
    if alerts:
        st.markdown("\n".join(f"- ⚠️ {alert}" for alert in alerts))
    st.plotly_chart(build_weather_chart(info), config={"displayModeBar": False})
    return info


def build_weather_chart(info: wx.WeatherInfo) -> go.Figure:
    h = info.hourly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h["time"], y=h["precipitation"], name="Pioggia (mm)", yaxis="y2",
                         marker_color="#007bff", opacity=0.45, hovertemplate="%{y:.1f} mm<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["time"], y=h["temperature_2m"], name="Temperatura (°C)", mode="lines",
                             line=dict(color="#e67e22", width=2), hovertemplate="%{y:.0f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["time"], y=h["apparent_temperature"], name="Percepita (°C)", mode="lines",
                             line=dict(color="#e67e22", width=1, dash="dot"), hovertemplate="%{y:.0f} °C<extra></extra>"))
    fig.add_vrect(x0=info.start, x1=info.end, fillcolor="#343a40", opacity=0.08, line_width=0, layer="below")
    fig.update_layout(
        template="plotly_white", height=240, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(tickformat="%H:%M"),
        yaxis=dict(title="°C"),
        yaxis2=dict(title="mm", overlaying="y", side="right", rangemode="tozero", showgrid=False),
    )
    return fig


def render_coach(a: bc.RouteAnalysis, climbs: pd.DataFrame, weather: wx.WeatherInfo | None) -> None:
    api_key = gc.resolve_api_key(_secret_key(), st.session_state.get("gemini_key"))
    if api_key is None:
        st.markdown(f"Per i consigli serve una chiave gratuita di Google AI Studio: "
                    f"[creala qui]({gc.API_KEY_URL}), poi incollala sotto. Resta solo in questa sessione.")
        st.text_input("Chiave API Gemini", type="password", key="gemini_key")
        return

    c_act, c_lvl, c_go = st.columns([2, 2, 1.2], vertical_alignment="bottom")
    activity = c_act.selectbox("Attività", gc.ACTIVITIES)
    level = c_lvl.selectbox("Livello", gc.LEVELS, index=1)

    context_json = gc.context_to_json(gc.build_context(a, climbs, weather.to_context() if weather else None))
    request_key = (context_json, activity, level)

    if c_go.button("Chiedi consigli"):
        try:
            with st.spinner("Gemini sta analizzando il percorso…"):
                st.session_state["coach"] = (request_key, cached_advice(api_key, context_json, activity, level))
        except gc.GeminiError as exc:
            st.error(str(exc))
            return

    stored = st.session_state.get("coach")
    if stored is None or stored[0] != request_key:
        meteo = (f"e il meteo del {weather.day:%d/%m} dalle {weather.start:%H:%M}" if weather
                 else "senza meteo (non disponibile per la data scelta)")
        st.caption(f"Premi Chiedi consigli: a Gemini arrivano km, quote, pendenze e fondo {meteo}, "
                   "nessuna coordinata. Data e ora si scelgono nella scheda Meteo.")
        return
    render_advice(stored[1])


def render_advice(advice: gc.CoachAdvice) -> None:
    data = advice.data
    if data is None:
        st.markdown(advice.raw_text)
        st.caption(f"Risposta non strutturata di {advice.model}.")
        return

    def km(value: float) -> str:
        return bc.fmt_num(value, 1)

    st.markdown(data.get("sintesi", ""))

    moves = data.get("salite_e_discese", [])
    if moves:
        st.markdown("##### Salite e discese")
        st.markdown("\n".join(
            f"- {gc.STRATEGY_ICONS.get(m['strategia'], '•')} **Km {km(m['km_da'])} → {km(m['km_a'])} · "
            f"{m['strategia']}.** {m['motivo']}"
            for m in moves
        ))

    food = data.get("alimentazione", [])
    if food:
        st.markdown("##### Alimentazione e idratazione")
        st.markdown("\n".join(f"- **Km {km(f['km'])}** · {f['consiglio']}" for f in food))

    rain = data.get("pioggia") or {}
    if rain:
        st.markdown("##### Se piove")
        st.markdown(f"**Scarpe consigliate:** {rain.get('scarpe_consigliate', '–')}\n\n"
                    f"**Da evitare:** {rain.get('scarpe_da_evitare', '–')}")
        slippery = rain.get("tratti_scivolosi", [])
        if slippery:
            st.markdown("\n".join(
                f"- **Km {km(t['km_da'])} → {km(t['km_a'])}** · {t['motivo']}" for t in slippery
            ))

    outfit = data.get("meteo_e_abbigliamento")
    if outfit:
        st.markdown("##### Meteo e abbigliamento")
        st.markdown(outfit)

    others = data.get("altri_consigli", [])
    if others:
        st.markdown("##### Altri consigli")
        st.markdown("\n".join(f"- {o}" for o in others))

    st.caption(f"Generato da {advice.model}. Sono indicazioni generiche: valuta sempre condizioni reali e "
               f"la tua forma.")


def render_results(a: bc.RouteAnalysis, options: Options) -> None:
    climbs = bc.detect_climbs(a.track, a.steps, options.climb_threshold_m)

    for warning in a.warnings:
        st.warning(warning)

    col_map, col_data = st.columns([3, 2], gap="medium")
    with col_map:
        st_folium(build_map(a), height=560, use_container_width=True, returned_objects=[])
    with col_data:
        st.markdown('<div class="br-title">▸ Data</div>', unsafe_allow_html=True)
        st.markdown(bc.narrative(a, options.climb_threshold_m))
        render_surface_legend(bc.breakdown(a.steps, "fondo"))

    render_stats(a.summary)
    st.plotly_chart(build_elevation_chart(a, climbs), config={"displayModeBar": False})

    tab_climb, tab_weather, tab_coach, tab_seg, tab_way, tab_raw = st.tabs(
        ["Salite e discese", "Meteo", "Consigli Gemini", "Tratti per km", "Tipo di via", "Data BRouter"])
    with tab_climb:
        render_climbs(climbs)
    # Il meteo va letto prima dei consigli: data e ora scelte qui finiscono nel prompt di Gemini
    with tab_weather:
        weather = render_weather(a)
    with tab_coach:
        render_coach(a, climbs, weather)
    with tab_seg:
        render_segments(a)
    with tab_way:
        render_ways(a)
    with tab_raw:
        raw_cols = [c for c in bc.BROUTER_COLUMNS if c in a.steps]
        st.dataframe(a.steps[raw_cols], hide_index=True, height=420)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(NAVBAR, unsafe_allow_html=True)

    try:
        request, options = render_toolbar()
        if request is not None:
            with st.spinner("Calcolo del percorso con BRouter…"):
                st.session_state["analysis"] = run_analysis(request, options.with_forest)
    except (ValueError, bc.BRouterError) as exc:
        st.error(str(exc))
        options = Options(with_forest=True, climb_threshold_m=bc.CLIMB_THRESHOLD_M)

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.markdown(
            '<div class="br-empty">Carica una traccia GPX oppure incolla un link BRouter-Web, '
            'scegli il profilo e premi <b>Analizza</b>.<br>'
            '<span style="color:#868e96">Vedrai fondo stradale, bosco e dislivello chilometro per chilometro.</span></div>',
            unsafe_allow_html=True,
        )
        return
    render_results(analysis, options)


main()
