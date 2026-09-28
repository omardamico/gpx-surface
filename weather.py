"""
Meteo del giorno dell'uscita tramite Open-Meteo (gratuito, senza chiave).

Per date fino a ~3 mesi fa e fino a 15 giorni avanti uso l'API forecast; per date più
vecchie l'archivio storico (ERA5), che non ha la probabilità di pioggia.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import requests

import brouter_core as bc

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_MAX_DAYS = 15
FORECAST_PAST_DAYS = 85
PREV_RAIN_DAYS = 3

_COMMON_HOURLY = ("temperature_2m", "apparent_temperature", "precipitation", "weather_code",
                  "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m")
FORECAST_HOURLY = (*_COMMON_HOURLY, "precipitation_probability")
DAILY = ("sunrise", "sunset", "precipitation_sum")

WMO_CODES = {
    0: "sereno", 1: "prevalentemente sereno", 2: "parzialmente nuvoloso", 3: "coperto",
    45: "nebbia", 48: "nebbia con brina",
    51: "pioviggine debole", 53: "pioviggine", 55: "pioviggine intensa",
    56: "pioviggine gelata", 57: "pioviggine gelata intensa",
    61: "pioggia debole", 63: "pioggia moderata", 65: "pioggia forte",
    66: "pioggia gelata", 67: "pioggia gelata forte",
    71: "neve debole", 73: "neve moderata", 75: "neve forte", 77: "granuli di neve",
    80: "rovesci deboli", 81: "rovesci moderati", 82: "rovesci violenti",
    85: "rovesci di neve", 86: "rovesci di neve forti",
    95: "temporale", 96: "temporale con grandine", 99: "temporale con grandine forte",
}


class WeatherError(Exception):
    """Meteo non disponibile per la data o la posizione richiesta."""


@dataclass(frozen=True)
class WeatherInfo:
    """Meteo nella finestra oraria dell'uscita, più il contesto dei giorni precedenti."""
    source: str
    day: date
    start: datetime
    end: datetime
    elevation_m: float
    hourly: pd.DataFrame
    temp_min: float
    temp_max: float
    feels_min: float
    feels_max: float
    rain_mm: float
    rain_prob_max: float
    wind_max: float
    gust_max: float
    worst_code: int
    main_code: int
    rain_prev_mm: float
    sunrise: datetime | None
    sunset: datetime | None

    @property
    def conditions(self) -> str:
        main, worst = describe_code(self.main_code), describe_code(self.worst_code)
        return main if main == worst else f"{main}, a tratti {worst}"

    @property
    def ground(self) -> str:
        """Stato probabile del fondo naturale, da pioggia dei giorni prima e durante l'uscita."""
        wet = self.rain_prev_mm + self.rain_mm
        if wet < 1:
            return "asciutto"
        if wet < 8:
            return "umido"
        return "bagnato, fango probabile su terra ed erba"

    @property
    def ends_after_sunset(self) -> bool:
        return self.sunset is not None and self.end > self.sunset

    def warnings(self) -> list[str]:
        checks = (
            (self.worst_code >= 95, "Temporali previsti: evita creste e zone esposte, valuta di rimandare."),
            (self.worst_code in (56, 57, 66, 67) or self.feels_min <= 1,
             "Rischio ghiaccio o gelo, soprattutto su roccia, legno e lastricato."),
            (self.rain_mm >= 1 or _gt(self.rain_prob_max, 50), "Pioggia probabile durante l'uscita."),
            (self.gust_max >= 50, f"Raffiche fino a {bc.fmt_num(self.gust_max)} km/h."),
            (self.temp_max >= 28, f"Caldo: fino a {bc.fmt_num(self.temp_max)} °C, aumenta l'idratazione."),
            (self.ends_after_sunset, f"Il tramonto è alle {self.sunset:%H:%M}: rischi di finire col buio."),
        )
        return [message for condition, message in checks if condition]

    def to_context(self) -> dict:
        """Versione compatta per il prompt di Gemini."""
        return {
            "fonte": self.source,
            "data": self.day.isoformat(),
            "orario": f"{self.start:%H:%M}-{self.end:%H:%M}",
            "condizioni": self.conditions,
            "temperatura_c": [_r(self.temp_min), _r(self.temp_max)],
            "percepita_c": [_r(self.feels_min), _r(self.feels_max)],
            "pioggia_durante_mm": _r(self.rain_mm, 1),
            "probabilita_pioggia_max_pct": _r(self.rain_prob_max),
            "vento_max_kmh": _r(self.wind_max),
            "raffiche_max_kmh": _r(self.gust_max),
            f"pioggia_{PREV_RAIN_DAYS}_giorni_prima_mm": _r(self.rain_prev_mm, 1),
            "stato_terreno": self.ground,
            "tramonto": f"{self.sunset:%H:%M}" if self.sunset else None,
        }


def describe_code(code: int) -> str:
    return WMO_CODES.get(int(code), "condizioni non classificate")


def fetch_weather(lat: float, lon: float, day: date, start_time: time,
                  duration_s: float, today: date) -> WeatherInfo:
    """
    Meteo alla partenza per il giorno scelto, nella finestra [start_time, start_time + durata].

    @throws WeatherError se la data è fuori copertura o Open-Meteo non risponde
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise WeatherError("Posizione di partenza non valida.")
    if not math.isfinite(duration_s) or duration_s <= 0:
        duration_s = 3600.0

    offset = (day - today).days
    if offset > FORECAST_MAX_DAYS:
        raise WeatherError(f"Le previsioni arrivano fino al {today + timedelta(days=FORECAST_MAX_DAYS):%d/%m/%Y}.")

    use_forecast = offset >= -FORECAST_PAST_DAYS
    url, hourly_vars = (FORECAST_URL, FORECAST_HOURLY) if use_forecast else (ARCHIVE_URL, _COMMON_HOURLY)
    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(DAILY),
        "timezone": "auto",
        "start_date": (day - timedelta(days=PREV_RAIN_DAYS)).isoformat(),
        "end_date": day.isoformat(),
    }
    try:
        resp = bc.http_session().get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        raise WeatherError(f"Open-Meteo non raggiungibile: {exc}") from exc
    if resp.status_code != 200:
        try:
            reason = resp.json().get("reason", resp.text[:200])
        except ValueError:
            reason = resp.text[:200]
        raise WeatherError(f"Open-Meteo ha risposto {resp.status_code}: {reason}")

    source = "Open-Meteo, previsioni" if offset >= 0 else ("Open-Meteo, dati recenti" if use_forecast
                                                            else "Open-Meteo, archivio storico")
    return _build_info(resp.json(), day, start_time, duration_s, source)


def _build_info(data: dict, day: date, start_time: time, duration_s: float, source: str) -> WeatherInfo:
    hourly = pd.DataFrame(data.get("hourly") or {})
    if hourly.empty or "time" not in hourly:
        raise WeatherError("Open-Meteo non ha restituito dati orari.")
    hourly["time"] = pd.to_datetime(hourly["time"])
    # Colonne mancanti (es. probabilità di pioggia nell'archivio) come NaN
    hourly = hourly.reindex(columns=["time", *FORECAST_HOURLY])
    day_rows = hourly[hourly["time"].dt.date == day].reset_index(drop=True)
    if day_rows.empty:
        raise WeatherError("Nessun dato meteo per il giorno scelto.")

    start = datetime.combine(day, start_time)
    end = min(start + timedelta(seconds=duration_s), datetime.combine(day, time(23, 59)))
    window = day_rows[(day_rows["time"] >= start.replace(minute=0)) & (day_rows["time"] <= end)]
    if window.empty:
        window = day_rows.iloc[[int((day_rows["time"] - start).abs().idxmin())]]

    daily = pd.DataFrame(data.get("daily") or {})
    daily_day = daily[daily["time"] == day.isoformat()] if "time" in daily else daily.iloc[0:0]
    prev_rain = (daily.loc[daily["time"] < day.isoformat(), "precipitation_sum"].sum()
                 if "precipitation_sum" in daily else float("nan"))

    codes = window["weather_code"].dropna().astype(int)
    return WeatherInfo(
        source=source,
        day=day,
        start=start,
        end=end,
        elevation_m=float(data.get("elevation", float("nan"))),
        hourly=day_rows,
        temp_min=float(window["temperature_2m"].min()),
        temp_max=float(window["temperature_2m"].max()),
        feels_min=float(window["apparent_temperature"].min()),
        feels_max=float(window["apparent_temperature"].max()),
        rain_mm=float(window["precipitation"].fillna(0).sum()),
        rain_prob_max=float(window["precipitation_probability"].max()),
        wind_max=float(window["wind_speed_10m"].max()),
        gust_max=float(window["wind_gusts_10m"].max()),
        worst_code=int(codes.max()) if not codes.empty else -1,
        main_code=int(codes.mode().iloc[0]) if not codes.empty else -1,
        rain_prev_mm=float(prev_rain),
        sunrise=_parse_dt(daily_day, "sunrise"),
        sunset=_parse_dt(daily_day, "sunset"),
    )


def _parse_dt(daily_day: pd.DataFrame, column: str) -> datetime | None:
    if daily_day.empty or column not in daily_day or pd.isna(daily_day[column].iloc[0]):
        return None
    return pd.to_datetime(daily_day[column].iloc[0]).to_pydatetime()


def _gt(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value > threshold)


def _r(value: float, digits: int = 0):
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits) if digits else int(round(float(value)))
