"""
Consigli sul percorso generati da Gemini (API REST generateContent di Google AI Studio).

A Gemini vengono inviati solo dati aggregati del percorso (km, quote, fondo, pendenze),
nessuna coordinata. La risposta è JSON strutturato secondo RESPONSE_SCHEMA.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

import brouter_core as bc

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
API_KEY_URL = "https://aistudio.google.com/apikey"
DEFAULT_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")

ACTIVITIES = ("Trail running", "Camminata veloce", "Escursione")
LEVELS = ("Principiante", "Intermedio", "Esperto")
STRATEGY_ICONS = {"corri": "🏃", "cammina": "🚶", "alterna": "🔁", "scendi con cautela": "⚠️"}

MAX_CONTEXT_ROWS = 150

_KM_RANGE = {
    "type": "OBJECT",
    "properties": {
        "km_da": {"type": "NUMBER"},
        "km_a": {"type": "NUMBER"},
        "motivo": {"type": "STRING"},
    },
    "required": ["km_da", "km_a", "motivo"],
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sintesi": {"type": "STRING"},
        "salite_e_discese": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "km_da": {"type": "NUMBER"},
                    "km_a": {"type": "NUMBER"},
                    "strategia": {"type": "STRING", "enum": list(STRATEGY_ICONS)},
                    "motivo": {"type": "STRING"},
                },
                "required": ["km_da", "km_a", "strategia", "motivo"],
            },
        },
        "alimentazione": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"km": {"type": "NUMBER"}, "consiglio": {"type": "STRING"}},
                "required": ["km", "consiglio"],
            },
        },
        "pioggia": {
            "type": "OBJECT",
            "properties": {
                "scarpe_consigliate": {"type": "STRING"},
                "scarpe_da_evitare": {"type": "STRING"},
                "tratti_scivolosi": {"type": "ARRAY", "items": _KM_RANGE},
            },
            "required": ["scarpe_consigliate", "scarpe_da_evitare", "tratti_scivolosi"],
        },
        "meteo_e_abbigliamento": {"type": "STRING"},
        "altri_consigli": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["sintesi", "salite_e_discese", "alimentazione", "pioggia", "altri_consigli"],
}

SYSTEM_PROMPT = """Sei un allenatore esperto di trail running ed escursionismo.
Ricevi i dati di un percorso calcolato con BRouter su dati OpenStreetMap e dai consigli pratici in italiano.

Regole:
- Riferisci ogni consiglio ai km del percorso, con una cifra decimale.
- Usa solo i dati forniti. Non inventare tratti, quote o fondi che non ci sono.
- Il fondo con km_dedotti > 0 è ricavato dal tipo di strada e può essere impreciso: se è rilevante, dillo.
- Salite e discese: per ogni tratto in "salite_e_discese" dei dati scegli una strategia
  (corri, cammina, alterna, scendi con cautela) adatta all'attività e al livello indicati,
  considerando pendenza media, punte, lunghezza e fondo. Motivo in una frase.
- Alimentazione: stima prima la durata per l'attività e il livello indicati (scrivila nella sintesi).
  Per il trail running parti da tempo_trail_stimato (riferito a un livello intermedio) e adattalo al livello.
  Sotto i 60 minuti non proporre gel, al massimo acqua. Sopra, colloca gel e acqua a km precisi,
  preferibilmente prima delle salite più dure o in tratti facili dove è comodo farlo.
- Pioggia: consiglia il tipo di scarpa (tassellatura, mescola, drop, protezione) in base al mix di fondi
  e indica cosa evitare. Elenca i tratti che col bagnato diventano scivolosi o fangosi
  (lastricato/pavé, roccia, erba, terra battuta, legno, scalinate, bosco con foglie e radici,
  discese ripide), con km e motivo.
- Meteo: se nei dati c'è "meteo", usalo davvero. In "meteo_e_abbigliamento" indica abbigliamento e strati
  in base a temperatura percepita, vento e pioggia, e adegua l'idratazione al caldo. Se la pioggia è prevista
  o lo stato_terreno non è asciutto, la sezione pioggia descrive la situazione reale e non un'ipotesi.
  Se si rischia di finire dopo il tramonto consiglia la frontale; con temporali suggerisci di valutare
  di rimandare. Senza "meteo" lascia "meteo_e_abbigliamento" vuoto e tratta la pioggia come ipotesi.
- Tono diretto e concreto, niente premesse o disclaimer generici."""


class GeminiError(Exception):
    """Errore nella chiamata a Gemini. retry_other_model indica se ha senso provare un altro modello."""

    def __init__(self, message: str, retry_other_model: bool = False):
        super().__init__(message)
        self.retry_other_model = retry_other_model


@dataclass(frozen=True)
class CoachAdvice:
    model: str
    data: dict | None
    raw_text: str


def resolve_api_key(*candidates: str | None) -> str | None:
    """Prima chiave valorizzata tra quelle passate e le variabili GEMINI_API_KEY / GOOGLE_API_KEY."""
    sources = (*candidates, os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY"))
    return next((c.strip() for c in sources if c and c.strip()), None)


def models() -> tuple[str, ...]:
    """Modelli da provare in ordine; GEMINI_MODEL permette di forzarne uno."""
    override = os.getenv("GEMINI_MODEL")
    return tuple(dict.fromkeys(filter(None, (override, *DEFAULT_MODELS))))


# ----------------------------------------------------------------------------------------
# Contesto
# ----------------------------------------------------------------------------------------

def build_context(analysis: bc.RouteAnalysis, climbs: pd.DataFrame, weather: dict | None = None) -> dict:
    """Dati del percorso (e meteo, se disponibile) in forma compatta e serializzabile per il prompt."""
    s = analysis.summary
    surfaces = bc.breakdown(analysis.steps, "fondo")
    return {
        "totali": {
            "lunghezza_km": _r(s.length_m / 1000, 2),
            "dislivello_positivo_m": _r(s.ascend_m),
            "dislivello_negativo_m": _r(s.descend_m),
            "quota_min_m": _r(s.ele_min),
            "quota_max_m": _r(s.ele_max),
            "pendenza_max_pct": _r(s.max_grade_pct, 1),
            "tempo_brouter_a_piedi": bc.fmt_duration(s.total_time_s),
            "tempo_trail_stimato": bc.fmt_duration(bc.estimate_trail_time(analysis).seconds),
        },
        "fondo_percentuali": [
            {"fondo": r.fondo, "km": _r(r.km, 2), "pct": _r(r.pct, 1), "km_dedotti": _r(r.dedotto_m / 1000, 2)}
            for r in surfaces.itertuples()
        ],
        "salite_e_discese": [
            {"tipo": c.tipo, "km_da": _r(c.km_start, 1), "km_a": _r(c.km_end, 1),
             "quota_da_m": _r(c.ele_start), "quota_a_m": _r(c.ele_end), "dislivello_m": _r(c.dz),
             "pendenza_media_pct": _r(c.pendenza_media, 1), "pendenza_max_pct": _r(c.pendenza_max, 1),
             "fondo": c.fondo}
            for c in climbs.itertuples() if c.tipo != "piano"
        ],
        "tratti": _osm_stretches(analysis),
        "bosco": None if analysis.forest is None else {
            "pct_percorso": _r(analysis.forest.pct, 1),
            "tratti_km": [[_r(a, 1), _r(b, 1)] for a, b in analysis.forest.ranges],
        },
        **({"meteo": weather} if weather else {}),
    }


def _osm_stretches(analysis: bc.RouteAnalysis) -> list[dict]:
    # Unisco i tratti consecutivi con stesso surface OSM e stesso highway, per non perdere il dettaglio
    # (es. surface=wood o sett) che la categoria di fondo accorpa
    steps = analysis.steps[analysis.steps["dist_m"] > 0].assign(
        surface_osm=lambda d: d["tags"].map(lambda t: t.get("surface", "assente")),
        highway=lambda d: d["tags"].map(lambda t: t.get("highway", "")),
    )
    key = steps[["surface_osm", "highway"]]
    runs = steps.groupby(key.ne(key.shift()).any(axis=1).cumsum()).agg(
        km_start=("km_start", "first"), km_end=("km_end", "last"), fondo=("fondo", "first"),
        surface_osm=("surface_osm", "first"), via=("via", "first"), dz=("dz", "sum"),
        difficolta=("difficolta", lambda d: ", ".join(sorted({x for x in d if x}))),
    )
    if len(runs) > MAX_CONTEXT_ROWS:
        # Percorsi molto lunghi: ripiego sui segmenti già aggregati per fondo e tipo di via
        runs = analysis.segments.assign(surface_osm="vari")

    forest = analysis.forest
    return [
        {
            "km_da": _r(r.km_start, 2), "km_a": _r(r.km_end, 2), "fondo": r.fondo, "surface_osm": r.surface_osm,
            "tipo_via": r.via, "pendenza_pct": _r(r.dz / ((r.km_end - r.km_start) * 1000) * 100, 1)
            if r.km_end > r.km_start else 0,
            **({"bosco_pct": _r(bc.forest_share(forest.samples, r.km_start, r.km_end))} if forest else {}),
            **({"difficolta": r.difficolta} if r.difficolta else {}),
        }
        for r in runs.itertuples()
    ]


def _r(value, digits: int = 0):
    """Arrotondamento sicuro per JSON: NaN e infiniti diventano null."""
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits) if digits else int(round(float(value)))


# ----------------------------------------------------------------------------------------
# Chiamata
# ----------------------------------------------------------------------------------------

def ask_coach(api_key: str, context_json: str, activity: str, level: str) -> CoachAdvice:
    """
    Chiede i consigli a Gemini provando i modelli in ordine.

    @throws GeminiError se nessun modello risponde
    """
    if not api_key or not api_key.strip():
        raise GeminiError("Manca la chiave API di Gemini.")
    if activity not in ACTIVITIES or level not in LEVELS:
        raise GeminiError("Attività o livello non validi.")

    prompt = (f"Attività: {activity}\nLivello: {level}\n\n"
              f"Dati del percorso (JSON):\n{context_json}")
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": RESPONSE_SCHEMA},
    }

    session = bc.http_session()
    errors: list[str] = []
    # Se un modello non è disponibile o ha la quota esaurita passo al successivo
    for model in models():
        try:
            text = _generate(session, api_key.strip(), model, body)
            return CoachAdvice(model=model, data=_parse_json(text), raw_text=text)
        except GeminiError as exc:
            if not exc.retry_other_model:
                raise
            errors.append(f"{model}: {exc}")
    raise GeminiError("Nessun modello Gemini disponibile. " + " | ".join(errors))


def _generate(session: requests.Session, api_key: str, model: str, body: dict) -> str:
    try:
        resp = session.post(GEMINI_URL.format(model=model), json=body, timeout=120,
                            headers={"x-goog-api-key": api_key})
    except requests.RequestException as exc:
        raise GeminiError(f"Gemini non raggiungibile: {exc}") from exc

    if resp.status_code != 200:
        raise _http_error(resp)

    data = resp.json()
    block = data.get("promptFeedback", {}).get("blockReason")
    if block:
        raise GeminiError(f"Richiesta bloccata da Gemini ({block}).")

    candidates = data.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        reason = candidates[0].get("finishReason", "sconosciuto") if candidates else "nessun candidato"
        raise GeminiError(f"Gemini ha restituito una risposta vuota ({reason}).", retry_other_model=True)
    return text


def _http_error(resp: requests.Response) -> GeminiError:
    try:
        message = resp.json().get("error", {}).get("message", "")
    except ValueError:
        message = resp.text[:200]

    match resp.status_code:
        case 400 if "api key" in message.lower():
            return GeminiError("Chiave API Gemini non valida. Controllala su Google AI Studio.")
        case 401 | 403:
            return GeminiError(f"Accesso negato da Gemini ({resp.status_code}): {message}")
        case 404:
            return GeminiError("modello non disponibile per questa chiave", retry_other_model=True)
        case 429:
            return GeminiError("quota gratuita esaurita, riprova più tardi", retry_other_model=True)
        case 500 | 502 | 503 | 504:
            return GeminiError(f"servizio temporaneamente non disponibile ({resp.status_code})",
                               retry_other_model=True)
        case _:
            return GeminiError(f"Errore Gemini {resp.status_code}: {message}")


def _parse_json(text: str) -> dict | None:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def context_to_json(context: dict) -> str:
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"),
                      default=lambda o: o.item() if isinstance(o, np.generic) else str(o))