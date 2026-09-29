"""
Integrazione con BRouter (https://brouter.de) e analisi del fondo di un percorso.

Il routing è delegato all'API pubblica di BRouter: la risposta GeoJSON contiene nella
property "messages" la stessa tabella del pannello "Data" di BRouter-Web (una riga per
tratto omogeneo, con WayTags OSM, quota, distanza, costi, tempo ed energia).
Il bosco non è presente nei WayTags, quindi viene ricavato da OpenStreetMap: con una sola query
Overpass scarico i poligoni landuse=forest / natural=wood attorno al percorso e verifico in locale
quali punti della traccia ci cadono dentro.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import gpxpy
import numpy as np
import pandas as pd
import requests
import shapely
from requests.adapters import HTTPAdapter
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from urllib3.util.retry import Retry

BROUTER_API = "https://brouter.de/brouter"
# Istanze pubbliche Overpass: se una rifiuta la connessione o è sovraccarica passo alla successiva
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
USER_AGENT = "gpx-surface-analyzer/1.0 (+https://brouter.de)"

MAX_WAYPOINTS = 200
PASSTHROUGH_PARAMS = ("nogos", "polylines", "polygons", "straight")
# Unico profilo a piedi di brouter.de ("Hiking, Original" in BRouter-Web): usa sentieri e scalinate
FOOT_PROFILE = "hiking-mountain"

FOREST_SAMPLE_M = 50
FOREST_MAX_SAMPLES = 600
FOREST_BBOX_PAD_DEG = 0.002

EARTH_RADIUS_M = 6_371_008.8

# Categorie di fondo
ASFALTO = "Asfalto"
LASTRICATO = "Lastricato / pavé"
STERRATO = "Sterrato / ghiaia"
TERRA = "Terra battuta"
ERBA = "Erba / prato"
ROCCIA = "Roccia"
SCALINATA = "Scalinata"
SCONOSCIUTO = "Non specificato"

SURFACE_COLORS = {
    ASFALTO: "#5a6270",
    LASTRICATO: "#9b59b6",
    STERRATO: "#f39c12",
    TERRA: "#a0522d",
    ERBA: "#7cb342",
    ROCCIA: "#e74c3c",
    SCALINATA: "#e91e63",
    SCONOSCIUTO: "#bdc3c7",
}
FOREST_COLOR = "#2e7d32"

# Rilevamento salite/discese sul profilo altimetrico
CLIMB_THRESHOLD_M = 15.0
CLIMB_RESAMPLE_M = 20
CLIMB_SMOOTH_SAMPLES = 5
CLIMB_PEAK_WINDOW = 5
FLAT_TOLERANCE_M = 2.0
MIN_FLAT_M = 300

# Stima tempo trail: metà del tempo a piedi, corretta di ±10 min in base alle salite ripide
TRAIL_TIME_FACTOR = 0.5
STEEP_GRADE_PCT = 8.0
STEEP_NEUTRAL_M_PER_KM = 20.0
TRAIL_MAX_ADJUST_MIN = 10.0

_SURFACE_MAP = {
    **dict.fromkeys(("asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes",
                     "chipseal", "metal", "wood", "tartan", "rubber", "acrylic"), ASFALTO),
    **dict.fromkeys(("paving_stones", "sett", "cobblestone", "unhewn_cobblestone",
                     "cobblestone:flattened", "bricks", "stone"), LASTRICATO),
    **dict.fromkeys(("compacted", "fine_gravel", "gravel", "pebblestone", "unpaved", "shells"), STERRATO),
    **dict.fromkeys(("ground", "dirt", "earth", "mud", "sand", "soil", "woodchips"), TERRA),
    **dict.fromkeys(("grass", "grass_paver"), ERBA),
    **dict.fromkeys(("rock", "bare_rock"), ROCCIA),
}

_TRACKTYPE_MAP = {
    "grade1": ASFALTO,
    "grade2": STERRATO,
    "grade3": TERRA,
    "grade4": TERRA,
    "grade5": TERRA,
}

_PAVED_HIGHWAYS = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "service", "footway", "pedestrian", "cycleway",
})

_MAIN_ROADS = frozenset({"motorway", "trunk", "primary", "secondary", "tertiary"})

BROUTER_COLUMNS = ("Longitude", "Latitude", "Elevation", "Distance", "CostPerKm", "ElevCost",
                   "TurnCost", "NodeCost", "InitialCost", "WayTags", "NodeTags", "Time", "Energy")


class BRouterError(Exception):
    """Errore bloccante nel calcolo o nell'interpretazione del percorso."""


@dataclass(frozen=True)
class RouteRequest:
    """Punti di passaggio (lon, lat) e profilo da inviare a BRouter."""
    lonlats: tuple[tuple[float, float], ...]
    profile: str
    extra: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RouteSummary:
    """Totali del percorso, equivalenti alla barra inferiore di BRouter-Web."""
    profile: str
    length_m: float
    ascend_m: float
    plain_ascend_m: float
    descend_m: float
    total_time_s: float
    energy_kwh: float
    cost: float
    ele_min: float
    ele_max: float
    max_grade_pct: float

    @property
    def mean_cost_factor(self) -> float:
        return self.cost / self.length_m if self.length_m > 0 else float("nan")


@dataclass(frozen=True)
class ForestInfo:
    """Campioni lungo la traccia con flag bosco e tratti boscosi aggregati (km)."""
    samples: pd.DataFrame
    pct: float
    ranges: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RouteAnalysis:
    summary: RouteSummary
    track: pd.DataFrame
    steps: pd.DataFrame
    segments: pd.DataFrame
    forest: ForestInfo | None
    warnings: tuple[str, ...] = ()


# ----------------------------------------------------------------------------------------
# Input: link BRouter-Web e file GPX
# ----------------------------------------------------------------------------------------

def parse_brouter_url(url: str) -> RouteRequest:
    """
    Estrae lonlats e parametri opzionali da un link BRouter-Web (#...) o API (?...).

    @throws ValueError se il link non contiene almeno due punti validi
    """
    if not url or not url.strip():
        raise ValueError("Incolla un link BRouter-Web.")

    parsed = urlparse(url.strip())
    params = next(
        (p for p in (_flat_qs(parsed.fragment), _flat_qs(parsed.query)) if "lonlats" in p),
        None,
    )
    if params is None:
        raise ValueError("Il link non contiene il parametro 'lonlats'.")

    lonlats = tuple(_parse_lonlat(chunk) for chunk in re.split(r"[;|]", params["lonlats"]) if chunk.strip())
    if len(lonlats) < 2:
        raise ValueError("Servono almeno due punti nel parametro 'lonlats'.")

    # BRouter-Web separa gli elementi con ';', l'API con '|'
    extra = tuple((k, params[k].replace(";", "|")) for k in PASSTHROUGH_PARAMS if params.get(k))
    # Il profilo del link viene ignorato: il tool analizza sempre il percorso a piedi
    return RouteRequest(lonlats=lonlats, profile=FOOT_PROFILE, extra=extra)


def request_from_gpx(content: bytes, spacing_m: float) -> RouteRequest:
    """
    Converte una traccia GPX in punti di aggancio per BRouter, campionati ogni spacing_m metri.

    @throws ValueError se il file non è un GPX valido o non contiene punti
    """
    if not content:
        raise ValueError("Il file GPX è vuoto.")
    if spacing_m <= 0:
        raise ValueError("La distanza tra i punti di aggancio deve essere positiva.")

    try:
        gpx = gpxpy.parse(content.decode("utf-8", errors="replace"))
    except Exception as exc:  # gpxpy solleva tipi diversi a seconda del problema
        raise ValueError(f"GPX non leggibile: {exc}") from exc

    # La traccia registrata ha priorità; in alternativa si usano i punti di rotta
    points = [(p.longitude, p.latitude) for t in gpx.tracks for s in t.segments for p in s.points] \
        or [(p.longitude, p.latitude) for r in gpx.routes for p in r.points]
    if len(points) < 2:
        raise ValueError("Il GPX non contiene una traccia o una rotta con almeno due punti.")

    return RouteRequest(lonlats=_subsample(np.asarray(points, dtype=float), spacing_m), profile=FOOT_PROFILE)


def _flat_qs(raw: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(raw).items()}


def _parse_lonlat(chunk: str) -> tuple[float, float]:
    try:
        lon, lat = (float(v) for v in chunk.split(",")[:2])
    except ValueError as exc:
        raise ValueError(f"Coordinata non valida nel link: '{chunk}'") from exc
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"Coordinata fuori range: '{chunk}'")
    return lon, lat


def _subsample(points: np.ndarray, spacing_m: float) -> tuple[tuple[float, float], ...]:
    cum = _cumulative_m(points[:, 0], points[:, 1])
    # Oltre MAX_WAYPOINTS allargo la spaziatura per restare entro i limiti di URL e server
    spacing = max(spacing_m, cum[-1] / (MAX_WAYPOINTS - 1))
    bins = np.floor(cum / spacing)
    keep = np.r_[True, np.diff(bins) > 0]
    keep[-1] = True
    return tuple((float(lon), float(lat)) for lon, lat in points[keep])


def _haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _cumulative_m(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(_haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]))]


# ----------------------------------------------------------------------------------------
# Chiamate esterne
# ----------------------------------------------------------------------------------------

def http_session(retries: int = 3) -> requests.Session:
    # Retry con backoff esponenziale sugli errori transitori e sul rate limit
    retry = Retry(total=retries, backoff_factor=1.5, status_forcelist=(429, 502, 503, 504),
                  allowed_methods=("GET", "POST"), raise_on_status=False)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_route(req: RouteRequest) -> dict:
    """
    Calcola il percorso con BRouter e restituisce il GeoJSON grezzo.

    @throws BRouterError se il server non risponde o non trova un percorso
    """
    if len(req.lonlats) < 2:
        raise BRouterError("Servono almeno due punti per calcolare un percorso.")

    params = {
        "lonlats": "|".join(f"{lon:.6f},{lat:.6f}" for lon, lat in req.lonlats),
        "profile": req.profile,
        "alternativeidx": "0",
        "format": "geojson",
        **dict(req.extra),
    }
    try:
        resp = http_session().get(BROUTER_API, params=params, timeout=120)
    except requests.RequestException as exc:
        raise BRouterError(f"BRouter non raggiungibile: {exc}") from exc

    if resp.status_code == 429:
        raise BRouterError("BRouter ha limitato le richieste (429). Riprova tra qualche minuto.")
    if resp.status_code != 200:
        raise BRouterError(f"BRouter ha risposto {resp.status_code}: {resp.text.strip()[:300]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise BRouterError(f"Risposta BRouter non valida: {resp.text.strip()[:300]}") from exc


def detect_forest(track: pd.DataFrame) -> ForestInfo:
    """
    Campiona la traccia ogni ~50 m e verifica quali punti cadono in un'area boscata OSM.

    @throws BRouterError se nessuna istanza Overpass risponde
    """
    total_km = float(track["km"].iloc[-1])
    spacing_km = max(FOREST_SAMPLE_M, total_km * 1000 / FOREST_MAX_SAMPLES) / 1000
    km = np.arange(spacing_km / 2, total_km, spacing_km) if total_km > spacing_km else np.array([total_km / 2])
    samples = pd.DataFrame({
        "km": km,
        "lon": np.interp(km, track["km"], track["lon"]),
        "lat": np.interp(km, track["km"], track["lat"]),
    })

    pad = FOREST_BBOX_PAD_DEG
    bbox = (track["lat"].min() - pad, track["lon"].min() - pad, track["lat"].max() + pad, track["lon"].max() + pad)
    forest = forest_geometry(_fetch_forest_elements(bbox))
    if forest is None:
        flags = np.zeros(len(samples), dtype=bool)
    else:
        shapely.prepare(forest)
        flags = shapely.contains_xy(forest, samples["lon"].to_numpy(), samples["lat"].to_numpy())
    samples = samples.assign(bosco=flags)

    # Raggruppo i campioni consecutivi in bosco in intervalli chilometrici
    edges = np.diff(np.r_[0, flags.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    half = spacing_km / 2
    ranges = tuple(
        (max(0.0, float(km[s]) - half), min(total_km, float(km[e - 1]) + half))
        for s, e in zip(starts, ends)
    )
    return ForestInfo(samples=samples, pct=float(flags.mean() * 100), ranges=ranges)


def _fetch_forest_elements(bbox: tuple[float, float, float, float]) -> list[dict]:
    south, west, north, east = bbox
    area = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    query = (
        "[out:json][timeout:90];("
        f'way["landuse"="forest"]{area};way["natural"="wood"]{area};'
        f'relation["landuse"="forest"]{area};relation["natural"="wood"]{area};'
        ");out geom;"
    )
    # Pochi retry per istanza: se una è giù conviene passare subito alla successiva
    session = http_session(retries=1)
    errors: list[str] = []
    for url in OVERPASS_ENDPOINTS:
        try:
            resp = session.post(url, data={"data": query}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{urlparse(url).hostname}: {type(exc).__name__}")
            continue
        # Overpass segnala timeout e sovraccarico in "remark" con risposta 200
        remark = data.get("remark", "")
        if "error" in remark.lower():
            errors.append(f"{urlparse(url).hostname}: {remark[:80]}")
            continue
        return data.get("elements", [])
    raise BRouterError("nessuna istanza Overpass disponibile (" + "; ".join(errors) + ")")


def forest_geometry(elements: list[dict]):
    """
    Unione dei poligoni boscati da una risposta Overpass "out geom", o None se non ce ne sono.

    Le way chiuse sono poligoni diretti; per le relazioni multipolygon ricompongo gli anelli
    outer/inner con polygonize, perché ogni anello può essere spezzato su più way.
    """
    def line(points: list[dict]) -> list[tuple[float, float]]:
        return [(p["lon"], p["lat"]) for p in points]

    def rings(members: list[dict], role: str):
        lines = [LineString(line(m["geometry"])) for m in members
                 if m.get("type") == "way" and m.get("role", "outer") in role and len(m.get("geometry", [])) >= 2]
        return unary_union(list(polygonize(unary_union(lines)))) if lines else None

    closed_ways = [
        Polygon(line(e["geometry"])) for e in elements
        if e.get("type") == "way" and len(e.get("geometry", [])) >= 4 and e["geometry"][0] == e["geometry"][-1]
    ]

    def relation_polygon(rel: dict):
        outer = rings(rel.get("members", []), ("outer", ""))
        if outer is None or outer.is_empty:
            return None
        inner = rings(rel.get("members", []), ("inner",))
        return outer.difference(inner) if inner is not None and not inner.is_empty else outer

    relations = [relation_polygon(e) for e in elements if e.get("type") == "relation"]
    # buffer(0) ripara i poligoni OSM non validi (autointersezioni) prima dell'unione
    polygons = [p.buffer(0) for p in (*closed_ways, *relations) if p is not None and not p.is_empty]
    return unary_union(polygons) if polygons else None


# ----------------------------------------------------------------------------------------
# Classificazione WayTags
# ----------------------------------------------------------------------------------------

def parse_waytags(raw: str) -> dict[str, str]:
    """'highway=track surface=gravel' -> {'highway': 'track', 'surface': 'gravel'}"""
    return dict(tok.split("=", 1) for tok in (raw or "").split() if "=" in tok)


def classify_surface(tags: dict[str, str]) -> tuple[str, bool]:
    """
    Categoria di fondo e flag "dedotto" (True se manca il tag surface
    e la categoria è ricavata dal tipo di strada).
    """
    highway = tags.get("highway", "")
    surface = tags.get("surface", "").split(";")[0]

    if highway == "steps":
        return SCALINATA, False
    if surface:
        return _SURFACE_MAP.get(surface, SCONOSCIUTO), False

    match highway:
        case "track":
            return _TRACKTYPE_MAP.get(tags.get("tracktype", ""), STERRATO), True
        case "path" | "bridleway":
            return TERRA, True
        case h if h in _PAVED_HIGHWAYS or h.endswith("_link"):
            return ASFALTO, True
        case _:
            return SCONOSCIUTO, False


def classify_way(tags: dict[str, str]) -> str:
    """Tipologia della via, raggruppata in modo leggibile."""
    if tags.get("route") == "ferry":
        return "Traghetto"
    match tags.get("highway", ""):
        case h if h in _MAIN_ROADS or h.endswith("_link"):
            return "Strada principale"
        case "unclassified":
            return "Strada secondaria"
        case "residential" | "living_street" | "service":
            return "Strada locale"
        case "track":
            return "Carrareccia"
        case "path" | "bridleway":
            return "Sentiero"
        case "footway" | "pedestrian":
            return "Pedonale"
        case "cycleway":
            return "Ciclabile"
        case "steps":
            return "Scalinata"
        case _:
            return "Altro"


def _difficulty(tags: dict[str, str]) -> str:
    return " ".join(f"{k}={tags[k]}" for k in ("sac_scale", "mtb:scale", "trail_visibility") if k in tags)


# ----------------------------------------------------------------------------------------
# Parsing risposta e aggregazioni
# ----------------------------------------------------------------------------------------

def analyze(req: RouteRequest, with_forest: bool = True) -> RouteAnalysis:
    """Calcola il percorso, classifica i tratti e, se richiesto, rileva il bosco."""
    summary, track, steps = parse_route(fetch_route(req), req.profile)
    segments = merge_segments(steps)

    forest, warnings = None, ()
    if with_forest:
        try:
            forest = detect_forest(track)
            segments = segments.assign(bosco_pct=[
                forest_share(forest.samples, s, e) for s, e in zip(segments["km_start"], segments["km_end"])
            ])
        except BRouterError as exc:
            warnings = (f"Rilevamento bosco non riuscito: {exc}",)

    return RouteAnalysis(summary, track, steps, segments, forest, warnings)


def parse_route(geojson: dict, profile: str) -> tuple[RouteSummary, pd.DataFrame, pd.DataFrame]:
    """
    Converte il GeoJSON di BRouter in totali, geometria densa e tabella tratti.

    @throws BRouterError se la risposta non ha la struttura attesa
    """
    features = (geojson or {}).get("features") or []
    if not features:
        raise BRouterError("La risposta di BRouter non contiene alcun percorso.")
    feature = features[0]
    props = feature.get("properties", {})

    coords = np.asarray(feature.get("geometry", {}).get("coordinates", []), dtype=float)
    if coords.ndim != 2 or len(coords) < 2:
        raise BRouterError("Geometria del percorso mancante o incompleta.")
    lon, lat = coords[:, 0], coords[:, 1]
    ele = coords[:, 2] if coords.shape[1] > 2 else np.full(len(coords), np.nan)

    messages = props.get("messages") or []
    if len(messages) < 2:
        raise BRouterError("La risposta non contiene la tabella dati (WayTags). Profilo non compatibile?")

    geo_len = _cumulative_m(lon, lat)
    length_m = _to_float(props.get("track-length"), geo_len[-1])
    scale = length_m / geo_len[-1] if geo_len[-1] > 0 else 1.0
    track = pd.DataFrame({"lon": lon, "lat": lat, "ele": ele, "km": geo_len * scale / 1000})

    steps = _build_steps(messages, length_m, _first_finite(ele))

    ele_start, ele_end = _first_finite(ele), _first_finite(ele[::-1])
    ascend = _to_float(props.get("filtered ascend"))
    long_rows = steps[steps["dist_m"] >= 50]
    summary = RouteSummary(
        profile=profile,
        length_m=length_m,
        ascend_m=ascend,
        plain_ascend_m=_to_float(props.get("plain-ascend")),
        # La discesa filtrata si ricava dalla salita filtrata e dal delta quota
        descend_m=ascend - (ele_end - ele_start),
        total_time_s=_to_float(props.get("total-time")),
        energy_kwh=_to_float(props.get("total-energy")) / 3.6e6,
        cost=_to_float(props.get("cost")),
        ele_min=float(np.nanmin(ele)) if np.isfinite(ele).any() else float("nan"),
        ele_max=float(np.nanmax(ele)) if np.isfinite(ele).any() else float("nan"),
        max_grade_pct=float(long_rows["grade_pct"].abs().max()) if not long_rows.empty else float("nan"),
    )
    return summary, track, steps


def _build_steps(messages: list[list], length_m: float, ele_start: float) -> pd.DataFrame:
    header, rows = messages[0], messages[1:]
    steps = pd.DataFrame(rows, columns=header)
    numeric = [c for c in steps.columns if c not in ("WayTags", "NodeTags")]
    steps[numeric] = steps[numeric].apply(pd.to_numeric, errors="coerce")
    steps[["Longitude", "Latitude"]] = steps[["Longitude", "Latitude"]] / 1e6

    dist = steps["Distance"].fillna(0).to_numpy(dtype=float)
    # Difesa: se la colonna risultasse cumulativa la riporto a distanza per tratto
    looks_cumulative = (len(dist) > 1 and np.all(np.diff(dist) >= 0)
                        and abs(dist[-1] - length_m) < 0.02 * length_m
                        and abs(dist.sum() - length_m) > 0.05 * length_m)
    if looks_cumulative:
        dist = np.diff(dist, prepend=0.0)

    tags = steps["WayTags"].fillna("").map(parse_waytags)
    classified = tags.map(classify_surface)
    prev_ele = steps["Elevation"].shift(fill_value=ele_start)
    dz = (steps["Elevation"] - prev_ele).fillna(0)

    return steps.assign(
        dist_m=dist,
        km_end=np.cumsum(dist) / 1000,
        km_start=(np.cumsum(dist) - dist) / 1000,
        dz=dz,
        grade_pct=np.where(dist > 0, dz / np.where(dist > 0, dist, 1) * 100, 0.0),
        tags=tags,
        fondo=classified.map(lambda c: c[0]),
        dedotto=classified.map(lambda c: c[1]),
        via=tags.map(classify_way),
        difficolta=tags.map(_difficulty),
    )


def merge_segments(steps: pd.DataFrame) -> pd.DataFrame:
    """Unisce i tratti consecutivi con stesso fondo e stesso tipo di via."""
    rows = steps[steps["dist_m"] > 0]
    key = rows[["fondo", "via"]]
    run = key.ne(key.shift()).any(axis=1).cumsum()

    seg = rows.groupby(run).agg(
        km_start=("km_start", "first"),
        km_end=("km_end", "last"),
        fondo=("fondo", "first"),
        via=("via", "first"),
        dedotto=("dedotto", "any"),
        salita_m=("dz", lambda s: s.clip(lower=0).sum()),
        discesa_m=("dz", lambda s: -s.clip(upper=0).sum()),
        dz=("dz", "sum"),
        quota_m=("Elevation", "last"),
        difficolta=("difficolta", lambda s: ", ".join(sorted({d for d in s if d}))),
    ).reset_index(drop=True)

    length = (seg["km_end"] - seg["km_start"]) * 1000
    return seg.assign(
        lunghezza_m=length,
        pendenza_pct=np.where(length > 0, seg["dz"] / length.where(length > 0, 1) * 100, 0.0),
    )


def breakdown(steps: pd.DataFrame, column: str) -> pd.DataFrame:
    """Metri, km e percentuale per categoria; per il fondo anche la quota dedotta."""
    total = steps["dist_m"].sum()
    grouped = steps.groupby(column)
    out = pd.DataFrame({
        "metri": grouped["dist_m"].sum(),
        "dedotto_m": steps["dist_m"].where(steps["dedotto"], 0).groupby(steps[column]).sum(),
    })
    return (out.assign(km=out["metri"] / 1000, pct=out["metri"] / total * 100 if total else 0.0)
               .sort_values("metri", ascending=False)
               .reset_index())


def track_slice(track: pd.DataFrame, km_start: float, km_end: float) -> pd.DataFrame:
    """Punti della geometria nel range, inclusi i punti di confine per evitare buchi."""
    km = track["km"].to_numpy()
    i0 = max(0, int(np.searchsorted(km, km_start, side="right")) - 1)
    i1 = min(len(km), int(np.searchsorted(km, km_end, side="left")) + 1)
    return track.iloc[i0:i1]


def forest_share(samples: pd.DataFrame, km_start: float, km_end: float) -> float:
    mask = (samples["km"] >= km_start) & (samples["km"] < km_end)
    if mask.any():
        return float(samples.loc[mask, "bosco"].mean() * 100)
    # Tratto più corto della spaziatura: uso il campione più vicino al centro
    nearest = (samples["km"] - (km_start + km_end) / 2).abs().idxmin()
    return 100.0 if samples.loc[nearest, "bosco"] else 0.0


def _to_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else float("nan")


# ----------------------------------------------------------------------------------------
# Salite e discese
# ----------------------------------------------------------------------------------------

def detect_climbs(track: pd.DataFrame, steps: pd.DataFrame,
                  threshold_m: float = CLIMB_THRESHOLD_M) -> pd.DataFrame:
    """
    Divide il profilo in salite, discese e tratti pianeggianti.

    Le oscillazioni inferiori a threshold_m vengono ignorate (isteresi), così i
    piccoli saliscendi non spezzano una salita lunga.
    """
    if threshold_m <= 0:
        raise ValueError("La soglia delle salite deve essere positiva.")

    valid = track.dropna(subset=["ele"])
    if len(valid) < 2:
        return pd.DataFrame(columns=_CLIMB_COLUMNS)

    total_km = float(valid["km"].iloc[-1])
    km = np.unique(np.r_[np.arange(0, total_km, CLIMB_RESAMPLE_M / 1000), total_km])
    raw = np.interp(km, valid["km"], valid["ele"])
    # Media mobile su ~100 m per non scambiare il rumore del DEM per una salita
    smooth = pd.Series(raw).rolling(CLIMB_SMOOTH_SAMPLES, center=True, min_periods=1).mean().to_numpy()

    pivots = _pivots(smooth, threshold_m)
    moves = [_trim(smooth, a, b) for a, b in zip(pivots[:-1], pivots[1:])
             if abs(smooth[b] - smooth[a]) >= threshold_m]

    # I buchi tra un movimento e l'altro sono tratti pianeggianti o leggermente ondulati
    bounds = [0, *(i for move in moves for i in move), len(km) - 1]
    flats = [(a, b) for a, b in zip(bounds[::2], bounds[1::2]) if (km[b] - km[a]) * 1000 >= MIN_FLAT_M]

    stretches = sorted([(a, b, True) for a, b in moves] + [(a, b, False) for a, b in flats])
    return pd.DataFrame([_describe_climb(km, raw, smooth, steps, a, b, is_move) for a, b, is_move in stretches],
                        columns=_CLIMB_COLUMNS)


_CLIMB_COLUMNS = ["tipo", "km_start", "km_end", "ele_start", "ele_end", "ele_min", "ele_max",
                  "dz", "lunghezza_m", "pendenza_media", "pendenza_max", "fondo"]


def _pivots(ele: np.ndarray, threshold: float) -> list[int]:
    """Indici di picchi e valli con isteresi: un'inversione conta solo se supera la soglia."""
    pivots, trend, ext, lo, hi = [0], 0, 0, 0, 0
    for i in range(1, len(ele)):
        z = ele[i]
        if trend == 0:
            # Prima del primo movimento significativo tengo traccia di minimo e massimo
            lo, hi = (i if z < ele[lo] else lo), (i if z > ele[hi] else hi)
            if z - ele[lo] >= threshold:
                pivots, trend, ext = [0, lo] if lo else [0], 1, i
            elif ele[hi] - z >= threshold:
                pivots, trend, ext = [0, hi] if hi else [0], -1, i
        elif (z - ele[ext]) * trend >= 0:
            ext = i
        elif (ele[ext] - z) * trend >= threshold:
            pivots.append(ext)
            trend, ext = -trend, i

    tail = [ext] if trend and ext != pivots[-1] else []
    last = len(ele) - 1
    return pivots + tail + ([last] if (tail or pivots)[-1] != last else [])


def _trim(ele: np.ndarray, a: int, b: int) -> tuple[int, int]:
    """Toglie da inizio e fine del movimento le parti piatte (entro FLAT_TOLERANCE_M)."""
    seg = ele[a:b + 1] * (1 if ele[b] >= ele[a] else -1)
    start = a + int(np.flatnonzero(seg <= seg[0] + FLAT_TOLERANCE_M).max())
    end = a + int(np.flatnonzero(seg >= seg[-1] - FLAT_TOLERANCE_M).min())
    return (start, end) if end > start else (a, b)


def _describe_climb(km, raw, smooth, steps, a: int, b: int, is_move: bool) -> list:
    length_m = (km[b] - km[a]) * 1000
    dz = raw[b] - raw[a]
    avg = dz / length_m * 100 if length_m > 0 else 0.0

    # Pendenza massima su finestre di ~100 m del profilo lisciato
    w = CLIMB_PEAK_WINDOW
    grades = (smooth[a + w:b + 1] - smooth[a:b + 1 - w]) / ((km[a + w:b + 1] - km[a:b + 1 - w]) * 1000) * 100 \
        if b - a > w else np.array([avg])
    peak = float(np.max(grades) if dz >= 0 else -np.min(grades))

    kind = ("salita" if dz > 0 else "discesa") if is_move else "piano"
    part = raw[a:b + 1]
    return [kind, float(km[a]), float(km[b]), float(raw[a]), float(raw[b]), float(part.min()), float(part.max()),
            float(dz), float(length_m), float(abs(avg)), max(peak, abs(avg)), _dominant_surface(steps, km[a], km[b])]


def _dominant_surface(steps: pd.DataFrame, km_start: float, km_end: float) -> str:
    """Fondo prevalente nel range; se nessuno supera il 70% cita i due principali."""
    overlap = (np.minimum(steps["km_end"], km_end) - np.maximum(steps["km_start"], km_start)).clip(lower=0)
    by_surface = overlap.groupby(steps["fondo"]).sum().sort_values(ascending=False)
    total = by_surface.sum()
    if total <= 0:
        return SCONOSCIUTO.lower()
    shares = by_surface / total * 100
    if shares.iloc[0] >= 70 or len(shares) == 1:
        return shares.index[0].lower()
    return " e ".join(f"{name.lower()} ({fmt_num(pct)}%)" for name, pct in shares.iloc[:2].items())



@dataclass(frozen=True)
class TrailEstimate:
    """Tempo di corsa stimato: base = camminata x TRAIL_TIME_FACTOR, più la correzione per le salite."""
    seconds: float
    base_seconds: float
    adjust_min: float
    steep_ascent_m: float


def estimate_trail_time(analysis: RouteAnalysis) -> TrailEstimate:
    """
    Metà del tempo a piedi di BRouter, corretta tra -10 e +10 minuti.

    La correzione dipende dal dislivello fatto su salite con pendenza media >= STEEP_GRADE_PCT,
    rapportato ai km: 0 m/km -> -10 min, 20 m/km -> 0, da 40 m/km in su -> +10 min.
    """
    s = analysis.summary
    base = s.total_time_s * TRAIL_TIME_FACTOR if np.isfinite(s.total_time_s) else float("nan")

    # Soglia fissa: la stima non deve cambiare muovendo lo slider delle salite nella UI
    climbs = detect_climbs(analysis.track, analysis.steps, CLIMB_THRESHOLD_M)
    steep = climbs[(climbs["tipo"] == "salita") & (climbs["pendenza_media"] >= STEEP_GRADE_PCT)]
    steep_m = float(steep["dz"].sum()) if not steep.empty else 0.0

    km = s.length_m / 1000
    ratio = (steep_m / km - STEEP_NEUTRAL_M_PER_KM) / STEEP_NEUTRAL_M_PER_KM if km > 0 else -1.0
    adjust_min = round(float(np.clip(ratio, -1, 1)) * TRAIL_MAX_ADJUST_MIN)
    return TrailEstimate(seconds=max(base + adjust_min * 60, 0.0) if np.isfinite(base) else base,
                         base_seconds=base, adjust_min=adjust_min, steep_ascent_m=steep_m)


# ----------------------------------------------------------------------------------------
# Testo riepilogativo
# ----------------------------------------------------------------------------------------

def fmt_num(value: float, decimals: int = 0) -> str:
    """Formato italiano: 21.925 / 13,8"""
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:,.{decimals}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "–"
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}:{minutes % 60:02d} h"


def fmt_ranges(ranges: tuple[tuple[float, float], ...], limit: int = 6) -> str:
    shown = ", ".join(f"km {fmt_num(a, 1)}–{fmt_num(b, 1)}" for a, b in ranges[:limit])
    return shown + (f" e altri {len(ranges) - limit} tratti" if len(ranges) > limit else "")


def _fmt_length(meters: float) -> str:
    return f"{fmt_num(round(meters, -1))} m" if meters < 1000 else f"{fmt_num(meters / 1000, 1)} km"


def _intensity(grade_pct: float) -> str:
    if grade_pct < 3:
        return "leggera"
    if grade_pct < 6:
        return "moderata"
    if grade_pct < 10:
        return "impegnativa"
    return "ripida"


def describe_stretch(row) -> str:
    """Frase in italiano per un tratto restituito da detect_climbs (riga namedtuple)."""
    head = f"**Km {fmt_num(row.km_start, 1)} → {fmt_num(row.km_end, 1)}"
    # Dislivello calcolato sulle quote arrotondate, così il testo torna sempre con i numeri mostrati
    gain = abs(round(row.ele_end) - round(row.ele_start))
    length = _fmt_length(row.lunghezza_m)
    peak = (f", con punte del {fmt_num(row.pendenza_max)}%"
            if row.pendenza_max - row.pendenza_media >= 1.5 else "")

    match row.tipo:
        case "salita":
            body = (f" · salita {_intensity(row.pendenza_media)}.** Passerai da {fmt_num(row.ele_start)} "
                    f"a {fmt_num(row.ele_end)} m di quota, guadagnando {gain} m in {length}: "
                    f"pendenza media del {fmt_num(row.pendenza_media, 1)}%{peak}.")
        case "discesa":
            body = (f" · discesa {_intensity(row.pendenza_media)}.** Scenderai da {fmt_num(row.ele_start)} "
                    f"a {fmt_num(row.ele_end)} m di quota, perdendo {gain} m in {length}: "
                    f"pendenza media del {fmt_num(row.pendenza_media, 1)}%{peak}.")
        case _ if row.ele_max - row.ele_min <= 5:
            body = (f" · tratto pianeggiante.** Per {length} resterai intorno ai "
                    f"{fmt_num((row.ele_min + row.ele_max) / 2)} m di quota.")
        case _:
            body = (f" · tratto ondulato.** Per {length} piccoli saliscendi tra "
                    f"{fmt_num(row.ele_min)} e {fmt_num(row.ele_max)} m, senza dislivelli rilevanti.")

    return f"{head}{body} Fondo: {row.fondo}."


def describe_climbs(climbs: pd.DataFrame) -> list[str]:
    return [describe_stretch(row) for row in climbs.itertuples()]


def narrative(analysis: RouteAnalysis, climb_threshold_m: float = CLIMB_THRESHOLD_M) -> str:
    """Riepilogo in linguaggio naturale (markdown)."""
    s = analysis.summary
    surfaces = breakdown(analysis.steps, "fondo")
    inferred_pct = surfaces["dedotto_m"].sum() / surfaces["metri"].sum() * 100 if surfaces["metri"].sum() else 0

    parts = [
        f"Il percorso è lungo **{fmt_num(s.length_m / 1000, 1)} km**, con **{fmt_num(s.ascend_m)} m** "
        f"di dislivello positivo e **{fmt_num(s.descend_m)} m** negativo. "
        f"La quota varia tra {fmt_num(s.ele_min)} e {fmt_num(s.ele_max)} m, "
        f"con pendenza massima intorno al {fmt_num(s.max_grade_pct)}%. "
        f"Tempo stimato: **{fmt_duration(s.total_time_s)}** camminando, "
        f"circa **{fmt_duration(estimate_trail_time(analysis).seconds)}** di corsa trail.",
        "Fondo: " + ", ".join(
            f"{r.fondo.lower()} **{fmt_num(r.pct)}%** ({fmt_num(r.km, 1)} km)" for r in surfaces.itertuples()
        ) + ".",
    ]
    ascents = detect_climbs(analysis.track, analysis.steps, climb_threshold_m).query("tipo == 'salita'")
    if not ascents.empty:
        # La più dura pesa dislivello e pendenza insieme
        hardest = ascents.loc[(ascents["dz"] * ascents["pendenza_media"]).idxmax()]
        count = "una salita" if len(ascents) == 1 else f"{len(ascents)} salite"
        parts.append(
            f"Incontrerai {count}; la più dura va dal km {fmt_num(hardest.km_start, 1)} al km "
            f"{fmt_num(hardest.km_end, 1)} (+{fmt_num(hardest.dz)} m, "
            f"{fmt_num(hardest.pendenza_media, 1)}% di media)."
        )
    if inferred_pct >= 1:
        parts.append(f"Il {fmt_num(inferred_pct)}% del fondo è dedotto dal tipo di strada perché su OSM "
                     f"manca il tag surface.")
    if analysis.forest is not None:
        forest = analysis.forest
        parts.append(
            f"Nel bosco: circa **{fmt_num(forest.pct)}%** del percorso"
            + (f", in particolare {fmt_ranges(forest.ranges)}." if forest.ranges else ".")
        )
    return "\n\n".join(parts)