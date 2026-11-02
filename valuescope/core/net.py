"""Accès réseau robuste : session « navigateur », tentatives avec attente progressive, journal des erreurs par source.

Les serveurs partagés (Streamlit Community Cloud, AWS, GCP…) sont souvent limités par Yahoo Finance (HTTP 429)
ou filtrés par les protections anti-robots. On utilise donc :
  • curl_cffi (empreinte TLS d'un vrai Chrome) quand il est installé, sinon requests avec des en-têtes de navigateur ;
  • plusieurs tentatives espacées ;
  • des sources de repli (API « chart » de Yahoo, SEC EDGAR, Banque mondiale)."""
from __future__ import annotations
import time
from datetime import datetime

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "application/json,text/plain,*/*", "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}

# Journal : source -> (succès, message, horodatage)
LOG: dict[str, tuple[bool, str, str]] = {}

_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        try:
            from curl_cffi import requests as creq
            _SESSION = creq.Session(impersonate="chrome")
        except Exception:
            import requests
            _SESSION = requests.Session()
        try:
            _SESSION.headers.update(HEADERS)
        except Exception:
            pass
    return _SESSION


def record(source: str, ok: bool, msg: str = ""):
    LOG[source] = (ok, msg[:400], datetime.now().strftime("%H:%M:%S"))


class SourceError(RuntimeError):
    pass


_BREAKER: dict[str, list] = {}   # source -> [échecs consécutifs, horodatage du dernier échec]


def get(url: str, source: str, params: dict | None = None, headers: dict | None = None, timeout: int = 25,
        retries: int = 3, expect_json: bool = True):
    """GET avec tentatives (attente 2 s puis 5 s) sur 429/5xx et erreurs réseau.
    Coupe-circuit : après 3 échecs consécutifs d'une source, elle est sautée pendant 90 s (évite d'attendre
    des minutes quand une source est bloquée)."""
    br = _BREAKER.setdefault(source, [0, 0.0])
    if br[0] >= 3 and time.time() - br[1] < 90:
        raise SourceError(f"{source} : suspendue 90 s après échecs répétés — {LOG.get(source, (0, '', ''))[1]}")
    last = ""
    for i in range(retries):
        try:
            r = session().get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=timeout)
            code = r.status_code
            if code == 200:
                data = r.json() if expect_json else r.content
                record(source, True, "OK")
                _BREAKER[source] = [0, 0.0]
                return data
            last = f"HTTP {code} — {url.split('?')[0][:90]}"
            if code in (400, 401, 403, 404):
                break                                   # inutile d'insister
        except Exception as e:  # réseau, TLS, JSON invalide
            last = f"{type(e).__name__}: {e}"
        if i < retries - 1:
            time.sleep([2, 5][min(i, 1)])
    record(source, False, last)
    _BREAKER[source] = [br[0] + 1, time.time()]
    raise SourceError(f"{source} : {last}")


def explain(err: Exception | str) -> str:
    """Traduit une erreur technique en conseil compréhensible."""
    t = str(err).lower()
    if "429" in t or "rate" in t or "too many" in t:
        return ("La source limite le nombre de requêtes depuis ce serveur (HTTP 429, fréquent sur Streamlit Cloud avec Yahoo Finance). "
                "Patientez quelques minutes, chargez moins de sociétés à la fois, ou passez en mode « Hors-ligne » avec la base CSV.")
    if "403" in t or "forbidden" in t:
        return ("La source refuse les requêtes venant de ce serveur (HTTP 403, protection anti-robots). "
                "Les sources de repli ont été tentées ; utilisez la base CSV ou réessayez plus tard.")
    if "timeout" in t or "timed out" in t:
        return "La source n'a pas répondu à temps. Réessayez : la page relancera la requête."
    if "name resolution" in t or "connection" in t or "resolve" in t:
        return "Le serveur ne parvient pas à joindre Internet (DNS ou connexion). Vérifiez l'accès réseau de l'hébergement."
    if "modulenotfound" in t or "no module" in t:
        return "Une bibliothèque manque sur le serveur : vérifiez que requirements.txt a bien été installé (yfinance, curl_cffi, requests)."
    return "Erreur inattendue : voir le détail technique ci-dessous et le diagnostic des sources (page 💾 Base de données)."
