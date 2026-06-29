"""API-Football (api-sports.io) wrapper class (DS-05)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings

__all__ = ["APIFootball"]


class APIFootball:
    """Thin wrapper around the API-Football v3 REST API.

    Authentication uses the `x-apisports-key` header — no Bearer token.
    The free tier allows 100 requests/day; check remaining via check_status().

    Usage::

        client = APIFootball()            # reads API_FOOTBALL_KEY from .env
        status = client.check_status()
        print(client.remaining_requests())
    """

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.API_FOOTBALL_KEY
        if not key:
            logger.warning(
                "API_FOOTBALL_KEY is empty — API calls will fail with 401. "
                "Set it in .env or pass api_key= explicitly."
            )
        # The only header needed — do NOT add X-RapidAPI-Host or similar
        self.headers: dict[str, str] = {"x-apisports-key": key}
        self._request_count: int = 0

    # ── Public methods ─────────────────────────────────────────────────────

    def check_status(self) -> dict[str, Any]:
        """Return account status including remaining daily requests."""
        return self._get("/status")

    def get_injuries(self, team_id: int, season: int = 2026) -> dict[str, Any]:
        """Return injury list for a team in a given season."""
        return self._get("/injuries", {"team": team_id, "season": season})

    def get_lineups(self, fixture_id: int) -> dict[str, Any]:
        """Return confirmed lineups for a fixture (call ~1h before kickoff)."""
        return self._get("/fixtures/lineups", {"fixture": fixture_id})

    def get_fixture_stats(self, fixture_id: int) -> dict[str, Any]:
        """Return match statistics for a completed fixture."""
        return self._get("/fixtures/statistics", {"fixture": fixture_id})

    def get_team_statistics(
        self, team_id: int, season: int, league_id: int
    ) -> dict[str, Any]:
        """Return aggregated team statistics for a season and league."""
        return self._get(
            "/teams/statistics",
            {"team": team_id, "season": season, "league": league_id},
        )

    def remaining_requests(self) -> int:
        """Parse the status response and return remaining daily requests."""
        try:
            status = self.check_status()
            return int(
                status.get("response", {})
                .get("requests", {})
                .get("remaining", 0)
            )
        except Exception as exc:
            logger.warning(f"Could not fetch remaining requests: {exc}")
            return -1

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        delay: float = 0.5,
    ) -> dict[str, Any]:
        """Make a GET request to the API, respecting rate limits.

        Raises ValueError if the response contains non-empty errors.
        Raises requests.HTTPError on 4xx / 5xx status codes.
        """
        time.sleep(delay)
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = requests.get(
                url,
                params=params or {},
                headers=self.headers,
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error(f"Network error on {endpoint}: {exc}")
            raise

        self._request_count += 1
        resp.raise_for_status()

        data: dict[str, Any] = resp.json()
        errors = data.get("errors")
        if errors:
            raise ValueError(f"API error on {endpoint}: {errors}")

        logger.debug(
            f"GET {endpoint} → {resp.status_code} "
            f"(total requests this session: {self._request_count})"
        )
        return data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = APIFootball()
    try:
        status = client.check_status()
        resp = status.get("response", {})
        account = resp.get("account", {})
        requests_info = resp.get("requests", {})

        firstname = account.get("firstname", "Unknown")
        lastname = account.get("lastname", "")
        remaining = requests_info.get("remaining", "N/A")
        used = requests_info.get("current", "N/A")
        limit = requests_info.get("limit_day", "N/A")

        print(f"Account:           {firstname} {lastname}".strip())
        print(f"Requests remaining: {remaining} / {limit}")
        print(f"Requests used today: {used}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
