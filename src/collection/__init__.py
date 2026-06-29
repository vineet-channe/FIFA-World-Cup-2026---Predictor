"""Data collection package for WC 2026 predictor."""

from .kaggle_results import download_results
from .elo_scraper import download_all_elo
from .fbref_scraper import discover_team_ids, download_all_fbref
from .fifa_rankings import download_rankings
from .api_football import APIFootball
from .transfermarkt import download_all_squads
from .schedule import build_schedule, load_schedule

__all__ = [
    "download_results",
    "download_all_elo",
    "discover_team_ids",
    "download_all_fbref",
    "download_rankings",
    "APIFootball",
    "download_all_squads",
    "build_schedule",
    "load_schedule",
]
