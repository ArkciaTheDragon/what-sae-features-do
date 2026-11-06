"""HTTP requests for experiment data and judge responses."""

import logging
import time

import httpx

LOGGER = logging.getLogger(__name__)


def get_json(client: httpx.Client, url: str) -> dict:
    for attempt in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            if error.response.status_code < 500 and error.response.status_code != 429:
                raise
            if attempt == 2:
                raise
            LOGGER.warning("Retrying %s after HTTP %s", url, error.response.status_code)
        except httpx.TransportError:
            if attempt == 2:
                raise
            LOGGER.warning("Retrying %s after a transport error", url, exc_info=True)
        time.sleep(attempt + 1)
    raise RuntimeError(f"Request did not complete: {url}")
