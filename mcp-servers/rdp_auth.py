import os
import httpx
import logging
import json
import tempfile
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

RDP_USERNAME = os.getenv("RDP_USERNAME")
RDP_PASSWORD = os.getenv("RDP_PASSWORD")
RDP_CLIENT_ID = os.getenv("RDP_CLIENT_ID")
RDP_BASE_URL = "https://api.refinitiv.com"

CACHE_FILE = Path(tempfile.gettempdir()) / "rdp_token_cache.json"


def _load_token_cache() -> dict:
    """Load token cache from file"""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                # Convert expires_at string back to datetime
                if cache.get("expires_at"):
                    cache["expires_at"] = datetime.fromisoformat(cache["expires_at"])
                return cache
    except Exception as e:
        logger.warning(f"Failed to load token cache: {e}")
    
    return {"access_token": None, "expires_at": None, "refresh_token": None}


def _save_token_cache(cache: dict):
    """Save token cache to file"""
    try:
        # Convert datetime to string for JSON serialization
        cache_to_save = cache.copy()
        if cache_to_save.get("expires_at"):
            cache_to_save["expires_at"] = cache_to_save["expires_at"].isoformat()
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_to_save, f)
        logger.debug(f"Token cache saved to {CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save token cache: {e}")


def _get_token_cache() -> dict:
    """Get current token cache (loads from file each time)"""
    return _load_token_cache()


def check_credentials() -> bool:
    """Check if all required credentials are available"""
    missing = []
    if not RDP_USERNAME:
        missing.append("RDP_USERNAME")
    if not RDP_PASSWORD:
        missing.append("RDP_PASSWORD")
    if not RDP_CLIENT_ID:
        missing.append("RDP_CLIENT_ID")

    if missing:
        logger.error(f"Missing required credentials: {', '.join(missing)}")
        return False

    logger.info(f"RDP_USERNAME set: {bool(RDP_USERNAME)}")
    logger.info(f"RDP_PASSWORD set: {bool(RDP_PASSWORD)}")
    logger.info(f"RDP_CLIENT_ID set: {bool(RDP_CLIENT_ID)}")
    return True


async def get_auth_token() -> Optional[str]:
    """
    Get the authentication token for the RDP API with caching to avoid unnecessary re-authentication
    """
    # Load current cache from file
    token_cache = _get_token_cache()
    
    # Check if we have a valid cached token
    if (
        token_cache["access_token"]
        and token_cache["expires_at"]
        and datetime.now() < token_cache["expires_at"]
    ):
        logger.info("Using cached auth token")
        return token_cache["access_token"]

    # Check if required credentials are available
    if not check_credentials():
        return None

    logger.info("Fetching new auth token")
    auth_token_url = f"{RDP_BASE_URL}/auth/oauth2/v1/token"
    payload = f"grant_type=password&username={RDP_USERNAME}&scope=trapi&client_id={RDP_CLIENT_ID}&password={RDP_PASSWORD}&takeExclusiveSignOnControl=true"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    logger.info(f"Auth URL: {auth_token_url}")
    logger.info(
        f"Auth payload (masked): grant_type=password&username={RDP_USERNAME}&scope=trapi&client_id={RDP_CLIENT_ID}&password=***&takeExclusiveSignOnControl=true"
    )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                auth_token_url, headers=headers, data=payload, timeout=30.0
            )
            logger.info(f"Debug - Auth response status: {response.status_code}")
            logger.info(f"Debug - Auth response headers: {dict(response.headers)}")

            if response.status_code != 200:
                response_text = response.text
                logger.error(f"Debug - Auth error response body: {response_text}")

            response.raise_for_status()
            data = response.json()

            # Cache the token with expiration
            access_token = data.get("access_token")
            expires_in = data.get(
                "expires_in", 3600
            )  # Default to 1 hour if not provided

            # Convert expires_in to integer if it's a string
            try:
                expires_in = int(expires_in)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid expires_in value: {expires_in}, using default 3600"
                )
                expires_in = 3600

            if access_token:
                # Update cache and save to file
                token_cache["access_token"] = access_token
                token_cache["refresh_token"] = data.get("refresh_token")
                # Set expiration to 5 minutes before actual expiry for safety margin
                token_cache["expires_at"] = datetime.now() + timedelta(
                    seconds=expires_in - 300
                )
                _save_token_cache(token_cache)

                logger.info(
                    f"Debug - Auth success, token cached until {token_cache['expires_at']} (expires_in: {expires_in})"
                )

            return access_token
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP Error fetching auth token: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching auth token: {type(e).__name__}: {e}")
            return None


async def refresh_auth_token() -> Optional[str]:
    """
    Refresh the authentication token using the refresh token if available
    """
    token_cache = _get_token_cache()
    
    if not token_cache["refresh_token"]:
        logger.info("No refresh token available, getting new token")
        return await get_auth_token()

    logger.info("Attempting to refresh auth token")
    auth_token_url = f"{RDP_BASE_URL}/auth/oauth2/v1/token"
    payload = f"grant_type=refresh_token&username={RDP_USERNAME}&client_id={RDP_CLIENT_ID}&refresh_token={token_cache['refresh_token']}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                auth_token_url, headers=headers, data=payload, timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)

                # Convert expires_in to integer if it's a string
                try:
                    expires_in = int(expires_in)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid expires_in value in refresh: {expires_in}, using default 3600"
                    )
                    expires_in = 3600

                if access_token:
                    token_cache["access_token"] = access_token
                    token_cache["refresh_token"] = data.get(
                        "refresh_token", token_cache["refresh_token"]
                    )
                    token_cache["expires_at"] = datetime.now() + timedelta(
                        seconds=expires_in - 300
                    )
                    _save_token_cache(token_cache)

                    logger.info("Token refreshed successfully")
                    return access_token

            # If refresh fails, fall back to getting a new token
            logger.info("Token refresh failed, getting new token")
            return await get_auth_token()

        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            # Fall back to getting a new token
            return await get_auth_token()


async def get_valid_token() -> Optional[str]:
    """
    Get a valid authentication token, using cache, refresh, or new authentication as needed
    """
    token_cache = _get_token_cache()
    
    # Check if we have a valid cached token
    if (
        token_cache["access_token"]
        and token_cache["expires_at"]
        and datetime.now() < token_cache["expires_at"]
    ):
        logger.info("Using cached auth token")
        return token_cache["access_token"]

    # Check if token is expired but we have a refresh token
    if token_cache["refresh_token"]:
        return await refresh_auth_token()

    # Otherwise get a new token
    return await get_auth_token()


async def make_authenticated_request(
    url: str, method: str = "GET", **kwargs
) -> httpx.Response:
    """
    Make an authenticated HTTP request with automatic token retry on 401 errors
    """
    auth_token = await get_valid_token()
    if not auth_token:
        raise Exception("Unable to authenticate with news service")

    headers = kwargs.get("headers", {})
    headers.update(
        {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "cache-control": "no-cache",
        }
    )
    kwargs["headers"] = headers

    async with httpx.AsyncClient() as client:
        if method.upper() == "GET":
            response = await client.get(url, **kwargs)
        elif method.upper() == "POST":
            response = await client.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        # If we get a 401, clear the cached token and try once more
        if response.status_code == 401:
            logger.info("Received 401, clearing token cache and retrying")
            # Clear the cache file
            clear_token_cache()

            # Get a fresh token and retry
            auth_token = await get_valid_token()
            if not auth_token:
                raise Exception("Unable to re-authenticate with news service")

            headers["Authorization"] = f"Bearer {auth_token}"
            kwargs["headers"] = headers

            if method.upper() == "GET":
                response = await client.get(url, **kwargs)
            elif method.upper() == "POST":
                response = await client.post(url, **kwargs)

        return response


def clear_token_cache():
    """Clear the token cache (useful for testing or logout)"""
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        logger.info("Token cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear token cache: {e}")


def get_token_info() -> dict:
    """Get current token cache information (for debugging)"""
    token_cache = _get_token_cache()
    return {
        "has_access_token": bool(token_cache["access_token"]),
        "has_refresh_token": bool(token_cache["refresh_token"]),
        "expires_at": token_cache["expires_at"].isoformat() if token_cache["expires_at"] else None,
        "is_valid": (token_cache["expires_at"] and datetime.now() < token_cache["expires_at"]) if token_cache["expires_at"] else False
    }
