"""Spotify Web API client.

Authentication
--------------
Two flows are supported.

1. Client Credentials (server-to-server, no user login required):
   - Used for searching tracks and reading public catalog data.
   - POST /api/token with grant_type=client_credentials and a Basic auth header
     containing base64(client_id:client_secret).

2. Authorization Code (requires user login):
   - Required for creating/modifying playlists on behalf of a user.
   - The user visits an authorization URL, Spotify redirects back with a code,
     that code is exchanged for an access token using grant_type=authorization_code.

Keys are loaded from the project-root .env file automatically:
    SPOTIFY_CLIENT_ID=
    SPOTIFY_CLIENT_SECRET=
    SPOTIFY_REDIRECT_URI=          # required for Authorization Code flow only
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

from src.env_loader import load_project_env
from src.logger import get_logger

logger = get_logger(__name__)


SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"

DEFAULT_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpotifyCredentials:
    client_id: str
    client_secret: str
    redirect_uri: str = ""

    @classmethod
    def from_env(cls) -> "SpotifyCredentials":
        load_project_env()
        client_id = (os.environ.get("SPOTIFY_CLIENT_ID") or "").strip()
        client_secret = (os.environ.get("SPOTIFY_CLIENT_SECRET") or "").strip()
        redirect_uri = (os.environ.get("SPOTIFY_REDIRECT_URI") or "").strip()

        if not client_id or not client_secret:
            raise ValueError(
                "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET in your .env file."
            )

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )

    @property
    def _basic_auth_header(self) -> str:
        """Base64-encoded 'client_id:client_secret' as required by Spotify."""
        raw = f"{self.client_id}:{self.client_secret}"
        return "Basic " + base64.b64encode(raw.encode("utf-8")).decode("utf-8")


def spotify_is_configured() -> bool:
    load_project_env()
    client_id = (os.environ.get("SPOTIFY_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("SPOTIFY_CLIENT_SECRET") or "").strip()
    return bool(client_id and client_secret)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post_form(
    url: str,
    body: Dict[str, str],
    headers: Dict[str, str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    encoded_body = parse.urlencode(body).encode("utf-8")
    http_request = request.Request(url, data=encoded_body, headers=headers, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        logger.error("Spotify token request failed HTTP %d: %s", exc.code, body_text[:200])
        raise RuntimeError(f"Spotify request failed with HTTP {exc.code}: {body_text}") from exc
    except error.URLError as exc:
        logger.error("Spotify token request connection error: %s", exc.reason)
        raise RuntimeError(f"Spotify request failed: {exc.reason}") from exc


def _api_request(
    method: str,
    endpoint: str,
    access_token: str,
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    url = f"{SPOTIFY_API_BASE}/{endpoint.lstrip('/')}"
    if params:
        url = f"{url}?{parse.urlencode(params)}"

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    data = json.dumps(body).encode("utf-8") if body is not None else None
    http_request = request.Request(url, data=data, headers=headers, method=method)

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        logger.error("Spotify API %s %s failed HTTP %d: %s", method, endpoint, exc.code, body_text[:200])
        raise RuntimeError(f"Spotify API request failed with HTTP {exc.code}: {body_text}") from exc
    except error.URLError as exc:
        logger.error("Spotify API %s %s connection error: %s", method, endpoint, exc.reason)
        raise RuntimeError(f"Spotify API request failed: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Client Credentials flow
# ---------------------------------------------------------------------------


def get_client_credentials_token(
    credentials: Optional[SpotifyCredentials] = None,
) -> str:
    """Fetch an app-level access token using the Client Credentials flow.

    This token can read public catalog data (tracks, artists, albums) but
    cannot create or modify playlists, which require a user token.
    """
    creds = credentials or SpotifyCredentials.from_env()

    payload = _post_form(
        SPOTIFY_TOKEN_URL,
        body={"grant_type": "client_credentials"},
        headers={
            "Authorization": creds._basic_auth_header,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    token = (payload.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("Spotify did not return an access token.")
    return token


# ---------------------------------------------------------------------------
# Authorization Code flow (required for playlist creation)
# ---------------------------------------------------------------------------


def get_authorization_url(
    credentials: Optional[SpotifyCredentials] = None,
    scopes: List[str] | None = None,
    state: str = "",
) -> str:
    """Return the Spotify URL the user must visit to grant access.

    After approving, Spotify redirects to SPOTIFY_REDIRECT_URI with a ``code``
    query parameter that you pass to ``exchange_code_for_token``.
    """
    creds = credentials or SpotifyCredentials.from_env()
    if not creds.redirect_uri:
        raise ValueError("SPOTIFY_REDIRECT_URI must be set for the Authorization Code flow.")

    default_scopes = ["playlist-modify-public", "playlist-modify-private", "user-read-private"]
    params = {
        "client_id": creds.client_id,
        "response_type": "code",
        "redirect_uri": creds.redirect_uri,
        "scope": " ".join(scopes or default_scopes),
    }
    if state:
        params["state"] = state

    return f"{SPOTIFY_AUTH_URL}?{parse.urlencode(params)}"


def exchange_code_for_token(
    code: str,
    credentials: Optional[SpotifyCredentials] = None,
) -> Dict[str, Any]:
    """Exchange the authorization code returned by Spotify for an access token.

    Returns the full token response dict which includes ``access_token``,
    ``refresh_token``, ``expires_in``, and ``token_type``.
    """
    creds = credentials or SpotifyCredentials.from_env()
    if not creds.redirect_uri:
        raise ValueError("SPOTIFY_REDIRECT_URI must be set for the Authorization Code flow.")

    logger.info("Exchanging authorization code for access token.")
    result = _post_form(
        SPOTIFY_TOKEN_URL,
        body={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": creds.redirect_uri,
        },
        headers={
            "Authorization": creds._basic_auth_header,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    token_type = result.get("token_type", "unknown")
    expires_in = result.get("expires_in", "?")
    logger.info("Access token received (type=%s, expires_in=%ss).", token_type, expires_in)
    return result


def refresh_access_token(
    refresh_token: str,
    credentials: Optional[SpotifyCredentials] = None,
) -> Dict[str, Any]:
    """Use a refresh token to obtain a new access token without user interaction."""
    creds = credentials or SpotifyCredentials.from_env()

    return _post_form(
        SPOTIFY_TOKEN_URL,
        body={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={
            "Authorization": creds._basic_auth_header,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )


def run_local_auth_flow(
    credentials: Optional[SpotifyCredentials] = None,
    scopes: List[str] | None = None,
    port: int = 8888,
) -> Dict[str, Any]:
    """Complete the Authorization Code flow using a local callback server.

    Uses SPOTIFY_REDIRECT_URI from credentials/env as the redirect URI sent to
    Spotify (must be registered in the Spotify dashboard). Listens locally on
    ``port`` to capture the callback — this works both for http://localhost
    and for an ngrok HTTPS tunnel that forwards to the same local port.

    Returns the full token response dict (access_token, refresh_token, etc.).
    """
    creds = credentials or SpotifyCredentials.from_env()

    redirect_uri = creds.redirect_uri or f"http://localhost:{port}/callback"
    default_scopes = ["playlist-modify-public", "playlist-modify-private", "user-read-private"]

    # Generate a CSRF state token and verify it in the callback to prevent
    # cross-site request forgery attacks on the OAuth flow.
    csrf_state = secrets.token_urlsafe(16)

    params = {
        "client_id": creds.client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes or default_scopes),
        "state": csrf_state,
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{parse.urlencode(params)}"

    # Capture the authorization code via a one-shot local HTTP server.
    captured: Dict[str, str] = {}

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            qs = parse.parse_qs(parse.urlparse(self.path).query)
            returned_state = (qs.get("state") or [""])[0]
            captured["code"] = (qs.get("code") or [""])[0]
            captured["error"] = (qs.get("error") or [""])[0]

            if returned_state != csrf_state:
                captured["code"] = ""
                captured["error"] = "state_mismatch"
                body = (
                    b"<html><body><h2>Authorization failed.</h2>"
                    b"<p>State mismatch - possible CSRF attack. Please try again.</p></body></html>"
                )
                self.send_response(400)
            elif captured.get("code"):
                body = (
                    b"<html><body><h2>Authorization successful!</h2>"
                    b"<p>You can close this tab and return to the terminal.</p></body></html>"
                )
                self.send_response(200)
            else:
                body = (
                    b"<html><body><h2>Authorization failed.</h2>"
                    b"<p>Check the terminal for details.</p></body></html>"
                )
                self.send_response(400)

            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:  # silence request logs
            pass

    class _ReuseAddrHTTPServer(HTTPServer):
        allow_reuse_address = True

    server = _ReuseAddrHTTPServer(("127.0.0.1", port), _CallbackHandler)
    print(f"\nOpening Spotify authorization in your browser...")
    opened = webbrowser.open(auth_url)
    if not opened:
        print("Could not open browser automatically. Open this URL manually:")
    print(f"\n  {auth_url}\n")
    print("Waiting for you to approve access in the browser (press Ctrl+C to cancel)...\n")

    server.handle_request()  # blocks until one request arrives
    server.server_close()

    if captured.get("error"):
        raise RuntimeError(f"Spotify returned an error: {captured['error']}")

    code = captured.get("code", "")
    if not code:
        raise RuntimeError("No authorization code received from Spotify.")

    # Override the redirect URI credential to match what was sent to Spotify.
    creds_for_exchange = SpotifyCredentials(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        redirect_uri=redirect_uri,
    )
    return exchange_code_for_token(code, credentials=creds_for_exchange)


# ---------------------------------------------------------------------------
# Track search (Client Credentials token is sufficient)
# ---------------------------------------------------------------------------


def search_tracks(
    query: str,
    access_token: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search for tracks by query string. Returns normalized track dicts."""
    logger.debug("Searching Spotify tracks: %r (limit=%d)", query, limit)
    payload = _api_request(
        "GET",
        "/search",
        access_token=access_token,
        params={"q": query, "type": "track", "limit": str(min(limit, 50))},
    )

    items = ((payload.get("tracks") or {}).get("items") or [])
    tracks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        artists = [a.get("name") for a in (item.get("artists") or []) if isinstance(a, dict) and a.get("name")]
        album = item.get("album") or {}
        tracks.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "artists": artists,
                "album": album.get("name"),
                "duration_ms": item.get("duration_ms"),
                "popularity": item.get("popularity"),
                "uri": item.get("uri"),
                "external_url": ((item.get("external_urls") or {}).get("spotify")),
            }
        )
    logger.debug("Spotify search returned %d result(s) for %r.", len(tracks), query)
    return tracks


# ---------------------------------------------------------------------------
# Playlist creation (requires Authorization Code user token)
# ---------------------------------------------------------------------------


def create_playlist(
    name: str,
    access_token: str,
    description: str = "",
    public: bool = True,
) -> Dict[str, Any]:
    """Create an empty playlist for the current user via POST /me/playlists.

    ``access_token`` must be a user-scoped token obtained via the
    Authorization Code flow with the ``playlist-modify-public`` or
    ``playlist-modify-private`` scope granted.
    """
    logger.info("Creating Spotify playlist: %r (public=%s)", name, public)
    payload = _api_request(
        "POST",
        "/me/playlists",
        access_token=access_token,
        body={
            "name": name,
            "description": description,
            "public": public,
        },
    )

    playlist = {
        "id": payload.get("id"),
        "name": payload.get("name"),
        "url": ((payload.get("external_urls") or {}).get("spotify")),
        "uri": payload.get("uri"),
        "public": payload.get("public"),
    }
    logger.info("Playlist created: id=%r url=%r", playlist["id"], playlist["url"])
    return playlist


def add_tracks_to_playlist(
    playlist_id: str,
    track_uris: List[str],
    access_token: str,
) -> None:
    """Add up to 100 tracks to a playlist in a single request.

    Spotify's endpoint accepts up to 100 URIs; pass them in batches if
    you have more.
    """
    if not track_uris:
        logger.debug("add_tracks_to_playlist called with empty URI list; skipping API call.")
        return

    logger.info("Adding %d track(s) to playlist %r.", len(track_uris[:100]), playlist_id)
    _api_request(
        "POST",
        f"/playlists/{playlist_id}/items",
        access_token=access_token,
        body={"uris": track_uris[:100]},
    )
    logger.debug("Tracks successfully added to playlist %r.", playlist_id)


def get_current_user(access_token: str) -> Dict[str, Any]:
    """Return the profile of the user associated with the given access token.

    Requires a user-scoped token (Authorization Code flow). Returns a dict
    with at least ``id`` and ``display_name``.
    """
    return _api_request("GET", "/me", access_token=access_token)


def build_playlist_from_recommendations(
    playlist_name: str,
    recommendations: List[Dict[str, Any]],
    access_token: str,
    description: str = "",
    public: bool = True,
) -> Dict[str, Any]:
    """Create a Spotify playlist and populate it from local recommender results.

    ``recommendations`` should be the list returned by ``recommend_songs``:
    each element is a tuple ``(song_dict, score, explanation)``.
    The function searches Spotify for each song by title + artist and adds the
    best match. Songs that cannot be found on Spotify are skipped.

    Returns the created playlist metadata dict.
    """
    playlist = create_playlist(
        name=playlist_name,
        access_token=access_token,
        description=description,
        public=public,
    )
    playlist_id = playlist["id"]

    track_uris: List[str] = []
    for rec in recommendations:
        song, _score, _explanation = rec
        query = f"{song.get('title', '')} {song.get('artist', '')}".strip()
        if not query:
            continue

        results = search_tracks(query, access_token=access_token, limit=1)
        if results and results[0].get("uri"):
            track_uris.append(results[0]["uri"])

    if track_uris:
        add_tracks_to_playlist(playlist_id, track_uris, access_token=access_token)

    playlist["track_count"] = len(track_uris)
    return playlist
