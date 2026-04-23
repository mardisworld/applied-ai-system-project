"""Unit tests for src/spotify_api.py.

All HTTP calls are intercepted with unittest.mock so no real network
or Spotify credentials are required to run these tests.
"""

from pathlib import Path
import base64
import io
import json
import sys
from unittest.mock import MagicMock, patch
from urllib import error as urllib_error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spotify_api import (
    SpotifyCredentials,
    add_tracks_to_playlist,
    build_playlist_from_recommendations,
    create_playlist,
    exchange_code_for_token,
    get_authorization_url,
    get_current_user,
    run_local_auth_flow,
    search_tracks,
    spotify_is_configured,
)
from urllib import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_creds(
    client_id: str = "test-id",
    client_secret: str = "test-secret",
    redirect_uri: str = "https://example.com/callback",
) -> SpotifyCredentials:
    return SpotifyCredentials(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


def _mock_urlopen(response_dict: dict, status: int = 200):
    """Return a context-manager mock that yields a fake HTTP response."""
    body = json.dumps(response_dict).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = body
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


def _http_error(code: int, body: str = "error") -> urllib_error.HTTPError:
    return urllib_error.HTTPError(
        url="https://api.spotify.com/v1/test",
        code=code,
        msg="Error",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# SpotifyCredentials
# ---------------------------------------------------------------------------

def test_basic_auth_header_is_correctly_base64_encoded():
    creds = _fake_creds(client_id="myid", client_secret="mysecret")
    expected = "Basic " + base64.b64encode(b"myid:mysecret").decode("utf-8")
    assert creds._basic_auth_header == expected


def test_credentials_from_env_raises_when_keys_missing():
    with patch.dict("os.environ", {"SPOTIFY_CLIENT_ID": "", "SPOTIFY_CLIENT_SECRET": ""}, clear=False):
        with pytest.raises(ValueError, match="Missing Spotify credentials"):
            SpotifyCredentials.from_env()


# ---------------------------------------------------------------------------
# spotify_is_configured
# ---------------------------------------------------------------------------

def test_spotify_is_configured_returns_true_when_both_keys_present():
    with patch.dict("os.environ", {"SPOTIFY_CLIENT_ID": "id", "SPOTIFY_CLIENT_SECRET": "secret"}, clear=False):
        assert spotify_is_configured() is True


def test_spotify_is_configured_returns_false_when_secret_missing():
    with patch.dict("os.environ", {"SPOTIFY_CLIENT_ID": "id", "SPOTIFY_CLIENT_SECRET": ""}, clear=False):
        assert spotify_is_configured() is False


# ---------------------------------------------------------------------------
# get_authorization_url
# ---------------------------------------------------------------------------

def test_get_authorization_url_contains_required_query_params():
    creds = _fake_creds()
    url = get_authorization_url(credentials=creds)
    assert "client_id=test-id" in url
    assert "response_type=code" in url
    assert "redirect_uri=" in url
    assert "scope=" in url
    assert "accounts.spotify.com/authorize" in url


def test_get_authorization_url_includes_custom_scopes():
    creds = _fake_creds()
    url = get_authorization_url(credentials=creds, scopes=["user-read-private"])
    assert "user-read-private" in url


def test_get_authorization_url_raises_without_redirect_uri():
    creds = _fake_creds(redirect_uri="")
    with pytest.raises(ValueError, match="SPOTIFY_REDIRECT_URI"):
        get_authorization_url(credentials=creds)


# ---------------------------------------------------------------------------
# exchange_code_for_token
# ---------------------------------------------------------------------------

def test_exchange_code_for_token_posts_correct_grant_type():
    token_response = {"access_token": "tok123", "token_type": "Bearer", "expires_in": 3600}
    creds = _fake_creds()

    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(token_response)) as mock_open:
        result = exchange_code_for_token("auth-code-xyz", credentials=creds)

    assert result["access_token"] == "tok123"
    call_args = mock_open.call_args
    sent_body = call_args[0][0].data.decode("utf-8")
    assert "grant_type=authorization_code" in sent_body
    assert "code=auth-code-xyz" in sent_body


def test_exchange_code_for_token_raises_without_redirect_uri():
    creds = _fake_creds(redirect_uri="")
    with pytest.raises(ValueError, match="SPOTIFY_REDIRECT_URI"):
        exchange_code_for_token("code", credentials=creds)


def test_exchange_code_for_token_raises_on_http_error():
    creds = _fake_creds()
    with patch("src.spotify_api.request.urlopen", side_effect=_http_error(401, "Unauthorized")):
        with pytest.raises(RuntimeError, match="HTTP 401"):
            exchange_code_for_token("bad-code", credentials=creds)


# ---------------------------------------------------------------------------
# search_tracks
# ---------------------------------------------------------------------------

def _spotify_track_item(track_id: str, name: str, artist: str, uri: str) -> dict:
    return {
        "id": track_id,
        "name": name,
        "artists": [{"name": artist}],
        "album": {"name": "Album"},
        "duration_ms": 210000,
        "popularity": 75,
        "uri": uri,
        "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
    }


def test_search_tracks_returns_normalized_track_list():
    api_response = {
        "tracks": {
            "items": [
                _spotify_track_item("id1", "It's My Life", "Bon Jovi", "spotify:track:id1"),
            ]
        }
    }
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(api_response)):
        results = search_tracks("It's My Life Bon Jovi", access_token="fake-token")

    assert len(results) == 1
    assert results[0]["name"] == "It's My Life"
    assert results[0]["artists"] == ["Bon Jovi"]
    assert results[0]["uri"] == "spotify:track:id1"


def test_search_tracks_returns_empty_list_when_no_results():
    api_response = {"tracks": {"items": []}}
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(api_response)):
        results = search_tracks("nonexistent song xyzzy", access_token="fake-token")

    assert results == []


def test_search_tracks_caps_limit_at_50():
    api_response = {"tracks": {"items": []}}
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(api_response)) as mock_open:
        search_tracks("anything", access_token="fake-token", limit=999)

    url = mock_open.call_args[0][0].full_url
    assert "limit=50" in url


def test_search_tracks_raises_on_http_error():
    with patch("src.spotify_api.request.urlopen", side_effect=_http_error(401, "Unauthorized")):
        with pytest.raises(RuntimeError, match="HTTP 401"):
            search_tracks("query", access_token="expired-token")


# ---------------------------------------------------------------------------
# create_playlist
# ---------------------------------------------------------------------------

def test_create_playlist_calls_me_playlists_endpoint():
    api_response = {
        "id": "playlist123",
        "name": "My Playlist",
        "external_urls": {"spotify": "https://open.spotify.com/playlist/playlist123"},
        "uri": "spotify:playlist:playlist123",
        "public": True,
    }
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(api_response)) as mock_open:
        result = create_playlist("My Playlist", access_token="fake-token")

    assert result["id"] == "playlist123"
    assert result["name"] == "My Playlist"
    assert result["url"] == "https://open.spotify.com/playlist/playlist123"
    url = mock_open.call_args[0][0].full_url
    assert "/me/playlists" in url


def test_create_playlist_sends_name_and_description_in_body():
    api_response = {
        "id": "pl1",
        "name": "Test",
        "external_urls": {},
        "uri": "spotify:playlist:pl1",
        "public": False,
    }
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(api_response)) as mock_open:
        create_playlist("Test", access_token="tok", description="A great playlist", public=False)

    sent_body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
    assert sent_body["name"] == "Test"
    assert sent_body["description"] == "A great playlist"
    assert sent_body["public"] is False


def test_create_playlist_raises_on_403():
    with patch("src.spotify_api.request.urlopen", side_effect=_http_error(403, "Forbidden")):
        with pytest.raises(RuntimeError, match="HTTP 403"):
            create_playlist("Blocked Playlist", access_token="no-scope-token")


# ---------------------------------------------------------------------------
# add_tracks_to_playlist
# ---------------------------------------------------------------------------

def test_add_tracks_to_playlist_calls_correct_endpoint():
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
        add_tracks_to_playlist(
            "playlist123",
            ["spotify:track:a", "spotify:track:b"],
            access_token="fake-token",
        )

    url = mock_open.call_args[0][0].full_url
    assert "/playlists/playlist123/items" in url


def test_add_tracks_to_playlist_sends_uris_in_body():
    uris = ["spotify:track:a", "spotify:track:b"]
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
        add_tracks_to_playlist("pl1", uris, access_token="tok")

    sent_body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
    assert sent_body["uris"] == uris


def test_add_tracks_to_playlist_skips_api_call_when_empty():
    with patch("src.spotify_api.request.urlopen") as mock_open:
        add_tracks_to_playlist("pl1", [], access_token="tok")

    mock_open.assert_not_called()


def test_add_tracks_to_playlist_truncates_to_100_uris():
    uris = [f"spotify:track:{i}" for i in range(150)]
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen({})) as mock_open:
        add_tracks_to_playlist("pl1", uris, access_token="tok")

    sent_body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
    assert len(sent_body["uris"]) == 100


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------

def test_get_current_user_returns_profile_dict():
    api_response = {"id": "user123", "display_name": "Test User"}
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen(api_response)):
        result = get_current_user(access_token="fake-token")

    assert result["id"] == "user123"
    assert result["display_name"] == "Test User"


def test_get_current_user_calls_me_endpoint():
    with patch("src.spotify_api.request.urlopen", return_value=_mock_urlopen({"id": "u1"})) as mock_open:
        get_current_user(access_token="fake-token")

    url = mock_open.call_args[0][0].full_url
    assert url.endswith("/me") or "/me?" in url


# ---------------------------------------------------------------------------
# build_playlist_from_recommendations
# ---------------------------------------------------------------------------

def test_build_playlist_from_recommendations_creates_playlist_and_adds_tracks():
    recommendations = [
        ({"title": "Livin' On A Prayer", "artist": "Bon Jovi"}, 19.5, "matches energy"),
        ({"title": "It's My Life", "artist": "Bon Jovi"}, 18.9, "matches genre"),
    ]

    create_response = {
        "id": "pl99",
        "name": "Rock Playlist",
        "external_urls": {"spotify": "https://open.spotify.com/playlist/pl99"},
        "uri": "spotify:playlist:pl99",
        "public": True,
    }
    search_response = {
        "tracks": {
            "items": [
                _spotify_track_item("t1", "Livin' On A Prayer", "Bon Jovi", "spotify:track:t1"),
            ]
        }
    }
    add_response = {}

    call_count = 0

    def fake_urlopen(req, timeout=30):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_urlopen(create_response)
        if call_count % 2 == 0:
            return _mock_urlopen(search_response)
        return _mock_urlopen(add_response)

    with patch("src.spotify_api.request.urlopen", side_effect=fake_urlopen):
        result = build_playlist_from_recommendations(
            playlist_name="Rock Playlist",
            recommendations=recommendations,
            access_token="fake-token",
        )

    assert result["id"] == "pl99"
    assert result["name"] == "Rock Playlist"
    assert "track_count" in result


def test_build_playlist_from_recommendations_skips_songs_with_no_title_and_no_artist():
    recommendations = [
        ({"title": "", "artist": ""}, 10.0, "empty song"),
    ]

    with patch("src.spotify_api.create_playlist", return_value={"id": "pl1", "name": "Empty", "url": None, "uri": None, "public": True}):
        with patch("src.spotify_api.search_tracks", return_value=[]) as mock_search:
            with patch("src.spotify_api.add_tracks_to_playlist") as mock_add:
                result = build_playlist_from_recommendations(
                    playlist_name="Empty",
                    recommendations=recommendations,
                    access_token="tok",
                )

    mock_search.assert_not_called()
    assert result["track_count"] == 0


# ---------------------------------------------------------------------------
# CSRF state verification in run_local_auth_flow
# ---------------------------------------------------------------------------

from src.spotify_api import run_local_auth_flow
import threading
from urllib.request import urlopen as _urlopen


def _fire_callback(port: int, code: str, state: str, delay: float = 0.05) -> None:
    """Send a fake Spotify callback to the local server in a background thread."""
    import time
    from urllib.request import urlopen as real_urlopen
    from urllib.error import URLError
    time.sleep(delay)
    url = f"http://127.0.0.1:{port}/callback?code={code}&state={state}"
    try:
        real_urlopen(url, timeout=3)
    except Exception:
        pass  # response content doesn’t matter to the test


def test_run_local_auth_flow_accepts_matching_state(monkeypatch):
    """A callback with the correct state token completes the flow."""
    creds = SpotifyCredentials(client_id="cid", client_secret="csec", redirect_uri="http://127.0.0.1:18888/callback")
    token_response = {"access_token": "tok", "token_type": "Bearer", "expires_in": 3600}

    monkeypatch.setattr("src.spotify_api.webbrowser.open", lambda url: True)
    monkeypatch.setattr("src.spotify_api.exchange_code_for_token", lambda code, credentials=None: token_response)

    # Intercept the CSRF state before the server starts by capturing it from the auth URL.
    captured_state: list = []
    original_urlencode = parse.urlencode
    def capturing_urlencode(params, *a, **kw):
        if isinstance(params, dict) and "state" in params:
            captured_state.append(params["state"])
        return original_urlencode(params, *a, **kw)
    monkeypatch.setattr("src.spotify_api.parse.urlencode", capturing_urlencode)

    # Fire the callback after the server starts, using the captured state.
    def delayed_callback():
        import time; time.sleep(0.1)
        state = captured_state[0] if captured_state else ""
        _fire_callback(18888, "auth-code-abc", state, delay=0)
    threading.Thread(target=delayed_callback, daemon=True).start()

    result = run_local_auth_flow(credentials=creds, port=18888)
    assert result == token_response


def test_run_local_auth_flow_rejects_mismatched_state(monkeypatch):
    """A callback with a wrong state token is rejected as a CSRF attempt."""
    creds = SpotifyCredentials(client_id="cid", client_secret="csec", redirect_uri="http://127.0.0.1:18889/callback")

    monkeypatch.setattr("src.spotify_api.webbrowser.open", lambda url: True)
    monkeypatch.setattr("src.spotify_api.exchange_code_for_token", lambda code, credentials=None: {})

    threading.Thread(
        target=_fire_callback,
        args=(18889, "auth-code-xyz", "wrong-state-value"),
        daemon=True,
    ).start()

    with pytest.raises(RuntimeError, match="state_mismatch|No authorization code"):
        run_local_auth_flow(credentials=creds, port=18889)
