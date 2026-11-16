import base64
import hashlib
import secrets
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from support_agent.auth import ISSUER

CLIENT_ID = "support-cli"
REDIRECT_HOST, REDIRECT_PORT = "127.0.0.1", 8765
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}/callback"
AUTH_URL = f"{ISSUER}/protocol/openid-connect/auth"
TOKEN_URL = f"{ISSUER}/protocol/openid-connect/token"
HTTP_TIMEOUT_S = 10
LOGIN_TIMEOUT_S = 120


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def wait_for_callback() -> dict[str, str]:
    received = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            received.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Signed in. You can close this tab.")

        def log_message(self, *args):
            pass

    with HTTPServer((REDIRECT_HOST, REDIRECT_PORT), Callback) as server:
        server.timeout = LOGIN_TIMEOUT_S
        server.handle_request()
    return received


def main() -> None:
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    url = f"{AUTH_URL}?{urlencode(params)}"
    print(f"Sign in at: {url}", file=sys.stderr)
    webbrowser.open(url)
    callback = wait_for_callback()
    if callback.get("state") != state:
        raise SystemExit("State mismatch: not the login this script started.")
    if callback.get("iss") != ISSUER:
        raise SystemExit(f"Unexpected issuer: {callback.get('iss')}")
    if "code" not in callback:
        raise SystemExit(f"Login failed: {callback.get('error', 'no code returned')}")
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": callback["code"],
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        },
        timeout=HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    print(response.json()["access_token"])


if __name__ == "__main__":
    main()
