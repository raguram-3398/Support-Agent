import base64
import hashlib
import hmac
import json

import jwt
import pytest
from conftest import KID, SIGNING_KEY, make_token, serve_test_jwks
from cryptography.hazmat.primitives import serialization

from support_agent.auth import JWKS_URL, Verifier

AUDIENCE = "support-api"


@pytest.fixture
def verifier(monkeypatch):
    verifier = Verifier(JWKS_URL, AUDIENCE)
    serve_test_jwks(monkeypatch, verifier)
    return verifier


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unsigned_parts(header: dict, payload: dict) -> str:
    return (
        f"{b64url(json.dumps(header).encode())}.{b64url(json.dumps(payload).encode())}"
    )


def payload_of(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})


def test_valid_token_returns_claims(verifier):
    claims = verifier.verify(make_token(AUDIENCE, customer_id=1))
    assert claims["customer_id"] == 1


def test_wrong_audience_is_rejected(verifier):
    with pytest.raises(jwt.InvalidAudienceError):
        verifier.verify(make_token("http://127.0.0.1:8000/mcp"))


def test_wrong_issuer_is_rejected(verifier):
    with pytest.raises(jwt.InvalidIssuerError):
        verifier.verify(make_token(AUDIENCE, iss="http://evil.example/realms/support"))


def test_expired_token_is_rejected(verifier):
    with pytest.raises(jwt.ExpiredSignatureError):
        verifier.verify(make_token(AUDIENCE, iat=1, exp=2))


def test_token_without_exp_is_rejected(verifier):
    token = make_token(AUDIENCE)
    payload = payload_of(token)
    del payload["exp"]
    token = jwt.encode(payload, SIGNING_KEY, algorithm="RS256", headers={"kid": KID})
    with pytest.raises(jwt.MissingRequiredClaimError):
        verifier.verify(token)


def test_edited_payload_fails_signature(verifier):
    header, _, signature = make_token(AUDIENCE, customer_id=1).split(".")
    payload = payload_of(make_token(AUDIENCE, customer_id=2))
    forged = f"{header}.{b64url(json.dumps(payload).encode())}.{signature}"
    with pytest.raises(jwt.InvalidSignatureError):
        verifier.verify(forged)


def test_alg_none_is_rejected(verifier):
    payload = payload_of(make_token(AUDIENCE, customer_id=2))
    forged = unsigned_parts({"alg": "none", "kid": KID}, payload) + "."
    with pytest.raises(jwt.InvalidAlgorithmError):
        verifier.verify(forged)


def test_hs256_signed_with_public_key_is_rejected(verifier):
    public_pem = SIGNING_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    payload = payload_of(make_token(AUDIENCE, customer_id=2))
    signing_input = unsigned_parts({"alg": "HS256", "kid": KID}, payload)
    mac = hmac.new(public_pem, signing_input.encode(), hashlib.sha256).digest()
    with pytest.raises(jwt.InvalidAlgorithmError):
        verifier.verify(f"{signing_input}.{b64url(mac)}")


def test_unknown_kid_is_rejected(verifier):
    with pytest.raises(jwt.PyJWKClientError):
        verifier.verify(make_token(AUDIENCE, headers={"kid": "rotated-away"}))
