import jwt

ISSUER = "http://127.0.0.1:8080/realms/support"
JWKS_URL = f"{ISSUER}/protocol/openid-connect/certs"
ALGORITHMS = ["RS256"]
REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub"]
JWKS_TIMEOUT_S = 5


class Verifier:
    def __init__(self, jwks_url: str, audience: str):
        self.jwks = jwt.PyJWKClient(jwks_url, timeout=JWKS_TIMEOUT_S)
        self.audience = audience

    def verify(self, token: str) -> dict:
        key = self.jwks.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            key,
            algorithms=ALGORITHMS,
            audience=self.audience,
            issuer=ISSUER,
            options={"require": REQUIRED_CLAIMS},
        )
