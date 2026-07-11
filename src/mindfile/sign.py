"""Authenticity — sign the file's state so verification proves *who* attested
to it, not just that it is internally intact.

The signed value is the **state root**: a digest of the file uuid + schema
version + the (hash-chained) history head + the latest checkpoint's manifest
root. Signing it pins a complete, attested snapshot.

* ``Ed25519Signer`` (needs the optional ``cryptography`` extra) — public-key:
  sign with a private key, anyone verifies with the stored public key, nobody
  can forge. This is the real authenticity story.
* ``HmacSigner`` (stdlib) — symmetric fallback; verification needs the shared
  secret, so the verifier could also forge. Use only when a dependency-free
  keyed check is enough.

Signing does NOT append to ``history`` (that would change the very head it
signs); the ``signatures`` table is the record.
"""

from __future__ import annotations

import hashlib
import hmac

from .file import MindFile, utcnow


def state_root(mf: MindFile) -> str:
    """Digest of the file's attestable state (uuid + schema + history head +
    latest checkpoint root)."""
    head = mf.meta_get("history_head") or ""
    cp = mf.conn.execute(
        "SELECT manifest_sha256 FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
    cproot = cp["manifest_sha256"] if cp else ""
    uuid = mf.meta_get("file_uuid") or ""
    sv = mf.meta_get("schema_version") or ""
    return hashlib.sha256(f"state:{uuid}\t{sv}\t{head}\t{cproot}".encode("utf-8")).hexdigest()


class HmacSigner:
    algo = "hmac-sha256"

    def __init__(self, secret: bytes, key_id: str = "hmac"):
        self.secret = secret
        self.key_id = key_id
        self.public_key = None  # symmetric: nothing public to store

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self.secret, data, hashlib.sha256).digest()


class Ed25519Signer:
    algo = "ed25519"

    def __init__(self, private_bytes: bytes | None = None, key_id: str | None = None):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        self._ser = serialization
        self._sk = (Ed25519PrivateKey.from_private_bytes(private_bytes)
                    if private_bytes is not None else Ed25519PrivateKey.generate())
        self.public_key = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.key_id = key_id or hashlib.sha256(self.public_key).hexdigest()[:16]

    def sign(self, data: bytes) -> bytes:
        return self._sk.sign(data)

    def private_bytes(self) -> bytes:
        return self._sk.private_bytes(
            self._ser.Encoding.Raw, self._ser.PrivateFormat.Raw, self._ser.NoEncryption())


def sign_state(mf: MindFile, signer, *, note: str | None = None) -> dict:
    """Sign the current state root and store the signature."""
    root = state_root(mf)
    sig = signer.sign(root.encode("utf-8"))
    mf.conn.execute(
        "INSERT INTO signatures(algo, key_id, public_key, state_root, signature, signed_at, note) "
        "VALUES(?,?,?,?,?,?,?)",
        (signer.algo, signer.key_id, signer.public_key, root, sig, utcnow(), note))
    mf.conn.commit()
    return {"algo": signer.algo, "key_id": signer.key_id, "state_root": root}


def _ed25519_verify(public_key: bytes, signature: bytes, data: bytes) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(bytes(public_key)).verify(bytes(signature), data)
        return True
    except Exception:
        return False


def trust_key(mf: MindFile, public_key: bytes, *, key_id: str | None = None,
              label: str | None = None) -> str:
    """Pin an expected/trusted signing key. After this, authenticity means a
    valid signature from THIS key, not merely any key embedded in the file."""
    kid = key_id or hashlib.sha256(bytes(public_key)).hexdigest()[:16]
    mf.conn.execute(
        "INSERT INTO trusted_keys(key_id, public_key, label, added_at) VALUES(?,?,?,?) "
        "ON CONFLICT(key_id) DO UPDATE SET public_key=excluded.public_key, label=excluded.label",
        (kid, bytes(public_key), label, utcnow()))
    mf.conn.commit()
    return kid


def trusted_key_ids(mf: MindFile) -> set[str]:
    return {r["key_id"] for r in mf.conn.execute("SELECT key_id FROM trusted_keys")}


def verify_signatures(mf: MindFile, *, hmac_secret: bytes | None = None) -> dict:
    """Check every stored signature. ``valid`` is the cryptographic verdict
    (``None`` = HMAC that needs a secret to check); ``matches_current`` says
    whether the file is still in the exact state that was signed; ``trusted``
    says the signing key is a pinned/expected one."""
    current = state_root(mf)
    trusted = trusted_key_ids(mf)
    out = []
    for r in mf.conn.execute(
            "SELECT id, algo, key_id, public_key, state_root, signature, signed_at FROM signatures "
            "ORDER BY id"):
        if r["algo"] == "ed25519":
            valid = _ed25519_verify(r["public_key"], r["signature"], r["state_root"].encode())
        elif r["algo"] == "hmac-sha256" and hmac_secret is not None:
            expected = hmac.new(hmac_secret, r["state_root"].encode(), hashlib.sha256).digest()
            valid = hmac.compare_digest(expected, bytes(r["signature"]))
        else:
            valid = None
        out.append({"id": r["id"], "algo": r["algo"], "key_id": r["key_id"],
                    "valid": valid, "matches_current": r["state_root"] == current,
                    "trusted": r["key_id"] in trusted, "signed_at": r["signed_at"]})
    # A file is "trust-attested" only if some valid, current signature is from a trusted key.
    attested = any(s["valid"] and s["matches_current"] and s["trusted"] for s in out)
    status = "corrupted" if any(s["valid"] is False for s in out) else "ok"
    return {"status": status, "current_state_root": current,
            "trusted_keys": len(trusted), "trusted_attested": attested, "signatures": out}


def list_signatures(mf: MindFile) -> list[dict]:
    return [dict(r) for r in mf.conn.execute(
        "SELECT id, algo, key_id, state_root, signed_at, note FROM signatures ORDER BY id")]
