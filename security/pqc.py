"""
Post-Quantum Cryptography evidence signing and verification.

Garud-Netra uses ML-DSA-65 through liboqs-python to protect
investigation evidence against unauthorized modification.

Important:
- PQC is used only for evidence integrity/authenticity.
- PQC is NOT part of anomaly detection, clustering, graph
  construction, or feature engineering.
- Private keys remain outside the project repository.
- No network calls are made by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import oqs

from security.key_manager import (
    ALGORITHM,
    load_private_key,
    load_public_key,
)


# -------------------------------------------------------------------
# Canonical evidence serialization
# -------------------------------------------------------------------

def canonicalize_evidence(
    evidence: Any,
) -> bytes:
    """
    Convert investigation evidence into deterministic JSON bytes.

    Deterministic serialization is essential because verification
    must recreate exactly the same byte representation that was
    originally signed.

    Rules:
    - keys sorted
    - compact separators
    - UTF-8 encoding
    - Unicode preserved
    """

    try:
        canonical_json = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Evidence could not be serialized into canonical JSON."
        ) from exc

    return canonical_json.encode(
        "utf-8"
    )


# -------------------------------------------------------------------
# Evidence digest
# -------------------------------------------------------------------

def calculate_evidence_hash(
    evidence: Any,
) -> str:
    """
    Calculate SHA-256 over canonical evidence bytes.

    The digest is used as an integrity fingerprint. The PQC
    signature itself is generated over the canonical evidence bytes.
    """

    canonical_bytes = canonicalize_evidence(
        evidence
    )

    return hashlib.sha256(
        canonical_bytes
    ).hexdigest()


# -------------------------------------------------------------------
# Algorithm validation
# -------------------------------------------------------------------

def validate_pqc_algorithm():
    """Confirm that the configured ML-DSA mechanism is available."""

    enabled = (
        oqs.get_enabled_sig_mechanisms()
    )

    if ALGORITHM not in enabled:
        raise RuntimeError(
            f"{ALGORITHM} is not available in the installed "
            "liboqs build."
        )


# -------------------------------------------------------------------
# Signing
# -------------------------------------------------------------------

def sign_evidence(
    evidence: Any,
    signature_path: str | Path,
    private_key_path: str | Path | None = None,
) -> dict:
    """
    Sign investigation evidence using ML-DSA-65.

    Parameters
    ----------
    evidence:
        JSON-serializable investigation evidence.

    signature_path:
        Destination for the binary PQC signature.

    private_key_path:
        Optional custom private-key path.
        Defaults to the key managed by key_manager.py.

    Returns
    -------
    dict
        Signature metadata containing:
        - algorithm
        - evidence_sha256
        - signature_file
        - canonical_size_bytes
        - signature_size_bytes
    """

    validate_pqc_algorithm()

    canonical_bytes = canonicalize_evidence(
        evidence
    )

    evidence_hash = hashlib.sha256(
        canonical_bytes
    ).hexdigest()

    if private_key_path is None:
        private_key = load_private_key()
    else:
        private_key = load_private_key(
            private_key_path
        )

    signature_path = Path(
        signature_path
    )

    signature_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with oqs.Signature(
        ALGORITHM,
        secret_key=private_key,
    ) as signer:

        signature = signer.sign(
            canonical_bytes
        )

    signature_path.write_bytes(
        signature
    )

    return {
        "algorithm": ALGORITHM,
        "evidence_sha256": evidence_hash,
        "signature_file": str(
            signature_path
        ),
        "canonical_size_bytes": len(
            canonical_bytes
        ),
        "signature_size_bytes": len(
            signature
        ),
    }


# -------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------

def verify_signature(
    evidence: Any,
    signature_path: str | Path,
    public_key_path: str | Path | None = None,
) -> dict:
    """
    Verify a PQC signature against investigation evidence.

    Returns a structured result rather than only True/False so the
    dashboard can later display useful verification information.
    """

    validate_pqc_algorithm()

    signature_path = Path(
        signature_path
    )

    if not signature_path.is_file():
        raise FileNotFoundError(
            f"Signature file not found: {signature_path}"
        )

    canonical_bytes = canonicalize_evidence(
        evidence
    )

    evidence_hash = hashlib.sha256(
        canonical_bytes
    ).hexdigest()

    signature = (
        signature_path.read_bytes()
    )

    if public_key_path is None:
        public_key = load_public_key()
    else:
        public_key = load_public_key(
            public_key_path
        )

    with oqs.Signature(
        ALGORITHM,
    ) as verifier:

        valid = verifier.verify(
            canonical_bytes,
            signature,
            public_key,
        )

    return {
        "valid": bool(valid),
        "status": (
            "VALID"
            if valid
            else "INVALID"
        ),
        "algorithm": ALGORITHM,
        "evidence_sha256": evidence_hash,
        "signature_file": str(
            signature_path
        ),
    }


# -------------------------------------------------------------------
# Convenience helper
# -------------------------------------------------------------------

def sign_case_evidence(
    case_directory: str | Path,
    evidence: Any,
) -> dict:
    """
    Sign evidence for a case directory.

    Expected package layout:

        case_directory/
        ├── evidence.json
        └── signature.bin
    """

    case_directory = Path(
        case_directory
    )

    case_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_file = (
        case_directory
        / "evidence.json"
    )

    signature_file = (
        case_directory
        / "signature.bin"
    )

    # Write canonical evidence so the exact representation used for
    # signing is persisted with the case.
    canonical_bytes = canonicalize_evidence(
        evidence
    )

    evidence_file.write_bytes(
        canonical_bytes
    )

    result = sign_evidence(
        evidence=evidence,
        signature_path=signature_file,
    )

    result["evidence_file"] = str(
        evidence_file
    )

    return result


# -------------------------------------------------------------------
# Case verification helper
# -------------------------------------------------------------------

def verify_case_evidence(
    case_directory: str | Path,
) -> dict:
    """
    Verify a stored case package.

    Expected package layout:

        case_directory/
        ├── evidence.json
        └── signature.bin
    """

    case_directory = Path(
        case_directory
    )

    evidence_file = (
        case_directory
        / "evidence.json"
    )

    signature_file = (
        case_directory
        / "signature.bin"
    )

    if not evidence_file.is_file():
        raise FileNotFoundError(
            f"Evidence file not found: {evidence_file}"
        )

    if not signature_file.is_file():
        raise FileNotFoundError(
            f"Signature file not found: {signature_file}"
        )

    try:
        evidence = json.loads(
            evidence_file.read_text(
                encoding="utf-8"
            )
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            f"Evidence JSON is invalid: {evidence_file}"
        ) from exc

    result = verify_signature(
        evidence=evidence,
        signature_path=signature_file,
    )

    result["evidence_file"] = str(
        evidence_file
    )

    return result


# -------------------------------------------------------------------
# Demonstration
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("GARUD-NETRA PQC EVIDENCE SIGNING")
    print("=" * 60)

    validate_pqc_algorithm()

    print(
        f"Algorithm : {ALGORITHM}"
    )

    print(
        "✓ PQC algorithm available"
    )

    print(
        "\nThe module is ready for evidence signing."
    )
