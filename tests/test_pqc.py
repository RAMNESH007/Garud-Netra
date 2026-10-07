"""
Automated post-quantum cryptography tests for Garud-Netra.

Tests use temporary ephemeral keys and files so the real
~/.garud_netra/keys/ keypair is never modified.
"""

from pathlib import Path

import oqs
import pytest

from security.pqc import (
    ALGORITHM,
    calculate_evidence_hash,
    canonicalize_evidence,
    sign_evidence,
    verify_signature,
)


def create_test_keypair(tmp_path):
    """
    Generate an ephemeral ML-DSA keypair for testing.

    The keys are stored only inside pytest's temporary directory.
    """

    private_key_path = (
        tmp_path / "private.key"
    )

    public_key_path = (
        tmp_path / "public.key"
    )

    with oqs.Signature(ALGORITHM) as signer:

        public_key = (
            signer.generate_keypair()
        )

        private_key = (
            signer.export_secret_key()
        )

    public_key_path.write_bytes(
        public_key
    )

    private_key_path.write_bytes(
        private_key
    )

    public_key_path.chmod(0o600)
    private_key_path.chmod(0o600)

    return (
        public_key_path,
        private_key_path,
    )


def sample_evidence():
    """Return deterministic test evidence."""

    return {
        "investigation_id": "TEST-CASE-001",
        "txid": "TEST-TX-001",
        "risk_score": 87,
        "anomaly_score": 91.5,
        "priority": "HIGH",
        "indicators": [
            "high network activity",
            "multiple destination IPs",
        ],
    }


def test_canonicalization_is_deterministic():
    """Equivalent dictionaries must produce identical bytes."""

    evidence_a = {
        "txid": "TEST-001",
        "risk_score": 87,
        "priority": "HIGH",
    }

    evidence_b = {
        "priority": "HIGH",
        "risk_score": 87,
        "txid": "TEST-001",
    }

    assert (
        canonicalize_evidence(evidence_a)
        == canonicalize_evidence(evidence_b)
    )


def test_evidence_hash_changes_after_modification():
    """Changing evidence must change its SHA-256 digest."""

    original = sample_evidence()

    modified = dict(
        original
    )

    modified["risk_score"] = 25

    assert (
        calculate_evidence_hash(original)
        != calculate_evidence_hash(modified)
    )


def test_valid_evidence_signature(tmp_path):
    """Correct evidence and signature must verify."""

    public_key, private_key = (
        create_test_keypair(tmp_path)
    )

    signature_file = (
        tmp_path / "signature.bin"
    )

    evidence = sample_evidence()

    sign_result = sign_evidence(
        evidence=evidence,
        signature_path=signature_file,
        private_key_path=private_key,
    )

    assert signature_file.is_file()
    assert sign_result["algorithm"] == ALGORITHM

    result = verify_signature(
        evidence=evidence,
        signature_path=signature_file,
        public_key_path=public_key,
    )

    assert result["valid"] is True
    assert result["status"] == "VALID"


def test_modified_evidence_is_rejected(tmp_path):
    """Modified evidence must fail signature verification."""

    public_key, private_key = (
        create_test_keypair(tmp_path)
    )

    signature_file = (
        tmp_path / "signature.bin"
    )

    original = sample_evidence()

    sign_evidence(
        evidence=original,
        signature_path=signature_file,
        private_key_path=private_key,
    )

    modified = dict(
        original
    )

    modified["risk_score"] = 25

    result = verify_signature(
        evidence=modified,
        signature_path=signature_file,
        public_key_path=public_key,
    )

    assert result["valid"] is False
    assert result["status"] == "INVALID"


def test_corrupted_signature_is_rejected(tmp_path):
    """A corrupted signature must fail verification."""

    public_key, private_key = (
        create_test_keypair(tmp_path)
    )

    signature_file = (
        tmp_path / "signature.bin"
    )

    corrupted_file = (
        tmp_path / "corrupted.bin"
    )

    evidence = sample_evidence()

    sign_evidence(
        evidence=evidence,
        signature_path=signature_file,
        private_key_path=private_key,
    )

    signature = bytearray(
        signature_file.read_bytes()
    )

    signature[0] ^= 0xFF

    corrupted_file.write_bytes(
        bytes(signature)
    )

    result = verify_signature(
        evidence=evidence,
        signature_path=corrupted_file,
        public_key_path=public_key,
    )

    assert result["valid"] is False
    assert result["status"] == "INVALID"


def test_missing_signature_is_rejected(tmp_path):
    """Verification must fail clearly when signature is absent."""

    public_key, _ = (
        create_test_keypair(tmp_path)
    )

    evidence = sample_evidence()

    with pytest.raises(
        FileNotFoundError
    ):
        verify_signature(
            evidence=evidence,
            signature_path=(
                tmp_path / "missing.bin"
            ),
            public_key_path=public_key,
        )
