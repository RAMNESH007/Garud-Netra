"""
Post-quantum key management for Garud-Netra.

Uses ML-DSA-65 through liboqs-python.

Private/public keys are stored outside the project repository
under:

    ~/.garud_netra/keys/

Private key material is never printed or embedded in source code.
"""

from pathlib import Path

import oqs


ALGORITHM = "ML-DSA-65"

KEY_ROOT = (
    Path.home()
    / ".garud_netra"
    / "keys"
)

PUBLIC_KEY_FILE = KEY_ROOT / "public.key"
PRIVATE_KEY_FILE = KEY_ROOT / "private.key"


def ensure_key_directory():
    """
    Create the key directory and restrict its permissions.

    Returns
    -------
    Path
        Key directory path.
    """

    KEY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Owner-only permissions.
    KEY_ROOT.chmod(0o700)

    return KEY_ROOT


def validate_algorithm():
    """
    Confirm that the configured PQC algorithm is available.
    """

    enabled = oqs.get_enabled_sig_mechanisms()

    if ALGORITHM not in enabled:
        raise RuntimeError(
            f"{ALGORITHM} is not available in the installed "
            "liboqs build."
        )


def validate_key_file_permissions(
    key_path,
    private=False,
):
    """
    Validate key-file permissions.

    Private keys must be readable/writable only by the owner.
    Public keys are also kept owner-only to avoid unnecessary
    exposure and accidental modification.
    """

    key_path = Path(key_path)

    if not key_path.exists():
        return

    mode = key_path.stat().st_mode & 0o777

    if mode != 0o600:
        key_type = (
            "private"
            if private
            else "public"
        )

        raise PermissionError(
            f"{key_type.capitalize()} key has unsafe permissions: "
            f"{oct(mode)}. Expected 0o600."
        )


def generate_keypair():
    """
    Generate and securely store an ML-DSA-65 keypair.

    Returns
    -------
    tuple[Path, Path]
        Public-key path and private-key path.
    """

    validate_algorithm()

    key_dir = ensure_key_directory()

    # Prevent accidental overwriting of an existing keypair.
    if PUBLIC_KEY_FILE.exists():
        raise FileExistsError(
            f"Public key already exists: {PUBLIC_KEY_FILE}"
        )

    if PRIVATE_KEY_FILE.exists():
        raise FileExistsError(
            f"Private key already exists: {PRIVATE_KEY_FILE}"
        )

    with oqs.Signature(ALGORITHM) as signer:

        public_key = signer.generate_keypair()

        # liboqs-python exposes the secret key through export_secret_key.
        private_key = signer.export_secret_key()

    PUBLIC_KEY_FILE.write_bytes(
        public_key
    )

    PRIVATE_KEY_FILE.write_bytes(
        private_key
    )

    PUBLIC_KEY_FILE.chmod(0o600)
    PRIVATE_KEY_FILE.chmod(0o600)

    validate_key_file_permissions(
        PUBLIC_KEY_FILE,
        private=False,
    )

    validate_key_file_permissions(
        PRIVATE_KEY_FILE,
        private=True,
    )

    return (
        PUBLIC_KEY_FILE,
        PRIVATE_KEY_FILE,
    )


def load_public_key(
    key_path=PUBLIC_KEY_FILE,
):
    """Load the stored public key."""

    key_path = Path(key_path)

    if not key_path.is_file():
        raise FileNotFoundError(
            f"Public key not found: {key_path}"
        )

    validate_key_file_permissions(
        key_path,
        private=False,
    )

    return key_path.read_bytes()


def load_private_key(
    key_path=PRIVATE_KEY_FILE,
):
    """Load the stored private key without printing it."""

    key_path = Path(key_path)

    if not key_path.is_file():
        raise FileNotFoundError(
            f"Private key not found: {key_path}"
        )

    validate_key_file_permissions(
        key_path,
        private=True,
    )

    return key_path.read_bytes()


def keypair_exists():
    """Return True when both key files exist."""

    return (
        PUBLIC_KEY_FILE.is_file()
        and PRIVATE_KEY_FILE.is_file()
    )


def main():
    """Demonstrate safe key management."""

    print("=" * 60)
    print("GARUD-NETRA PQC KEY MANAGEMENT")
    print("=" * 60)

    print(
        f"Algorithm : {ALGORITHM}"
    )

    validate_algorithm()

    print(
        "✓ ML-DSA-65 is available"
    )

    ensure_key_directory()

    print(
        f"Key directory: {KEY_ROOT}"
    )

    if keypair_exists():

        print(
            "Keypair already exists."
        )

        validate_key_file_permissions(
            PUBLIC_KEY_FILE,
            private=False,
        )

        validate_key_file_permissions(
            PRIVATE_KEY_FILE,
            private=True,
        )

        public_key = load_public_key()
        private_key = load_private_key()

        print(
            f"✓ Public key loaded ({len(public_key)} bytes)"
        )

        print(
            f"✓ Private key loaded ({len(private_key)} bytes)"
        )

    else:

        public_path, private_path = (
            generate_keypair()
        )

        print(
            "✓ New ML-DSA-65 keypair generated"
        )

        print(
            f"✓ Public key: {public_path}"
        )

        print(
            f"✓ Private key: {private_path}"
        )

    print(
        "✓ Key files are outside the source repository"
    )

    print(
        "✓ Private key material was not printed"
    )

    print(
        "\nPQC key management completed successfully."
    )


if __name__ == "__main__":
    main()
