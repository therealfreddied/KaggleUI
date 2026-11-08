"""Pinned BigMoeOnEdge release: one tarball per platform, carrying ``bmoe-server`` and its ggml backends.

The tarball must stay a single folder: the server loads ``libggml-*`` from beside itself. Sizes and
digests are for the published v0.40.1 assets; a newer release needs both re-checked before the pin moves.
"""

from __future__ import annotations

BMOE_RELEASE = "v0.40.1"
RELEASE_BASE = f"https://github.com/Helldez/BigMoeOnEdge/releases/download/{BMOE_RELEASE}"

# arch -> (asset name, sha256, minimum glibc, note)
ASSETS: dict[str, tuple[str, str, tuple[int, int], str]] = {
    "x86_64": (
        f"bmoe-cli-{BMOE_RELEASE}-linux-x86_64.tar.gz",
        "1538b8007e61d83338c22e854767f1f81b22899b8e07a0fa1d784bd9b8f9e16b",
        (2, 35),
        "Ubuntu 22.04 or newer",
    ),
    "aarch64": (
        f"bmoe-cli-{BMOE_RELEASE}-linux-aarch64.tar.gz",
        "a81f9ecd23e77f521abe3a173380f9076c9ee79f57927b5a9659d849067f4e9d",
        (2, 38),
        "Ubuntu 24.04, Debian 13 or Fedora 39 or newer",
    ),
}


def asset(arch: str) -> tuple[str, str, tuple[int, int], str]:
    try:
        return ASSETS[arch]
    except KeyError:
        raise ValueError(f"BigMoeOnEdge publishes no Linux build for {arch}") from None
