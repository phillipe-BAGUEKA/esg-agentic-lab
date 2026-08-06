import esg_agentic_lab


def test_package_version() -> None:
    """The installed package exposes the expected initial version."""
    assert esg_agentic_lab.__version__ == "0.1.0"
