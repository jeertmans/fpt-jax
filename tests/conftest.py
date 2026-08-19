from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--plot-failures",
        action="store_true",
        default=False,
        help=(
            "For failing benchmark dataset test cases (see "
            "tests/test_trace_rays.py), save a 3D plot comparing the "
            "expected path against the one returned by `trace_rays` (and, for "
            "cases sourced from a Sionna RT scene, the scene geometry itself) "
            "to --plot-dir. Requires the 'plots' dependency group."
        ),
    )
    parser.addoption(
        "--plot-dir",
        default=".benchmark-plots",
        help="Directory in which to save --plot-failures plots (default: %(default)s).",
    )
    parser.addoption(
        "--generate-sionna-dataset",
        action="store_true",
        default=False,
        help=(
            "Regenerate tests/data/sionna_paths.toml from Sionna RT city "
            "scenes (see tests/generate_sionna_dataset.py) before collecting "
            "tests, so that the benchmark tests then run against the fresh "
            "dataset. Requires the 'tests' dependency group (in particular, "
            "'sionna-rt'). Progress is logged via the "
            "'tests.generate_sionna_dataset' logger; add e.g. "
            "--log-cli-level=INFO to see it."
        ),
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Regenerate the Sionna dataset before collection, if requested.

    `pytest_configure` runs before pytest's own logging plugin starts
    capturing records, so `--log-cli-level` would not show anything logged
    from there; `pytest_sessionstart` runs just after and does.
    """
    if session.config.getoption("--generate-sionna-dataset"):
        from .generate_sionna_dataset import generate_sionna_dataset

        generate_sionna_dataset()


@pytest.fixture
def plot_failures(request: pytest.FixtureRequest) -> Path | None:
    """Directory to save failure plots to, or None if --plot-failures was not passed."""
    if not request.config.getoption("--plot-failures"):
        return None
    return Path(request.config.getoption("--plot-dir"))
