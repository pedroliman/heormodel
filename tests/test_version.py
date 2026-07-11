import heormodel


def test_version_reports_installed_metadata():
    # `version("heormodel")` must name the installed distribution, otherwise
    # the lookup misses and `__version__` falls back to the "0.0.0" sentinel.
    assert heormodel.__version__ != "0.0.0"
