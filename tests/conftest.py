import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="run tests that call OpenRouter / Modal (cost money)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
