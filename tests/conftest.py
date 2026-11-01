def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow regression tests (reproduce old bugs on purpose)")
