"""Shared fixtures: load the repo's models (folders have spaces/digits) via importlib."""
import os
import sys
import importlib.util

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FOLDERS = ["0-D Hotspot", "1-D Lagrangian Hydro", "Rayleigh-Taylor", "Gain Model"]
for _f in _FOLDERS:                         # so `from hohlraum_viewfactor import ...` resolves
    _p = os.path.join(ROOT, _f)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(folder, filename, name):
    path = os.path.join(ROOT, folder, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def gain_model():
    return _load("Gain Model", "gain_model.py", "gain_model")


@pytest.fixture(scope="session")
def coupled():
    return _load("Gain Model", "coupled_gain.py", "coupled_gain")


@pytest.fixture(scope="session")
def rtmix():
    return _load("Gain Model", "rt_mix.py", "rt_mix")


@pytest.fixture(scope="session")
def hotspot():
    return _load("0-D Hotspot", "hotspot_0d.py", "hotspot_0d")


@pytest.fixture(scope="session")
def hydro():
    return _load("1-D Lagrangian Hydro", "lagrangian_1d.py", "lagrangian_1d")


@pytest.fixture(scope="session")
def crt():
    return _load("Rayleigh-Taylor", "convergent_rt.py", "convergent_rt")


@pytest.fixture(scope="session")
def viewfactor():
    return _load("Rayleigh-Taylor", "hohlraum_viewfactor.py", "hohlraum_viewfactor")


@pytest.fixture(scope="session")
def asym():
    return _load("Rayleigh-Taylor", "hohlraum_asymmetry.py", "hohlraum_asymmetry")


@pytest.fixture(scope="session")
def hohlraum_fwd(asym):
    return asym.build_forward()


@pytest.fixture(scope="session")
def crt_P0(crt):
    return crt.tune_P0()


@pytest.fixture(scope="session")
def dashboard():
    return _load("Gain Model", "dashboard.py", "dashboard")
