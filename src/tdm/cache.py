from joblib import Memory
from tdm.paths import PACKAGE_ROOT


def _init_peristent_cache():
    memory = Memory(PACKAGE_ROOT / ".persistent_cache", verbose=-1)
    return memory.cache


def _clear_cache():
    memory = Memory(PACKAGE_ROOT / ".persistent_cache", verbose=-1)
    memory.clear(warn=True)


persistent_cache = _init_peristent_cache()
