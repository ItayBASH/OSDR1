import logging
from tdm.paths import PACKAGE_ROOT

log_file = PACKAGE_ROOT / "log.txt"

logging.basicConfig(
    level=logging.INFO,  # global level i want logged (e.g from numpy)
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filename=log_file,
    filemode="w",
)

# want everything from my logger:
logger = logging.getLogger("tdm")
logger.setLevel(logging.DEBUG)
logger.debug("logger initialized")
