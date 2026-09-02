"""Added by the shared-layer shim.

G-Designer imports `GDesigner.utils.log` from readers.py and
tools/coding/executor_factory.py but never ships the module, so the package
cannot be imported at all. CARD, its fork, restores exactly this file; the
content below matches CARD's.
"""
import logging

logger = logging.getLogger("GDesigner")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)
