import logging
import os

PAGE_SIZE = 2 * 1024 * 1024  # 2MB

# Configuration constants for KVCacheManager
GPU_UTILIZATION = float(os.getenv("KVCACHED_GPU_UTILIZATION", "0.95"))
PAGE_PREALLOC_ENABLED = os.getenv("KVCACHED_PAGE_PREALLOC_ENABLED",
                                  "true").lower() == "true"
MIN_RESERVED_PAGES = int(os.getenv("KVCACHED_MIN_RESERVED_PAGES", "5"))
MAX_RESERVED_PAGES = int(os.getenv("KVCACHED_MAX_RESERVED_PAGES", "10"))
SANITY_CHECK = os.getenv("KVCACHED_SANITY_CHECK", "false").lower() == "true"

# Allow overriding the shared-memory segment name via env var so multiple
# kvcached deployments on one machine can coexist without collision.
DEFAULT_IPC_NAME = os.getenv("KVCACHED_IPC_NAME", "kvcached_mem_info")
SHM_DIR = "/dev/shm"

LOG_USE_COLOR = os.getenv("KVCACHED_LOG_COLOR", "true").lower() == "true"
_UNIFORM_COLOR = os.getenv("KVCACHED_LOG_COLOR_CODE", "\033[36m")

_LEVEL_TO_COLOR = {
    logging.DEBUG: "\033[36m",  # Cyan
    logging.INFO: "\033[32m",  # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}
_COLOR_RESET = "\033[0m"


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells


class ColorFormatter(logging.Formatter):
    """A logging formatter that injects ANSI colors based on the log level."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)

        color = _LEVEL_TO_COLOR.get(record.levelno, _UNIFORM_COLOR)

        prefix, sep, rest = formatted.partition("] ")

        if sep:
            prompt = f"{prefix}{sep}"
            return f"{color}{prompt}{_COLOR_RESET}{rest}"
        else:
            return f"{color}{formatted}{_COLOR_RESET}"


def get_log_level():
    level = os.getenv("KVCACHED_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def get_kvcached_logger(name: str = "kvcached") -> logging.Logger:
    logger = logging.getLogger(name)

    # Only add handler if none exists (prevents duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()

        fmt_str = (f"[{name}]"
                   "[%(levelname)s]"
                   "[%(asctime)s]"
                   "[%(filename)s:%(lineno)d] %(message)s")

        if LOG_USE_COLOR and handler.stream.isatty():
            formatter: logging.Formatter = ColorFormatter(
                fmt_str, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            formatter = logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(get_log_level())
        # Prevent propagation to inference engines; avoid duplicate messages
        logger.propagate = False

    return logger
