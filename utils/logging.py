import functools
import logging
from functools import lru_cache, partial
from importlib.util import find_spec
from itertools import accumulate
from typing import Callable, Literal

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def filter_debug(record: logging.LogRecord) -> bool:
    # filter out if debug level
    return record.levelno >= logging.INFO


JOIN_WITH_DOT = lambda left, right: left + "." + right


def filter_debug_w_whitelist(record: logging.LogRecord, whitelist: set) -> bool:
    # filter out if debug level, except for whitelisted
    if record.levelno >= logging.INFO:
        return True
    # first, find all loggers this message would have traversed
    # e.g. package.aa.bb -> {"package", "package.aa", "package.aa.bb"}
    loggers = set(accumulate(record.name.split("."), JOIN_WITH_DOT))
    # check for any matches using set intersection
    return bool(loggers & whitelist)


def configure_logging(logging_cfg: DictConfig) -> None:
    """Set up logging after hydra job logging has been configured. Debug
    messages will be logged to file output, while still being suppressed in
    console output. Arguments passed to hydra.verbose are respected.

    Config Parameters
    ----------
    always_log_debug_to_file : null | True | str | list[str]
        Debug messages from these modules (or all modules if True) should
        always be logged to file, logging level is INFO.
    always_suppress_debug_from : null | str | list[str]
        Debug messages from these modules should never be logged to file,
        even if hydra.verbose=True. This setting overrides
        `always_log_debug_to_file`.
    """
    whitelist: Literal[True] | str | list[str] = (
        logging_cfg.get("always_log_debug_to_file", []) or []
    )
    blacklist: str | list[str] = logging_cfg.get("always_suppress_debug_from", []) or []
    hydra_verbose: bool | str | list[str] = HydraConfig.get().verbose
    root_logger = logging.getLogger()
    root_debug = root_logger.level == logging.DEBUG

    # mimic behavior of hydra.verbose
    # if hydra.verbose is True or the root logger is already set to debug level,
    # then all debug messages should be logged to all outputs anyway, so nothing
    # needs to be changed
    if not (hydra_verbose is True or root_debug):
        # if hydra_verbose is False, or a (list of) module names, we set root
        # logger level to DEBUG and remove messages that are not from
        # whitelisted modules with a filter
        root_logger.setLevel("DEBUG")

        if hydra_verbose is False:
            # filter out all debug messages, since no modules set to verbose
            verbose_set = set()
            console_filter = filter_debug
        else:
            # filter out debug messages, except those from modules set to verbose
            verbose_set = (
                {hydra_verbose}
                if isinstance(hydra_verbose, str)
                else set(hydra_verbose)
            )
            console_filter = partial(filter_debug_w_whitelist, whitelist=verbose_set)

        console_handler = next(h for h in root_logger.handlers if h.name == "console")
        console_handler.addFilter(console_filter)

        # apply whitelist for printing debug messages to log file
        # if whitelist is True (print all debug messages to file), then just
        # don't filter messages in the file handler
        if whitelist is not True:
            # otherwise filter out debug messages, except those from modules
            # set to verbose or whitelisted
            whitelist_set = (
                {whitelist} if isinstance(whitelist, str) else set(whitelist)
            )
            verbose_or_whitelist = whitelist_set | verbose_set

            file_filter = partial(
                filter_debug_w_whitelist, whitelist=verbose_or_whitelist
            )

            file_handler = next(h for h in root_logger.handlers if h.name == "file")
            file_handler.addFilter(file_filter)

    # set loggers of "blacklisted" modules to INFO, so that DEBUG messages are
    # always suppressed, even if root logger is set to DEBUG
    for logger in {blacklist} if isinstance(blacklist, str) else set(blacklist):
        logging.getLogger(logger).setLevel("INFO")


@lru_cache(maxsize=None)
def warn_once(logger: logging.Logger, msg: str, *args, **kwargs):
    # TODO: replace with logging.captureWarnings, set up via config
    # https://docs.python.org/3/library/logging.html#logging.captureWarnings
    logger.warning(msg, *args, stacklevel=2, **kwargs)


def log_exception_and_finish_wandb(func: Callable) -> Callable:
    """Decorator to log uncaught exceptions raised by the decorated function
    using the logger for the module where the function is defined.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(func.__module__)
        exit_code = 0
        try:
            return func(*args, **kwargs)

        except Exception as e:
            log.exception(f"Exception in {func.__name__}: {e}")
            exit_code = 1
            raise

        finally:
            # always close wandb run (even if exception occurs so multirun won't fail)
            finished = try_finish_wandb(exit_code)
            if finished:
                log.debug(
                    f"Finished wandb run with {exit_code=} after {func.__name__} finished running"
                )

    return wrapper


def try_finish_wandb(exit_code: int = 0) -> bool:
    """Finish the current wandb run if it exists.

    Args:
        exit_code: The exit code to pass to wandb.finish(). Default is 0.
    """
    if find_spec("wandb"):  # check if wandb is installed
        import wandb

        if wandb.run:
            wandb.finish(exit_code=exit_code)
            return True
    return False
