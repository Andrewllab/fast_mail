import logging
from functools import partial
from itertools import accumulate
from typing import Literal

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
        always be logged to file, even if suppressed in console output.
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

    if hydra_verbose is not True:
        # if hydra_verbose is True, all debug messages should be logged to all
        # outputs anyway, so nothing needs to be changed
        # if hydra_verbose is not True, we set root logger level to DEBUG and
        # remove messages that are not from whitelisted modules with a filter
        root_logger = logging.getLogger()
        root_logger.setLevel("DEBUG")

        if hydra_verbose is False:
            verbose_set = set()
            # only consider message level, since no loggers set to verbose
            console_filter = filter_debug
        else:
            verbose_set = (
                {hydra_verbose}
                if isinstance(hydra_verbose, str)
                else set(hydra_verbose)
            )
            # filter out if debug level, except loggers set to verbose
            console_filter = partial(filter_debug_w_whitelist, whitelist=verbose_set)

        console_handler = next(h for h in root_logger.handlers if h.name == "console")
        console_handler.addFilter(console_filter)

        if whitelist is not True:
            # if whitelist is True (print all debug messages to file), then
            # just don't filter messages in the file handler

            whitelist_set = (
                {whitelist} if isinstance(whitelist, str) else set(whitelist)
            )
            verbose_or_whitelist = whitelist_set | verbose_set

            # filter out if debug level, except loggers set to verbose or whitelisted
            file_filter = partial(
                filter_debug_w_whitelist, whitelist=verbose_or_whitelist
            )

            file_handler = next(h for h in root_logger.handlers if h.name == "file")
            file_handler.addFilter(file_filter)

    # set loggers of "blacklisted" modules to INFO, so that DEBUG messages are
    # always suppressed, even if root logger is set to DEBUG
    for logger in {blacklist} if isinstance(blacklist, str) else set(blacklist):
        logging.getLogger(logger).setLevel("INFO")
