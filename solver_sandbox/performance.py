import contextlib
import functools
import inspect
import logging
import time

__all__ = ("profile", "timed", "solver_report_callback")

log = logging.getLogger(__name__)


@contextlib.contextmanager
def profile(label=None, log_level=logging.INFO):
    label = label if label is not None else 'context'
    tick = time.perf_counter()
    try:
        yield None
    finally:
        tock = time.perf_counter()
        log.log(level=log_level, msg=(f"{label} time: {tock-tick} seconds"))


def timed(f, log_level=None):
    log_level = logging.INFO if log_level is None else log_level

    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):
        tick = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            tock = time.perf_counter()
            log.log(
                level=log_level,
                msg=(f"{f.__name__} time: {tock-tick} seconds"))

    return wrapped_f


def solver_report_callback(log_level=logging.DEBUG):

    def report(_):
        frame = inspect.currentframe().f_back
        n_iter = frame.f_locals['iter_']
        residual = frame.f_locals['resid']
        log.log(log_level, f"iter: {n_iter} residual: {residual}")

    return report