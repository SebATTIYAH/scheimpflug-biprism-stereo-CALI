"""Configure CLI defaults before loading numerical libraries."""
import os


def configure():
    # Respect explicit user settings; avoid nested BLAS pools by default.
    for name, value in (("OPENBLAS_NUM_THREADS", "1"),
                        ("OMP_NUM_THREADS", "1"), ("MPLBACKEND", "Agg")):
        os.environ.setdefault(name, value)


def main():
    configure()
    from .__main__ import main as dispatch
    return dispatch()
