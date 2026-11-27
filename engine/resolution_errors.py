"""Exceptions shared by resolution proposals and their verifiers."""


class GiveUp(Exception):
    """The procedure cannot continue; nothing is claimed."""


def computation_limits():
    """Exceptions raised when a computation exceeds its internal or process budget."""
    from subprocess import TimeoutExpired
    from .polynomial import LimitError
    return LimitError, TimeoutExpired


def rejected_proposals():
    """Exceptions that decline a proposed chart or its verification."""
    from .polynomial import InputError
    return (GiveUp, InputError, ArithmeticError, *computation_limits())
