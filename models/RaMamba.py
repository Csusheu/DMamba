"""Canonical RaMamba model entry point.

The implementation remains in ``models.DMamba`` so existing experiment scripts
and checkpoints keep working after the public method rename.
"""

from models.DMamba import Model

__all__ = ["Model"]
