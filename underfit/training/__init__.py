"""Underfit raw-PyTorch training loop.

Replaces stable-audio-tools' Lightning training wrapper with a backend-
agnostic loop that works on top of either sat or sa3 model primitives.
"""
from .loop import run_training
