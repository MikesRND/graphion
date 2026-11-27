"""Spike of an inspection layer between a Graphion model and its user interfaces.

A session speaks one protocol to every client and reaches a model only through
an adapter. The live adapter drives a Python/NumPy frame model; the replay
adapter serves the same calls from a recording, with no model present.
"""
