"""Workflow entrypoints.

Each module here is one GitHub Actions job, run as ``python -m tobacco.jobs.<name>``
with ``PYTHONPATH=src``. They are the only modules that write to disk or the
network on their own initiative; everything else is a library.

All four are **idempotent**. Actions cron is best-effort and can fire late or
twice, so every job upserts by key and re-running one is a no-op.
"""
