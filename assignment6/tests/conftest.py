"""Pytest config — registers hypothesis profiles for local vs CI runs.

Profiles:
  - default     : 200 examples per @given (configured in test files)
  - ci          : 500 examples per @given (set via HYPOTHESIS_PROFILE=ci)
  - dev         : 50 examples — fast iteration locally

Activate a profile with the ``HYPOTHESIS_PROFILE`` env var. The CI workflow
(``.github/workflows/assignment6-ci.yml``) sets ``HYPOTHESIS_PROFILE=ci`` to
extend coverage beyond the local 200-example default."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
