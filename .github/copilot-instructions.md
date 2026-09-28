# FireRescueAI workspace guidance

- This repository is a Python 3.12 research prototype for camera-based person detection and tracking.
- Preserve the explicit safety boundary: detections are unverified, require human review, and must not be presented as emergency response decisions.
- Do not add drone flight-control, autonomous navigation, dispatch, or RGB-through-smoke claims.
- Keep the thermal camera integration as an interface only until a separately validated adapter is requested.
- Prefer focused tests using prerecorded or synthetic footage and keep runtime dependencies in `requirements.txt`.
