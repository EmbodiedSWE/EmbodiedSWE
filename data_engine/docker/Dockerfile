# Diversify-session image: the eval agent image plus what diversify sessions
# additionally need. Currently: usd-core (bare pxr for catalog_assets — the venv
# has no pip and is root-owned, so sessions cannot install it at runtime).
# The container keeps docker's default (bridge) networking: full outbound access.
#   docker build -t rb-diversify:2.1.216 data_engine/docker
ARG BASE=rb-l1-agent:2.1.216
FROM ${BASE}
RUN uv pip install --python /opt/venv/bin/python usd-core
