#!/bin/bash
set -e
uv run dagster-webserver -h 0.0.0.0 -p 3000 --path-prefix /dagster &

# Start Dagster daemon in the background
uv run dagster-daemon run
