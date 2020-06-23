#!/bin/sh

set -e

. /venv/bin/activate

eval heroic-rl "$@"
