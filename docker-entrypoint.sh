#!/bin/sh
# A mounted volume is often owned by root. Fix the owner of the two writable folders,
# then drop to the sewtrace user. The server never runs as root.
set -e
if [ "$(id -u)" = "0" ]; then
    chown -R sewtrace:sewtrace /app/data /app/cache
    exec setpriv --reuid=sewtrace --regid=sewtrace --init-groups "$@"
fi
exec "$@"
