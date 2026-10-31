#!/bin/sh
# A mounted volume is often owned by root and starts empty. Create the cache folder in it, fix
# the owner of the writable folder, then drop to the sewtrace user. The server never runs as root.
set -e
if [ "$(id -u)" = "0" ]; then
    mkdir -p "${SEWTRACE_CACHE_DIR:-/app/data/cache}"
    chown -R sewtrace:sewtrace /app/data
    # setpriv keeps root's HOME; libraries that write user config (ezdxf) need the sewtrace home.
    export HOME=/home/sewtrace
    exec setpriv --reuid=sewtrace --regid=sewtrace --init-groups "$@"
fi
exec "$@"
