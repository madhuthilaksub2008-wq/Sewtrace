# Deploy the Sewtrace test page

The test page lets a tester run Sewtrace in a browser. It is a test tool. It is not the demo.

**WARNING:** REAL mode makes paid calls to the Meta Model API. Every run costs money. REAL mode needs a password. The server refuses to start in REAL mode without one.

**WARNING:** The spend ledger is the file `data/ledger.csv`. If the ledger is lost, the budget guard starts again at $0. Put `data/` on a persistent volume in REAL mode. In the Docker image, REAL mode refuses to start unless `/app/data` is a mount point. See "The ledger volume".

## Modes

| Mode | When | Cost |
| --- | --- | --- |
| MOCK | Default. No `META_API_KEY`, or `SEWTRACE_MOCK=1`. | None. Only the sample tee works. |
| REAL | `META_API_KEY` is set and `SEWTRACE_MOCK=0`. | Paid calls. |
| OFFLINE | `SEWTRACE_OFFLINE=1`. | None. Answers come from the cache only. |

The page shows the mode, the reasoning model and the ledger total in the bar at the top.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `META_API_KEY` | The Meta Model API key. Set it as a secret on the host. Never put it in the image or in git. |
| `SEWTRACE_MOCK` | `1` is mock mode (default). `0` is REAL mode. |
| `SEWTRACE_WEB_PASSWORD` | The access password. Required in REAL mode. Optional in MOCK mode. |
| `SEWTRACE_OFFLINE` | `1` is cache-only mode. |
| `SEWTRACE_REQUIRE_LEDGER_MOUNT` | The image sets it to `1`. REAL mode then refuses to start when `/app/data` is not a mount point. Do not unset it on a public host. |
| `SEWTRACE_CACHE_DIR` | The image sets it to `/app/data/cache`, inside the volume. |
| `PORT` | The port in the Docker image. Default 8000. Render sets it for you. |

The ledger path is not an environment variable. It is `api.ledger_path` in `sewtrace/config.toml`. This rule stops a second ledger that starts at $0.

The web limits are in `sewtrace/config.toml` under `[web]`: `max_upload_mb`, `max_pixels`, `job_retention_min`, `auth_fail_delay_s` and `port`.

## Access control

The page uses HTTP Basic auth when `SEWTRACE_WEB_PASSWORD` is set. The browser asks for a password. Enter any user name and the password. The server compares the password in constant time. A wrong password waits `auth_fail_delay_s` (1 s) before the answer, which slows guessing. Use HTTPS on the host, because Basic auth sends the password with each request. The path `/healthz` has no password, so the host can check that the server is up.

Every POST request needs the header `X-Sewtrace: 1`. The page sends it. A form on another site cannot send it, so it cannot start a paid run with the password that your browser remembers.

### Make a password

A public host gets guessed. Use a long random password.

1. Make the password.

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```

2. Set the output as `SEWTRACE_WEB_PASSWORD` on the host.
3. Do not use this password anywhere else.

## Run locally

1. Create the environment and install the package.

   ```bash
   python -m venv .venv && . .venv/bin/activate
   pip install -e .
   ```

2. Start the server.

   ```bash
   sewtrace serve
   ```

3. Open `http://127.0.0.1:8000`.
4. Click "Try a sample tee".

The server writes the sample tee to `data/synth` when it is missing.

To use REAL mode locally, install the real extra, set the variables and start the server.

```bash
pip install -e '.[real]'
export META_API_KEY=...          # your key
export SEWTRACE_MOCK=0
export SEWTRACE_WEB_PASSWORD=...  # choose a password
sewtrace serve
```

## Run with Docker

1. Build the image.

   ```bash
   docker build -t sewtrace .
   ```

   For REAL mode, add the SAM decoder: `docker build --build-arg INSTALL_REAL=1 -t sewtrace .`. The default build does not install it, because its version is not yet verified (see `pyproject.toml`).

2. Start the container in MOCK mode.

   ```bash
   docker run --rm -p 8000:8000 sewtrace
   ```

3. Open `http://127.0.0.1:8000`.

For REAL mode, pass the secrets and mount a volume for the ledger.

```bash
docker run --rm -p 8000:8000 \
  -e META_API_KEY=... -e SEWTRACE_MOCK=0 -e SEWTRACE_WEB_PASSWORD=... \
  -v sewtrace-data:/app/data \
  sewtrace
```

The container starts as root only to create the cache folder and to fix the owner of `/app/data`. Then it drops to the user `sewtrace`. The server never runs as root.

## The ledger volume

The code finds the ledger at `<repo root>/data/ledger.csv`. In the image the repo root is `/app`. So the ledger is `/app/data/ledger.csv`. The cache is `/app/data/cache`, in the same folder. Use one volume for both. Render allows one disk for each service, and a Fly Machine mounts one volume.

1. Create a persistent volume or disk on the host.
2. Mount it at `/app/data`.
3. Start the server in REAL mode.
4. Read the first lines of the server log. Look for the line that starts with `ledger`. It shows the ledger path, the total spent and the cache folder. After a redeploy the total must be the same as before.
5. Check the file `ledger.csv` after the first REAL run. It must still exist after a redeploy.

The mount check works like this. In the image, REAL mode calls `os.path.ismount("/app/data")`. If the answer is false, the server stops with the message "Attach a persistent volume". The check relies on the host showing the volume as a mount point. Render disks and Fly volumes should (this was not tested here). If your host mounts the volume in a different way, the server will refuse to start. Do not unset `SEWTRACE_REQUIRE_LEDGER_MOUNT` to get around this. Find out how the host mounts the volume instead. Outside Docker (`sewtrace serve` on your computer) the check is off.

Cached answers cost nothing to replay, and they let OFFLINE mode work after a redeploy. Never clear the cache to get a fresh result.

## Deploy on Render

1. Push the repository to GitHub.
2. In Render, click "New", then "Web Service".
3. Choose the repository. Set "Runtime" to "Docker". Render finds the `Dockerfile`.
4. To use REAL mode, add the environment variable `INSTALL_REAL` with the value `1`. Render passes environment variables to the Docker build, so the build gets it as the build argument. Check the build log to see that the real extra was installed.
5. Add one persistent disk. Set the mount path to `/app/data`. A disk needs a paid instance type. Without a disk, REAL mode does not start.
6. Add these environment variables:
   - `SEWTRACE_WEB_PASSWORD`: a long random password (see "Make a password").
   - `SEWTRACE_MOCK`: `1` for a test without cost. `0` for REAL mode.
   - `META_API_KEY`: your key. Needed only in REAL mode.
7. Set the health check path to `/healthz`.
8. Click "Create Web Service".
9. Open the URL that Render shows. Log in with the password.
10. Run the sample tee. Make sure that the mode in the bar is correct.

## Deploy on any container host

Fly.io, Railway and similar hosts work the same way.

1. Build the image from the `Dockerfile`, or let the host build it.
2. Set the three environment variables as secrets.
3. Mount one persistent volume at `/app/data`.
4. Route HTTPS traffic to port 8000, or to the value of `PORT`.
5. Set the health check path to `/healthz`.
6. Run only one instance. Two instances would each keep their own run lock and their own ledger copy.

## Safety rules of the page

- Only one run is active at a time. A second run gets the message "A run is already active".
- Every run goes through `MetaClient`: budget guard, ledger and cache.
- The page accepts JPEG and PNG files up to `max_upload_mb` for each photo.
- A photo may have at most `max_pixels` (50 million) pixels. The server reads the size from the file header and refuses a larger photo before it decodes it.
- The photos that go to the Muse model are decoded and encoded again as JPEG. The EXIF data (GPS position, camera) is not sent.
- A proof photo is checked against the pattern of the last successful run. The page shows which run. A failed run clears the pattern.
- Uploaded photos are stored in a temporary folder for each job. The server deletes the folder after `job_retention_min` minutes.
- The server never sends the API key to the browser and never writes it to a log.
- In MOCK mode, uploaded photos are refused. The mock SAM works only with the synthetic sample tee.
- Do not upload photos of people. Use photos of a t-shirt on a plain surface.
