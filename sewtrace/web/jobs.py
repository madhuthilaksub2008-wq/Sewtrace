"""Jobs for the web page: one pipeline run at a time, state kept in memory, folders cleaned by age."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sewtrace.web.errors import describe_error

log = logging.getLogger(__name__)


class Busy(RuntimeError):
    """A run is active. Paid calls never run in parallel."""


@dataclass
class Job:
    id: str
    kind: str  # "run" or "proof"
    dir: Path
    created: float
    state: str = "running"  # running, done, error
    step: str = "Starting"
    result: dict | None = None
    error: dict | None = None
    files: dict[str, Path] = field(default_factory=dict)  # name -> file, the only files served

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "step": self.step,
                "result": self.result, "error": self.error}


class JobStore:
    def __init__(self, root: Path, retention_min: float, scrub: Sequence[str] = ()):
        self.root = root
        self.retention_s = retention_min * 60
        self.scrub = tuple(scrub)
        self.jobs: dict[str, Job] = {}
        self.latest: dict[str, str] = {}  # kind -> job id, so a refreshed page finds its job
        self._guard = threading.Lock()  # protects ``jobs``
        self._run_lock = threading.Lock()  # held by the one active job
        self.root.mkdir(parents=True, exist_ok=True)
        self._purge_stale_folders()

    def _purge_stale_folders(self) -> None:
        """Delete job folders that an earlier process left in the root (a restart loses the job
        list, so ``purge`` cannot see them). Age is the folder modification time."""
        cutoff = time.time() - self.retention_s
        for folder in self.root.iterdir():
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)

    @property
    def busy(self) -> bool:
        return self._run_lock.locked()

    def get(self, job_id: str) -> Job | None:
        with self._guard:
            return self.jobs.get(job_id)

    def start(self, kind: str, work: Callable[[Job], dict], uploads: dict[str, bytes] | None = None) -> Job:
        """Run ``work(job)`` in a thread. Raises Busy when another job is active.
        ``uploads`` (file name -> bytes) are written into the job folder first."""
        if not self._run_lock.acquire(blocking=False):
            raise Busy("A run is already active. Wait for it to finish, then run again.")
        try:
            self.purge()
            job_id = uuid.uuid4().hex
            job = Job(job_id, kind, self.root / job_id, time.time())
            job.dir.mkdir(parents=True)
            for name, data in (uploads or {}).items():
                (job.dir / name).write_bytes(data)
            with self._guard:
                self.jobs[job.id] = job
                self.latest[kind] = job.id
            threading.Thread(target=self._run, args=(job, work), daemon=True).start()
        except BaseException:
            self._run_lock.release()
            raise
        return job

    def _run(self, job: Job, work: Callable[[Job], dict]) -> None:
        scrub = (f"{job.dir}/", str(job.dir), *self.scrub)
        try:
            job.result = work(job)
            job.state = "done"
        except Exception as err:  # the thread boundary: every failure becomes a message for the page
            job.error = describe_error(err, scrub)
            job.state = "error"
        finally:
            self._run_lock.release()

    def purge(self) -> None:
        """Delete finished job folders older than the retention time."""
        cutoff = time.time() - self.retention_s
        with self._guard:
            old = [j for j in self.jobs.values() if j.state != "running" and j.created < cutoff]
            for job in old:
                del self.jobs[job.id]
                for kind, job_id in list(self.latest.items()):
                    if job_id == job.id:
                        del self.latest[kind]
        for job in old:
            shutil.rmtree(job.dir, ignore_errors=True)
