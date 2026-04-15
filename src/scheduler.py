import asyncio
import json
from config import logger
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from croniter import croniter

from pydantic import BaseModel

_STORAGE_PATH = Path("./data/cron.json")
_TICK_INTERVAL = 5.0


class CronSchedule(BaseModel):
    type: Literal["at", "every", "cron"]
    at: int | None = None  # epoch ms (one-shot)
    every: int | None = None  # interval ms
    cron_expr: str | None = None  # e.g. "0 9 * * 1-5" (weekdays at 09:00)
    tz: str = "UTC"


class CronJob(BaseModel):
    id: str
    name: str
    schedule: CronSchedule
    message: str  # injected as user message when triggered
    delete_after_run: bool = False
    next_run: int = 0  # epoch ms
    last_run: int | None = None


class CronService:
    def __init__(self) -> None:
        self._storage = _STORAGE_PATH
        self._jobs: dict[str, CronJob] = {}
        self._task: asyncio.Task | None = None
        self._load()

    def add_job(
        self,
        schedule: CronSchedule,
        message: str,
        name: str = "",
    ) -> CronJob:
        job = CronJob(
            id=str(uuid.uuid4()),
            name=name,
            schedule=schedule,
            message=message,
        )
        job.next_run = self._compute_next_run(job)
        self._jobs[job.id] = job
        self._save()
        logger.info(f"added cron job {job.id[:8]} ({name or schedule.type})")
        return job

    def remove_job(self, job_id: str) -> None:
        # Accept full UUID or any unambiguous prefix — historical list output only showed id[:8].
        if job_id not in self._jobs:
            matches = [jid for jid in self._jobs if jid.startswith(job_id)]
            if len(matches) == 1:
                job_id = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"job id prefix {job_id!r} is ambiguous ({len(matches)} matches)")
        if not (job := self._jobs.pop(job_id, None)):
            raise ValueError(f"job {job_id!r} not found")
        self._save()
        logger.info(f"removed cron job {job_id} ({job.name})")

    def list_jobs(self) -> list[CronJob]:
        return list(self._jobs.values())

    async def start(self, on_trigger: Callable[[CronJob], Awaitable[None]]) -> None:
        self._task = asyncio.create_task(self._loop(on_trigger), name="cron-loop")
        logger.info("cron service started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("cron service stopped")

    async def _loop(self, on_trigger: Callable[[CronJob], Awaitable[None]]) -> None:
        while True:
            now = _now_ms()
            for job in [j for j in self._jobs.values() if j.next_run <= now]:
                asyncio.create_task(self._fire(job, on_trigger))
            await asyncio.sleep(_TICK_INTERVAL)

    async def _fire(self, job: CronJob, on_trigger: Callable[[CronJob], Awaitable[None]]) -> None:
        # Advance next_run (or pop) BEFORE awaiting the trigger. on_trigger can hold the session
        # lock for tens of seconds; if next_run stays in the past during that time, the tick loop
        # keeps enqueuing duplicate _fire tasks and floods the agent with backlogged turns once
        # the lock releases.
        job.last_run = _now_ms()
        if job.delete_after_run or job.schedule.type == "at":
            self._jobs.pop(job.id, None)
        else:
            try:
                job.next_run = self._compute_next_run(job)
            except ValueError:
                logger.error(f"cron job {job.id[:8]} has invalid schedule; removing")
                self._jobs.pop(job.id, None)
        self._save()

        try:
            await on_trigger(job)
        except Exception:
            logger.exception(f"cron job {job.id[:8]} raised")

    def _compute_next_run(self, job: CronJob) -> int:
        now = _now_ms()
        sched = job.schedule

        if sched.type == "at":
            if sched.at is None:
                raise ValueError("schedule type 'at' requires 'at' (epoch ms)")
            return sched.at

        if sched.type == "every":
            if sched.every is None:
                raise ValueError("schedule type 'every' requires 'every' (interval ms)")
            return now + sched.every

        if sched.cron_expr is None:
            raise ValueError("schedule type 'cron' requires 'cron_expr'")
        return self._next_cron(sched.cron_expr, sched.tz)

    def _next_cron(self, expr: str, tz: str) -> int:
        zone = ZoneInfo(tz)
        now_dt = datetime.now(tz=zone)
        it = croniter(expr, now_dt)
        next_dt = it.get_next(datetime)
        return int(next_dt.astimezone(timezone.utc).timestamp() * 1000)

    def _load(self) -> None:
        if not self._storage.exists():
            return
        try:
            data = json.loads(self._storage.read_text())
            for raw in data.get("jobs", []):
                job = CronJob.model_validate(raw)
                self._jobs[job.id] = job
            logger.debug(f"loaded {len(self._jobs)} cron jobs")
        except Exception as exc:
            logger.warning(f"failed to load cron storage: {exc}")

    def _save(self) -> None:
        self._storage.parent.mkdir(parents=True, exist_ok=True)
        data = {"jobs": [j.model_dump() for j in self._jobs.values()]}
        self._storage.write_text(json.dumps(data, indent=2))


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
