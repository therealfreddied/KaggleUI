"""Durable jobs: crash recovery, cancel-before-start and log paging."""

import sys

from kaggleui.workbench.jobs import JobStore, read_log
from kaggleui.workbench.runner import JobRunner


def test_interrupted_jobs_are_recoverable_after_restart(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    queued = store.create("llamacpp.install", {"cmd": ["true"]})
    running = store.create("llamacpp.download", {"cmd": ["true"]})
    store.update(running.id, state="running", pid=1234)
    store.close()

    reopened = JobStore(db)
    assert reopened.recover_interrupted() == 2
    assert reopened.get(running.id).state == "interrupted"
    assert "restarted" in (reopened.get(running.id).error or "")
    assert reopened.get(queued.id).state == "interrupted"
    reopened.close()


def test_cancel_before_start_is_immediate(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    queued = store.create("llamacpp.fetch", {"cmd": ["true"]})
    record = store.request_cancel(queued.id)
    assert record.state == "cancelled"
    store.close()


async def test_runner_records_events_and_exit(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create("stage", {"cmd": ["x"]}, logs_path=str(tmp_path / "stage.log"))
    runner = JobRunner(store, job, [sys.executable, "-c",
                                    "print('KUI_EVENT ' + '{\"progress\": 0.5, \"message\": \"half\"}'); print('done')"],
                       log_path=str(tmp_path / "stage.log"))
    finished = await runner.run()
    assert finished.state == "succeeded"
    assert finished.progress == 0.5
    assert "done" in "\n".join(read_log(finished.logs_path))
    store.close()


async def test_runner_failure_keeps_tail(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create("stage", {"cmd": ["x"]})
    runner = JobRunner(store, job, [sys.executable, "-c", "raise SystemExit(3)"])
    finished = await runner.run()
    assert finished.state == "failed"
    assert "3" in (finished.error or "")
    store.close()


def test_read_log_pages_by_line(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(10)) + "\n")
    assert read_log(log, 2, 3) == ["line 2", "line 3", "line 4"]
    assert read_log(None) == []
