"""Dataset ingestion: every supported shape normalizes, and bad rows are reported by index."""

from __future__ import annotations

import csv
import json

from kaggleui.ml.datasets import validate


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_jsonl_messages_and_token_estimate(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "hello there"}]},
            {"messages": [{"role": "system", "content": "be brief"},
                          {"role": "user", "content": "2+2?"},
                          {"role": "assistant", "content": "4"}]}]
    path = _write(tmp_path, "d.jsonl", "\n".join(json.dumps(r) for r in rows))
    report = validate(path)
    assert report.ok and report.samples == 2
    assert report.detected_format == "jsonl"
    assert report.token_estimate == 5  # 4 chars per token, floor per message


def test_json_list_sharegpt_and_alpaca_shapes(tmp_path):
    rows = [
        {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]},
        {"instruction": "do it", "input": "detail", "output": "done"},
        {"prompt": "p", "response": "r"},
    ]
    path = _write(tmp_path, "d.json", json.dumps(rows))
    report = validate(path)
    assert report.ok and report.samples == 3
    assert report.detected_format == "json:prompt-response"


def test_csv_columns_and_json_messages(tmp_path):
    import io
    turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["messages"])
    writer.writeheader()
    writer.writerow({"messages": json.dumps(turns)})
    assert validate(_write(tmp_path, "inline.csv", buf.getvalue())).ok

    path = _write(tmp_path, "flat.csv", "instruction,output\nname a city,Paris\n")
    report = validate(path)
    assert report.ok and report.samples == 1


def test_reports_every_shape_error_without_raising(tmp_path):
    good = {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}]}
    bad = [
        {"messages": [{"role": "assistant", "content": "first"}]},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user", "content": 5}]},
        {"messages": []},
        {"messages": [{"role": "user", "content": "no answer"}]},
        {"nope": 1},
    ]
    text = "\n".join(json.dumps(row) for row in [good, *bad])
    report = validate(_write(tmp_path, "mixed.jsonl", text))
    assert not report.ok and report.samples == 1
    assert len(report.errors) == len(bad)
    assert report.errors[0].startswith("sample 2: conversation starts with an assistant")
    assert "unknown role 'tool'" in report.errors[1]
    assert "content must be a string" in report.errors[2]
    assert "empty conversation" in report.errors[3]
    assert "no assistant turn" in report.errors[4]
    assert "unrecognized sample shape" in report.errors[5]


def test_duplicates_are_dropped_with_a_warning(tmp_path):
    row = {"messages": [{"role": "user", "content": "same"}, {"role": "assistant", "content": "same"}]}
    report = validate(_write(tmp_path, "d.jsonl", "\n".join([json.dumps(row), json.dumps(row)])), limit=10)
    assert report.ok and report.samples == 1
    assert report.warnings == ["1 duplicate sample dropped"]


def test_invalid_json_and_empty_input_are_errors(tmp_path):
    assert validate(_write(tmp_path, "bad.jsonl", "{nope")).errors[0].startswith("invalid JSON line")
    assert validate(_write(tmp_path, "bad.json", "[]")).samples == 0
    assert validate(_write(tmp_path, "empty.json", "[ ]")).errors == ["dataset contains no samples"]
    assert "unsupported format" in validate(_write(tmp_path, "x.txt", "hi"), "txt").errors[0]


def test_limit_only_reads_the_prefix(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "bad"}]}] * 3
    report = validate(_write(tmp_path, "d.jsonl", "\n".join(json.dumps(r) for r in rows)), limit=1)
    assert len(report.errors) == 1
