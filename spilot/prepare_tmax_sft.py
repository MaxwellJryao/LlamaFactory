#!/usr/bin/env python3
"""Convert the TMax OpenAI trajectory parquet into LlamaFactory ShareGPT data.

The source stores Qwen thinking in ``reasoning_content`` and commonly ends a
trajectory with a tool response. LlamaFactory's legacy OpenAI converter drops
``reasoning_content`` and visible assistant content on tool-call turns. This
converter preserves both and removes only trailing prompt/tool messages that
have no assistant target.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONVERTER_VERSION = "tmax-openai-to-sharegpt-v3"
DATASET_NAME = "spilot_skill_tmax_sft_only_success"
PROMPT_ROLES = {"human", "observation"}
TARGET_ROLES = {"gpt", "function_call"}
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field(
            "conversations",
            pa.list_(pa.struct([pa.field("from", pa.string()), pa.field("value", pa.string())])),
        ),
        pa.field("tools", pa.string()),
    ]
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_tools(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError(f"tools must be a JSON list, got {type(parsed).__name__}")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, list):
        raise ValueError(f"tool_calls must be a list, got {type(value).__name__}")

    calls = []
    for raw_call in value:
        if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
            raise ValueError(f"invalid tool call: {raw_call!r}")
        function = raw_call["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool call has no function name: {raw_call!r}")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Preserve unusual string arguments rather than dropping a sample.
                pass
        calls.append({"name": name, "arguments": arguments})
    return calls


def _assistant_value(message: dict[str, Any], tool_calls: list[dict[str, Any]]) -> str:
    reasoning = _text(message.get("reasoning_content")).strip()
    visible = _text(message.get("content"))
    if reasoning:
        # Some source turns retain only the closing tag after their opening tag
        # was separated into reasoning_content by the serving stack.
        visible = visible.replace("</think>", "")
    value = f"<think>\n{reasoning}\n</think>\n\n" if reasoning else ""
    value += visible

    if tool_calls:
        if value and not value.endswith("\n"):
            value += "\n"
        call_json = json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
        # FunctionFormatter recognizes this wrapper, preserves text/thinking before
        # it, and emits the target Qwen template's canonical tool-call syntax.
        value += f"<tool_call>{call_json}</tool_call>"
    return value


def _convert_messages(messages: Any, row_index: int) -> tuple[list[dict[str, str]] | None, dict[str, int]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index}: messages must be a non-empty list")

    conversations: list[dict[str, str]] = []
    stats = {"reasoning_turns": 0, "tool_call_turns": 0, "trimmed_tail_messages": 0}
    cursor = 0
    if messages[0].get("role") == "system":
        conversations.append({"from": "system", "value": _text(messages[0].get("content"))})
        cursor = 1

    while cursor < len(messages):
        message = messages[cursor]
        if not isinstance(message, dict):
            raise ValueError(f"row {row_index}: message {cursor} is not an object")
        role = message.get("role")

        if role == "user":
            conversations.append({"from": "human", "value": _text(message.get("content"))})
        elif role == "assistant":
            tool_calls = _normalize_tool_calls(message.get("tool_calls"))
            if _text(message.get("reasoning_content")).strip():
                stats["reasoning_turns"] += 1
            if tool_calls:
                stats["tool_call_turns"] += 1
            conversations.append(
                {
                    "from": "function_call" if tool_calls else "gpt",
                    "value": _assistant_value(message, tool_calls),
                }
            )
        elif role == "tool":
            # Parallel tool calls can produce consecutive tool responses. The
            # outer Qwen template supplies the first/last response tags.
            tool_responses = [_text(message.get("content"))]
            while cursor + 1 < len(messages) and messages[cursor + 1].get("role") == "tool":
                cursor += 1
                tool_responses.append(_text(messages[cursor].get("content")))
            conversations.append(
                {
                    "from": "observation",
                    "value": "\n</tool_response>\n<tool_response>\n".join(tool_responses),
                }
            )
        elif role == "system":
            raise ValueError(f"row {row_index}: system message is only supported at index 0")
        else:
            raise ValueError(f"row {row_index}: unsupported role {role!r} at message {cursor}")
        cursor += 1

    body_start = 1 if conversations and conversations[0]["from"] == "system" else 0
    while len(conversations) > body_start and conversations[-1]["from"] in PROMPT_ROLES:
        conversations.pop()
        stats["trimmed_tail_messages"] += 1

    body = conversations[body_start:]
    if not body or len(body) % 2 != 0:
        raise ValueError(f"row {row_index}: converted conversation has {len(body)} non-system messages")
    for turn_index, message in enumerate(body):
        allowed = PROMPT_ROLES if turn_index % 2 == 0 else TARGET_ROLES
        if message["from"] not in allowed:
            raise ValueError(f"row {row_index}: role {message['from']!r} is invalid at converted index {turn_index}")
        # Drop trajectories that contain an empty assistant-target turn (e.g. a
        # "format error" recovery step). Training on an empty response is harmful,
        # and the empty turn cannot be removed in isolation without breaking the
        # human/assistant alternation, so the whole row is skipped.
        if turn_index % 2 == 1 and not message["value"].strip():
            return None, stats
    return conversations, stats


def _fingerprint(input_path: Path) -> dict[str, Any]:
    stat = input_path.stat()
    return {
        "converter_version": CONVERTER_VERSION,
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_dataset_info(output_dir: Path, parquet_name: str) -> None:
    dataset_info = {
        DATASET_NAME: {
            "file_name": parquet_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        }
    }
    _write_json_atomic(output_dir / "dataset_info.json", dataset_info)


def prepare(input_path: Path, output_dir: Path, force: bool = False) -> Path:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.parquet"
    metadata_path = output_dir / "prepare_metadata.json"
    lock_path = output_dir / ".prepare.lock"
    expected_fingerprint = _fingerprint(input_path)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not force and output_path.is_file() and metadata_path.is_file():
            try:
                cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if cached.get("fingerprint") == expected_fingerprint:
                _write_dataset_info(output_dir, output_path.name)
                print(f"Prepared dataset cache is current: {output_path}", flush=True)
                return output_path

        parquet_file = pq.ParquetFile(input_path)
        required_columns = {"messages", "tools"}
        missing_columns = required_columns.difference(parquet_file.schema_arrow.names)
        if missing_columns:
            raise ValueError(f"input parquet is missing columns: {sorted(missing_columns)}")

        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        totals = {
            "rows": 0,
            "messages": 0,
            "reasoning_turns": 0,
            "tool_call_turns": 0,
            "trimmed_tail_messages": 0,
            "skipped_empty_target": 0,
        }
        writer: pq.ParquetWriter | None = None
        try:
            writer = pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression="zstd")
            for batch in parquet_file.iter_batches(batch_size=256, columns=["messages", "tools"]):
                converted_rows = []
                for row in batch.to_pylist():
                    row_index = totals["rows"] + totals["skipped_empty_target"]
                    conversations, row_stats = _convert_messages(row["messages"], row_index)
                    if conversations is None:
                        totals["skipped_empty_target"] += 1
                        continue
                    converted_rows.append(
                        {"conversations": conversations, "tools": _normalize_tools(row.get("tools"))}
                    )
                    totals["rows"] += 1
                    totals["messages"] += len(conversations)
                    for key, value in row_stats.items():
                        totals[key] += value
                if converted_rows:
                    writer.write_table(pa.Table.from_pylist(converted_rows, schema=OUTPUT_SCHEMA))
            writer.close()
            writer = None
            os.replace(temporary, output_path)
        finally:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)

        metadata = {"fingerprint": expected_fingerprint, "output": str(output_path), **totals}
        _write_dataset_info(output_dir, output_path.name)
        _write_json_atomic(metadata_path, metadata)
        print(
            "Prepared {rows} rows ({reasoning_turns} reasoning turns, "
            "{tool_call_turns} tool-call turns, {trimmed_tail_messages} trailing messages trimmed, "
            "{skipped_empty_target} rows skipped for empty assistant target): {output}".format(
                **metadata
            ),
            flush=True,
        )
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source OpenAI-format parquet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Shared prepared-data cache directory")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the cache fingerprint matches")
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.force)


if __name__ == "__main__":
    main()
