"""Check that every Bedrock Mantle model works with default chat model settings.

New Mantle models have repeatedly broken default usage of `ChatOpenAIMantle` and
`ChatAnthropicMantle`: a model served on a different path, a parameter it
rejects, tool calling that needs another API, or a profile missing data. This
lists each region's catalog and runs every available model through the code
paths those fixes touched.

It also probes the data the chat models rely on: which base path serves each
`ChatOpenAIMantle` model, and which `reasoning_effort` levels the `/openai/v1`
models accept. With `--write`, it updates `_mantle_models.py` and the profile
overrides to match, and writes a changelog for the pull request. Models missing
from `_mantle_models.py` are new.

Usage (from `libs/aws`, with AWS credentials):
    uv run python ../../.github/scripts/check_mantle_support.py us-east-1 us-west-2
    uv run python ../../.github/scripts/check_mantle_support.py --write \
        --changelog changelog.md us-east-1 us-west-2

Exits non-zero if a model with a profile fails a check. Profile gaps, and
failures of models without a profile, are reported as warnings. Writes a
Markdown report to `$GITHUB_STEP_SUMMARY` when set.
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import tomllib
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_aws import ChatAnthropicMantle, ChatOpenAIMantle
from langchain_aws.data._mantle_models import MANTLE_MODEL_PATHS

DATA_DIR = Path(__file__).resolve().parents[2] / "libs/aws/langchain_aws/data"
MODELS_FILE = DATA_DIR / "_mantle_models.py"
OVERRIDES_FILE = DATA_DIR / "profile_augmentations.toml"
ROUTE_ERROR = "isn't supported on this route"
TRANSIENT = ("Timeout", "Connection", "RateLimit", "InternalServer", "Overloaded")
TIMEOUT = 60

PROMPT = "What is 2 + 2? Reply with just the number."

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Answer(BaseModel):
    """The answer to the question."""
    value: int


def is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return any(name in type(error).__name__ for name in TRANSIENT) or (
        isinstance(status, int) and (status == 429 or status >= 500)
    )


def run_check(check: Callable[[str, str], Any], model: str, region: str) -> Any:
    try:
        return check(model, region)
    except Exception as e:  # noqa: BLE001
        if not is_transient(e):
            raise
    return check(model, region)


def chat_model(model: str, region: str, **kwargs: Any) -> BaseChatModel:
    if model.startswith("anthropic."):
        kwargs.setdefault("max_tokens", 1024)
        return ChatAnthropicMantle(
            model=model,
            region_name=region,
            max_retries=1,
            default_request_timeout=TIMEOUT,
            **kwargs,
        )
    return ChatOpenAIMantle(
        model=model, region_name=region, max_retries=1, timeout=TIMEOUT, **kwargs
    )


#: Each check covers a default code path a new model has broken before.
CHECKS: dict[str, Callable[[str, str], Any]] = {
    "invoke": lambda m, r: chat_model(m, r).invoke(PROMPT),
    "stream": lambda m, r: list(chat_model(m, r).stream(PROMPT)),
    "max_tokens": lambda m, r: chat_model(m, r, max_tokens=16).invoke(PROMPT),
    "temperature": lambda m, r: chat_model(m, r, temperature=0.5).invoke(PROMPT),
    "tools": lambda m, r: (
        chat_model(m, r).bind_tools([add]).invoke("Use the add tool to add 2 and 3.")
    ),
    "structured_output": lambda m, r: (
        chat_model(m, r).with_structured_output(Answer).invoke(PROMPT)
    ),
}


def catalog(region: str) -> list[str]:
    """Return the IDs of the models available in a region's Mantle catalog."""
    llm = ChatOpenAIMantle(model="openai.gpt-oss-20b", region_name=region)
    return sorted(
        model.id
        for model in llm.root_client.models.list()
        if getattr(model, "status", "available") == "available"
    )


def probe_route(model: str, region: str) -> str | None:
    """Return the base path that serves a `ChatOpenAIMantle` model.

    Returns:
        `"/v1"` or `"/openai/v1"`, or `None` if neither answered conclusively.
    """
    base = f"https://bedrock-mantle.{region}.api.aws"
    try:
        chat_model(model, region, base_url=f"{base}/v1").invoke(PROMPT)
    except Exception as e:  # noqa: BLE001
        if ROUTE_ERROR not in str(e):
            return None
    else:
        return "/v1"
    try:
        chat_model(model, region, base_url=f"{base}/openai/v1").invoke(PROMPT)
    except Exception:  # noqa: BLE001
        return None
    return "/openai/v1"


def accepted_effort_levels(model: str, region: str) -> list[str] | None:
    """Return the `reasoning_effort` values a `/openai/v1` model accepts.

    An invalid value lists the API's candidate values, and each is then sent on
    both Chat Completions and Responses; a level counts only if both accept it.
    `/v1` models aren't probed: Mantle rejects unknown values there before the
    model sees them.
    """
    try:
        chat_model(model, region, reasoning_effort="invalid").invoke(PROMPT)
    except Exception as e:  # noqa: BLE001
        match = re.search(r"Supported values are: ([^.]*)\.", str(e))
        if not match:
            return None
        candidates = re.findall(r"'([^']+)'", match.group(1))
    else:
        return None
    accepted = []
    for level in candidates:
        try:
            for responses in (False, True):
                chat_model(
                    model, region, reasoning_effort=level, use_responses_api=responses
                ).invoke(PROMPT)
        except Exception:  # noqa: BLE001, S112
            continue
        accepted.append(level)
    return accepted


def check_model(job: tuple[str, str]) -> dict[str, Any]:
    """Run every check for one model in one region."""
    region, model = job
    start = time.monotonic()
    failures: dict[str, str] = {}
    gaps = []
    for name, check in CHECKS.items():
        try:
            run_check(check, model, region)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {str(e)[:160]}".replace("\n", " ")
            if is_transient(e):
                gaps.append(f"{name} hit a transient error twice: {error}")
            else:
                failures[name] = error

    profile = dict(chat_model(model, region).profile or {})
    if not profile.get("max_input_tokens"):
        # Without a profile, a failure may be a model limitation, not a bug.
        gaps.append("no model profile")
        gaps += [f"{name} fails: {error}" for name, error in failures.items()]
        failures = {}

    route = levels = None
    if not model.startswith("anthropic."):
        route = probe_route(model, region)
        if route == "/openai/v1" and profile:
            levels = accepted_effort_levels(model, region)

    elapsed = time.monotonic() - start
    status = "FAIL" if failures else "ok"
    print(f"{status:4} {elapsed:5.1f}s {region} {model}", file=sys.stderr)  # noqa: T201
    return {
        "region": region,
        "model": model,
        "failures": failures,
        "gaps": gaps,
        "route": route,
        "levels": levels,
        "profile_levels": profile.get("reasoning_effort_levels"),
    }


def by_model(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Group a result field by model, merging regions that report the same."""
    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        if not result[key]:
            continue
        entry = grouped.setdefault(result["model"], {"regions": [], key: result[key]})
        entry["regions"].append(result["region"])
    return grouped


def path_changes(
    results: list[dict[str, Any]], seen: set[str]
) -> tuple[dict[str, str], dict[str, str], set[str], set[str]]:
    """Compare the probed base paths with `_mantle_models.py`.

    A model whose probes didn't settle on one path keeps its current entry, so
    a transient failure never changes or drops it.

    Args:
        results: Results for the models that were checked.
        seen: Every model in the catalogs, checked or not.

    Returns:
        `(paths, changes, new, removed)`: the updated model paths, a change
        description per model, and the models added to and removed from the
        catalog.
    """
    probed: dict[str, set[str]] = {}
    for result in results:
        model = result["model"]
        path = "/anthropic" if model.startswith("anthropic.") else result["route"]
        if path:
            probed.setdefault(model, set()).add(path)
    paths = dict(MANTLE_MODEL_PATHS)
    changes = {}
    for model in sorted(probed):
        found = probed[model]
        if len(found) != 1:
            continue
        path = next(iter(found))
        old = paths.get(model)
        if old == path:
            continue
        paths[model] = path
        change = f"`{path}`" if old is None else f"`{old}` → `{path}`"
        changes[model] = f"base path {change} (`_mantle_models.py`)"
    new = seen - set(MANTLE_MODEL_PATHS)
    removed = set(MANTLE_MODEL_PATHS) - seen
    for model in removed:
        del paths[model]
    return paths, changes, new, removed


def effort_changes(
    results: list[dict[str, Any]],
) -> tuple[dict[str, list], dict[str, str], list[str]]:
    """Return (verified working) effort level overrides for profiles that have none."""
    overrides = tomllib.loads(OVERRIDES_FILE.read_text()).get("overrides", {})
    accepted: dict[str, set[str]] = {}
    declared: dict[str, list[str] | None] = {}
    order: dict[str, list[str]] = {}
    for result in results:
        if result["levels"] is None:
            continue
        model = result["model"]
        levels = set(result["levels"])
        accepted[model] = accepted.get(model, levels) & levels
        order[model] = result["levels"]
        declared[model] = result["profile_levels"]
    updates: dict[str, list] = {}
    changes = {}
    mismatches = []
    for model, levels in sorted(accepted.items()):
        new = [level for level in order[model] if level in levels]
        current = overrides.get(model, {}).get("reasoning_effort_levels")
        old = declared[model]
        if not new or sorted(new) == sorted(old or []) or new == current:
            continue
        if old:
            mismatches.append(f"`{model}`: profile lists {old}, model accepts {new}")
            continue
        updates[model] = new
        changes[model] = (
            f"reasoning_effort_levels unset → {new} (`profile_augmentations.toml`)"
        )
    return updates, changes, mismatches


def write_paths(paths: dict[str, str]) -> None:
    """Update/add model-specific endpoint path prefixes to `_mantle_models.py`."""
    text = MODELS_FILE.read_text()
    head = text[: text.index("MANTLE_MODEL_PATHS")]
    entries = "".join(
        f'    "{model}": "{path}",\n' for model, path in sorted(paths.items())
    )
    MODELS_FILE.write_text(
        f"{head}MANTLE_MODEL_PATHS: dict[str, str] = {{\n{entries}}}\n"
    )


def write_overrides(updates: dict[str, list]) -> None:
    """Write any required model profile overrides to `profile_augmentations.toml`."""
    lines = OVERRIDES_FILE.read_text().splitlines()
    for model, levels in updates.items():
        value = f"reasoning_effort_levels = {json.dumps(levels)}"
        header = f'[overrides."{model}"]'
        if header not in lines:
            lines += ["", header, value]
            continue
        start = lines.index(header) + 1
        end = next(
            (i for i in range(start, len(lines)) if lines[i].startswith("[")),
            len(lines),
        )
        for i in range(start, end):
            if lines[i].startswith("reasoning_effort_levels ="):
                lines[i] = value
                break
        else:
            lines.insert(start, value)
    OVERRIDES_FILE.write_text("\n".join(lines) + "\n")


def changelog(
    results: list[dict[str, Any]],
    data_changes: list[dict[str, str]],
    new: set[str],
    removed: set[str],
) -> str:
    """Generate the description for the autocut updates PR (if one is needed)"""
    entries: dict[str, list[str]] = {}
    for changes in data_changes:
        for model, change in changes.items():
            entries.setdefault(model, []).append(change)
    for model in new:
        notes = entries.setdefault(model, [])
        if not notes:
            notes.append("base path not determined; retried on the next run")
        failing = sorted(
            {name for r in results if r["model"] == model for name in r["failures"]}
        )
        if failing:
            checks = ", ".join(f"`{name}`" for name in failing)
            entries[model].append(f"fails {checks} (needs a code fix; see the run log)")
    for model in removed:
        entries.setdefault(model, []).append("no longer in the catalog")

    lines = [f"**{len(entries)} models changed**", ""]
    for model in sorted(entries):
        tag = (
            "**[NEW]** "
            if model in new
            else "**[REMOVED]** "
            if model in removed
            else ""
        )
        lines.append(f"- {tag}`{model}`")
        lines += [f"  - {change}" for change in entries[model]]
    return "\n".join(lines) + "\n"


def report(results: list[dict[str, Any]], regions: list[str]) -> tuple[str, bool]:
    """Generate the model checks summary to be displayed in the GitHub action logs."""
    failed = by_model(results, "failures")
    warned = by_model(results, "gaps")
    lines = [
        f"Checked {len(results)} models in {', '.join(regions)} ({', '.join(CHECKS)}).",
        "",
    ]
    if failed:
        lines += ["**Failing:**", ""]
        for model, entry in sorted(failed.items()):
            lines.append(f"- `{model}` ({', '.join(entry['regions'])})")
            lines += [f"  - {name}: {err}" for name, err in entry["failures"].items()]
        lines.append("")
    if warned:
        lines += ["**Warnings** (not failing the check):", ""]
        for model, entry in sorted(warned.items()):
            lines.append(f"- `{model}` ({', '.join(entry['regions'])})")
            lines += [f"  - {gap}" for gap in entry["gaps"]]
        lines.append("")
    if not failed:
        lines.append("Every model with a profile passes every check.")
    return "\n".join(lines), bool(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("regions", nargs="*", default=["us-east-1"])
    parser.add_argument("--write", action="store_true", help="update the data files")
    parser.add_argument("--changelog", type=Path, help="write the changes here")
    parser.add_argument(
        "--full",
        action="store_true",
        help="check every model, not only those missing from `_mantle_models.py`",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    catalogs = {region: catalog(region) for region in args.regions}
    seen = {model for models in catalogs.values() for model in models}
    jobs = [
        (region, model)
        for region, models in catalogs.items()
        for model in models
        if args.full or model not in MANTLE_MODEL_PATHS
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check_model, jobs))

    text, failed = report(results, args.regions)
    if not args.full:
        text = (
            f"{len(seen)} models in the catalogs; checked the {len(jobs)} not yet "
            "in `_mantle_models.py`. Run with `--full` to check every model.\n\n" + text
        )
    paths, path_log, new, removed = path_changes(results, seen)
    updates, effort_log, mismatches = effort_changes(results)
    summary = ""
    if path_log or effort_log or new or removed:
        summary = changelog(results, [path_log, effort_log], new, removed)
        text += "\n\n**Data changes:**\n\n" + summary
    if mismatches:
        text += (
            "\n\n**Effort levels to review** (accepted levels differ from the "
            "profile; not changed automatically):\n\n"
            + "\n".join(f"- {m}" for m in mismatches)
        )

    print(text)  # noqa: T201
    if step_summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as f:
            f.write(text + "\n")
    if args.write:
        if path_log or removed:
            write_paths(paths)
        if updates:
            write_overrides(updates)
    if args.changelog and summary:
        args.changelog.write_text(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
