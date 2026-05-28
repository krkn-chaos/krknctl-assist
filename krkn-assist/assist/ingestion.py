from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from .settings import EXCLUDED_SCENARIO_IDS, MERGED_DOCS_DEBUG_DIR, NON_SCENARIO_DOCS

logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(
    r"<krkn-hub-scenario\s+id=[\"']([^\"']+)[\"']>(.*?)</krkn-hub-scenario>",
    re.IGNORECASE | re.DOTALL,
)
_SCENARIO_ID_PATTERN = re.compile(
    r"<krkn-hub-scenario\s+id=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_HOW_TO_RUN_PATTERN = re.compile(r"^##\s+How\s+to\s+Run\b.*$", re.IGNORECASE | re.MULTILINE)
_KRKNCTL_RUN_PATTERN = re.compile(
    r"krknctl\s+run\s+([a-z0-9][a-z0-9-]*)",
    re.IGNORECASE,
)
_NOT_SUPPORTED_PATTERN = re.compile(
    r"not yet supported|not currently available via krknctl",
    re.IGNORECASE,
)
_SCENARIO_ID_ALIASES = {
    "power-outage-scenarios": "power-outages",
    "pvc-scenario": "pvc-scenarios",
    "storage-throttle-scenario": "storage-throttle",
}


def _canonical_scenario_id(scenario_id: str) -> str:
    scenario_id = (scenario_id or "").strip()
    return _SCENARIO_ID_ALIASES.get(scenario_id, scenario_id)


def load_local_scenario_docs(docs_dir: str) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.warning("Docs dir not found: %s", docs_dir)
        return docs

    for md_file in sorted(docs_path.glob("*.md")):
        if md_file.name in NON_SCENARIO_DOCS:
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        docs.append((md_file.stem, content))

    return docs


def _extract_tagged_scenario(content: str) -> tuple[str | None, str | None]:
    match = _TAG_PATTERN.search(content or "")
    if not match:
        return None, None
    scenario_id = match.group(1).strip()
    scenario_content = match.group(2).strip()
    if not scenario_id or not scenario_content:
        return None, None
    return scenario_id, scenario_content


def _extract_scenario_id(content: str) -> str | None:
    match = _SCENARIO_ID_PATTERN.search(content or "")
    if not match:
        return None
    scenario_id = match.group(1).strip()
    return scenario_id or None


def _strip_how_to_run(content: str) -> str:
    match = _HOW_TO_RUN_PATTERN.search(content or "")
    if not match:
        return (content or "").strip()
    return (content or "")[: match.start()].strip()


def _strip_scenario_tag_markers(content: str) -> str:
    if not content:
        return ""
    return re.sub(r"</?krkn-hub-scenario[^>]*>", "", content, flags=re.IGNORECASE).strip()


def _extract_krknctl_run_scenario_id(content: str) -> str | None:
    match = _KRKNCTL_RUN_PATTERN.search(content or "")
    if not match:
        return None
    scenario_id = match.group(1).strip()
    return scenario_id or None


def _krknctl_tab_supported(content: str) -> bool:
    if not content:
        return False
    return _NOT_SUPPORTED_PATTERN.search(content) is None


def _clone_repository(repo_url: str, dest: str, branch: str | None = None) -> None:
    clone_cmd = ["git", "clone", "--depth", "1", "--quiet"]
    if branch:
        clone_cmd.extend(["--branch", branch])
        logger.info("Cloning %s (branch: %s)", repo_url, branch)
    else:
        logger.info("Cloning %s (default branch)", repo_url)
    clone_cmd.extend([repo_url, dest])
    subprocess.run(clone_cmd, check=True, capture_output=True, text=True)


def _append_doc(doc_map: Dict[str, List[str]], scenario_id: str, content: str) -> None:
    scenario_id = _canonical_scenario_id(scenario_id)
    if not scenario_id or not content:
        return
    if scenario_id in EXCLUDED_SCENARIO_IDS:
        logger.debug("Skipping excluded scenario id: %s", scenario_id)
        return
    normalized = re.sub(r"\s+", " ", content).strip().lower()
    if not normalized:
        return
    parts = doc_map.setdefault(scenario_id, [])
    for existing in parts:
        existing_normalized = re.sub(r"\s+", " ", existing).strip().lower()
        if not existing_normalized:
            continue
        if normalized == existing_normalized:
            return
        if normalized in existing_normalized or existing_normalized in normalized:
            return
    parts.append(content.strip())


def _merge_docs(doc_map: Dict[str, List[str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for scenario_id in sorted(doc_map.keys()):
        parts = [part for part in doc_map[scenario_id] if part]
        if not parts:
            continue
        merged.append((scenario_id, "\n\n".join(parts)))
    return merged


def persist_merged_scenario_docs(
    docs: list[tuple[str, str]],
    output_dir: str = MERGED_DOCS_DEBUG_DIR,
) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for stale_path in target_dir.glob("*.md"):
        try:
            stale_path.unlink()
        except OSError as exc:
            logger.warning("Failed to remove stale merged doc %s: %s", stale_path, exc)

    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_path.unlink()
        except OSError as exc:
            logger.warning("Failed to remove stale manifest %s: %s", manifest_path, exc)

    manifest: list[dict[str, str | int]] = []
    for scenario_id, content in docs:
        scenario_path = target_dir / f"{scenario_id}.md"
        try:
            scenario_path.write_text(content.rstrip() + "\n", encoding="utf-8")
            manifest.append(
                {
                    "id": scenario_id,
                    "path": str(scenario_path),
                    "chars": len(content),
                }
            )
        except OSError as exc:
            logger.warning("Failed to persist merged doc %s: %s", scenario_path, exc)

    try:
        manifest_path.write_text(
            json.dumps({"docs": manifest}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to write merged docs manifest %s: %s", manifest_path, exc)


def _collect_scenario_folder_docs(scenarios_root: str) -> dict[str, str]:
    scenario_parts: dict[str, list[str]] = {}
    root_path = Path(scenarios_root)
    if not root_path.exists():
        return {}

    for root, _, files in os.walk(root_path):
        if "_index.md" not in files or "_tab-krknctl.md" not in files:
            continue
        index_path = Path(root) / "_index.md"
        tab_path = Path(root) / "_tab-krknctl.md"
        try:
            index_content = index_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", index_path, exc)
            continue
        try:
            tab_content = tab_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to read %s: %s", tab_path, exc)
            tab_content = ""

        if not _krknctl_tab_supported(tab_content):
            logger.info(
                "Skipping %s because the krknctl tab marks it as unsupported",
                index_path,
            )
            continue

        scenario_id = _extract_scenario_id(index_content)
        if not scenario_id:
            scenario_id = Path(root).name
            runnable_scenario_id = _extract_krknctl_run_scenario_id(tab_content)
            if runnable_scenario_id:
                logger.info(
                    "Using folder name %s as source scenario id; runnable krknctl scenario is %s from %s",
                    scenario_id,
                    runnable_scenario_id,
                    tab_path,
                )
            else:
                logger.warning(
                    "Missing krkn-hub-scenario id in %s; using folder name %s",
                    index_path,
                    scenario_id,
                )

        index_content = _strip_scenario_tag_markers(_strip_how_to_run(index_content))
        parts = [part.strip() for part in [index_content, tab_content] if part and part.strip()]
        if not parts:
            continue
        scenario_parts.setdefault(scenario_id, []).append("\n\n".join(parts))

    return {
        scenario_id: "\n\n".join(parts)
        for scenario_id, parts in scenario_parts.items()
        if parts
    }


def _collect_tagged_docs_legacy(docs_root: str) -> dict[str, str]:
    tagged_docs: dict[str, str] = {}
    for root, _, files in os.walk(docs_root):
        for file_name in files:
            if not file_name.endswith(".md"):
                continue
            file_path = os.path.join(root, file_name)
            try:
                content = Path(file_path).read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)
                continue
            scenario_id, scenario_content = _extract_tagged_scenario(content)
            if not scenario_id or not scenario_content:
                continue
            tagged_docs[scenario_id] = scenario_content
    return tagged_docs


def _collect_tagged_docs(docs_root: str) -> dict[str, str]:
    docs_root_path = Path(docs_root)
    scenarios_root = docs_root_path / "scenarios"
    if scenarios_root.exists():
        scenario_docs = _collect_scenario_folder_docs(str(scenarios_root))
    else:
        scenario_docs = _collect_scenario_folder_docs(str(docs_root_path))
    if scenario_docs:
        return scenario_docs
    return _collect_tagged_docs_legacy(docs_root)


def _format_krkn_hub_inputs(scenario_name: str, scenario_inputs: list[dict]) -> str:
    content_parts = [
        f"# krknctl Scenario: {scenario_name}",
        f"Command: krknctl run {scenario_name}",
        "",
        "## Parameters",
        "",
    ]

    for param in scenario_inputs:
        name = param.get("name", "")
        description = param.get("description", "")
        short_desc = param.get("short_description", "")
        param_type = param.get("type", "")
        default = param.get("default", "")
        required = param.get("required", "false")
        validator = param.get("validator", "")

        content_parts.extend(
            [
                f"### --{name}",
                f"Type: {param_type}",
                f"Required: {required}",
                f"Description: {description}",
            ]
        )

        if short_desc and short_desc != description:
            content_parts.append(f"Short Description: {short_desc}")
        if default:
            content_parts.append(f"Default: {default}")
        if validator:
            content_parts.append(f"Validation Pattern: {validator}")

        content_parts.extend(["", "---", ""])

    content_parts.extend(["## Usage Example", f"krknctl run {scenario_name}", ""])
    example_params = []
    for param in scenario_inputs:
        name = param.get("name", "")
        default = param.get("default", "")
        if default:
            example_params.append(f"--{name} {default}")

    if example_params:
        content_parts.append(
            f"krknctl run {scenario_name} {' '.join(example_params[:3])}"
        )

    return "\n".join(content_parts)


def _collect_krkn_hub_inputs(repo_url: str, branch: str | None = None) -> dict[str, str]:
    docs: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        _clone_repository(repo_url, temp_dir, branch)
        for root, _, files in os.walk(temp_dir):
            for file_name in files:
                if file_name != "krknctl-input.json":
                    continue
                file_path = os.path.join(root, file_name)
                scenario_name = os.path.basename(root)
                try:
                    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Failed to parse %s: %s", file_path, exc)
                    continue
                if not isinstance(payload, list):
                    logger.warning("Unexpected format in %s", file_path)
                    continue
                docs[scenario_name] = _format_krkn_hub_inputs(scenario_name, payload)
    return docs


def build_scenario_documents(
    *,
    docs_dir: str,
    github_repo: str | None,
    repo_path: str,
    github_branch: str | None,
    krkn_hub_repo: str | None,
    krkn_hub_branch: str | None,
    local_docs_path: str | None,
) -> list[tuple[str, str]]:
    scenario_parts: dict[str, list[str]] = {}

    tagged_docs: dict[str, str] = {}
    if local_docs_path:
        local_root = Path(local_docs_path)
        preferred_root = local_root / repo_path
        if preferred_root.exists():
            docs_root = str(preferred_root)
            logger.info("Loading docs from local repo root: %s", docs_root)
            tagged_docs = _collect_tagged_docs(docs_root)
        elif (local_root / "scenarios").exists():
            docs_root = str(local_root)
            logger.info("Loading docs from local scenarios root: %s", docs_root)
            tagged_docs = _collect_tagged_docs(docs_root)
        elif local_root.exists():
            docs_root = str(local_root)
            logger.info("Loading docs from local docs root: %s", docs_root)
            tagged_docs = _collect_tagged_docs(docs_root)
        else:
            logger.warning("Local docs path not found: %s", local_docs_path)
    elif github_repo:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                _clone_repository(github_repo, temp_dir, github_branch)
                docs_root = str(Path(temp_dir) / repo_path)
                if Path(docs_root).exists():
                    tagged_docs = _collect_tagged_docs(docs_root)
                else:
                    logger.warning("Docs path not found: %s", docs_root)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Failed to clone docs repo: %s", exc)

    if tagged_docs:
        for scenario_id, content in tagged_docs.items():
            _append_doc(scenario_parts, scenario_id, content)
    else:
        logger.info("No tagged docs found; falling back to local docs")
        for scenario_id, content in load_local_scenario_docs(docs_dir):
            _append_doc(scenario_parts, scenario_id, content)

    if krkn_hub_repo:
        try:
            scenario_inputs = _collect_krkn_hub_inputs(krkn_hub_repo, krkn_hub_branch)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Failed to clone krkn-hub repo: %s", exc)
            scenario_inputs = {}
        for scenario_id, content in scenario_inputs.items():
            _append_doc(scenario_parts, scenario_id, content)

    merged_docs = _merge_docs(scenario_parts)
    persist_merged_scenario_docs(merged_docs)
    return merged_docs
