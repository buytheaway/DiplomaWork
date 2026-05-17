from __future__ import annotations


def pipeline_template_counts(database_stats: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    groups = database_stats.get("embeddings_by_pipeline_model", [])
    if not isinstance(groups, list):
        return counts
    for item in groups:
        if not isinstance(item, dict):
            continue
        pipeline = str(item.get("pipeline") or "").strip()
        if not pipeline:
            continue
        counts[pipeline] = counts.get(pipeline, 0) + int(item.get("count", 0) or 0)
    return counts


def pipeline_index_counts(stats: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pipeline, payload in stats.items():
        if not isinstance(payload, dict):
            continue
        counts[str(pipeline)] = int(payload.get("embeddings_count", 0) or 0)
    return counts


def selected_pipeline(health: dict, counts: dict[str, int]) -> str:
    default_pipeline = str(health.get("default_pipeline") or "").strip()
    if default_pipeline:
        return default_pipeline
    available = health.get("available_pipelines", [])
    if isinstance(available, list) and available:
        return str(available[0])
    return next(iter(counts), "-")


def count_for_pipeline(counts: dict[str, int], pipeline: str) -> int:
    return int(counts.get(pipeline, 0) or 0)


def format_pipeline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return " | ".join(f"{pipeline}: {count:,}" for pipeline, count in sorted(counts.items()))
