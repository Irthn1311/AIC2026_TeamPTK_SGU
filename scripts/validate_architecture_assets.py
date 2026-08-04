"""Validate TRIAGE-EG v1.1 source and every generated architecture asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from generate_architecture_assets import (
    EXPECTED_PAGE_COUNT,
    PROCESSING_TYPES,
    _build_quality_report,
    _drawio_label,
    _drawio_tooltip,
    _render_mermaid,
    analyze_page_layout,
    load_spec,
    validate_spec,
)

LOGGER = logging.getLogger("triage_eg.architecture.validator")
TOOLTIP_SECTIONS = (
    "Responsibility:",
    "Non-responsibility:",
    "Implementation:",
    "Artifacts:",
    "Metrics:",
    "Failure modes:",
    "Fallback:",
    "Owner:",
)


def validate_assets(spec_path: str | Path, drawio_path: str | Path) -> dict[str, Any]:
    """Validate YAML, Mermaid, draw.io, summary and quality report consistency."""
    spec_file = Path(spec_path).resolve()
    drawio_file = Path(drawio_path).resolve()
    spec = load_spec(spec_file)
    validate_spec(spec)
    # A generated bundle is rooted beside the draw.io file. This matters for
    # validation of temporary/release bundles: reading Mermaid or summaries
    # beside the source spec would silently validate the wrong asset set.
    architecture_root = drawio_file.parent

    mermaid_dir = architecture_root / "mermaid"
    generated_mermaid = sorted(mermaid_dir.glob("*.mmd"))
    if len(generated_mermaid) != EXPECTED_PAGE_COUNT:
        raise ValueError(f"Mermaid directory must contain exactly {EXPECTED_PAGE_COUNT} files")
    for page in spec["pages"]:
        mermaid_path = mermaid_dir / page["mermaid_file"]
        if not mermaid_path.is_file():
            raise ValueError(f"Missing Mermaid file for {page['id']}: {mermaid_path}")
        content = mermaid_path.read_text(encoding="utf-8")
        if content != _render_mermaid(page, spec):
            raise ValueError(f"Mermaid asset is stale or manually drifted: {mermaid_path}")
        if f"%% {page['title']}" not in content or "flowchart " not in content:
            raise ValueError(f"Mermaid title or flowchart declaration missing: {mermaid_path}")
        for node in page["nodes"]:
            if not re.search(rf"(?m)^\s*{re.escape(node['id'])}[\[(\{{]", content):
                raise ValueError(f"Mermaid {mermaid_path} is missing node {node['id']}")
        for edge in page["edges"]:
            if (
                edge["source"] not in content
                or edge["target"] not in content
                or edge["label"] not in content
            ):
                raise ValueError(
                    f"Mermaid {mermaid_path} is missing edge semantics for {edge['id']}"
                )
    event_graph_mermaid = (mermaid_dir / "04_event_graph_internals.mmd").read_text(encoding="utf-8")
    for token in (
        "Q1",
        "Q2",
        "Q3",
        "E7",
        "E9",
        "E12",
        "PARTICIPATES_IN",
        "POSSIBLE_SAME_ENTITY",
        "SUPPORTS",
        "MATCH",
    ):
        if token not in event_graph_mermaid:
            raise ValueError(f"PAGE_04 Mermaid is missing actual graph token: {token}")

    if not drawio_file.is_file():
        raise ValueError(f"draw.io file does not exist: {drawio_file}")
    root = ET.parse(drawio_file).getroot()
    if root.tag != "mxfile" or root.get("compressed") != "false":
        raise ValueError("draw.io must be an uncompressed mxfile")
    diagrams = root.findall("diagram")
    if len(diagrams) != EXPECTED_PAGE_COUNT:
        raise ValueError(f"draw.io must contain exactly {EXPECTED_PAGE_COUNT} diagrams")
    expected_titles = [page["title"] for page in spec["pages"]]
    if [diagram.get("name") for diagram in diagrams] != expected_titles:
        raise ValueError("draw.io page titles or order do not match architecture-spec.yaml")

    pages_by_id = {page["id"]: page for page in spec["pages"]}
    for diagram in diagrams:
        page_id = diagram.get("id")
        if page_id not in pages_by_id:
            raise ValueError(f"draw.io contains unexpected page ID: {page_id}")
        page = pages_by_id[page_id]
        model = diagram.find("mxGraphModel")
        if model is None:
            raise ValueError(f"draw.io page {page_id} is missing mxGraphModel")
        layout = analyze_page_layout(page)
        if (
            int(model.get("pageWidth", "0")) != layout["page_width"]
            or int(model.get("pageHeight", "0")) != layout["page_height"]
        ):
            raise ValueError(f"draw.io page dimensions do not match spec layout on {page_id}")
        cells = diagram.findall("./mxGraphModel/root/mxCell")
        cell_ids = {cell.get("id") for cell in cells}
        title_cell = next((cell for cell in cells if cell.get("id") == f"{page_id}_TITLE"), None)
        if title_cell is None or "fontSize=24" not in title_cell.get("style", ""):
            raise ValueError(f"draw.io page {page_id} is missing its 24px title")
        for node in page["nodes"]:
            expected_id = f"{page_id}_{node['id']}"
            cell = next((item for item in cells if item.get("id") == expected_id), None)
            if cell is None:
                raise ValueError(f"draw.io page {page_id} is missing node {node['id']}")
            if cell.get("value") != _drawio_label(page, node, spec):
                raise ValueError(f"draw.io node {expected_id} label is stale or drifted")
            tooltip = cell.get("tooltip", "")
            if tooltip != _drawio_tooltip(node):
                raise ValueError(f"draw.io node {expected_id} tooltip is stale or drifted")
            if any(section not in tooltip for section in TOOLTIP_SECTIONS):
                raise ValueError(f"draw.io node {expected_id} has an incomplete tooltip")
            for size in re.findall(r"fontSize=(\d+)", cell.get("style", "")):
                if int(size) < 10:
                    raise ValueError(f"draw.io node {expected_id} uses a font below 10px")
        for edge in page["edges"]:
            expected_id = f"{page_id}_{edge['id']}"
            matching = [cell for cell in cells if cell.get("id") == expected_id]
            if len(matching) != 1:
                raise ValueError(f"draw.io page {page_id} is missing edge {edge['id']}")
            cell = matching[0]
            if cell.get("source") not in cell_ids or cell.get("target") not in cell_ids:
                raise ValueError(f"draw.io edge {expected_id} has an orphan endpoint")
            if (
                cell.get("source") != f"{page_id}_{edge['source']}"
                or cell.get("target") != f"{page_id}_{edge['target']}"
            ):
                raise ValueError(f"draw.io edge {expected_id} endpoints drifted from the spec")
            edge_prefix = {
                "control": "CONTROL: ",
                "optional": "OPTIONAL: ",
                "fallback": "FALLBACK: ",
                "evidence": "EVIDENCE: ",
                "artifact": "ARTIFACT: ",
                "match": "MATCH: ",
                "data": "",
            }[edge["flow_type"]]
            if cell.get("value") != edge_prefix + edge["label"]:
                raise ValueError(f"draw.io edge {expected_id} label drifted from the spec")
            if edge["flow_type"] == "match" and "strokeWidth=3" not in cell.get("style", ""):
                raise ValueError(f"draw.io MATCH edge {expected_id} lacks match styling")

    summary_path = architecture_root / "generated" / "architecture_summary.json"
    report_path = architecture_root / "generated" / "architecture_quality_report.md"
    if not summary_path.is_file() or not report_path.is_file():
        raise ValueError("Architecture summary or quality report is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_summary = {
        "page_count",
        "module_count",
        "node_count",
        "edge_count",
        "pages",
        "criticality_statistics",
        "maturity_statistics",
        "source_basis_statistics",
        "ownership_statistics",
        "models_confirmed",
        "models_unselected",
        "contract_quality",
        "spec_sha256",
        "warnings",
        "validation_status",
    }
    if required_summary - set(summary):
        raise ValueError("Architecture summary is missing v1.1 quality fields")
    expected_nodes = sum(len(page["nodes"]) for page in spec["pages"])
    expected_edges = sum(len(page["edges"]) for page in spec["pages"])
    expected_spec_hash = hashlib.sha256(spec_file.read_bytes()).hexdigest()
    if (
        summary["page_count"] != EXPECTED_PAGE_COUNT
        or summary["node_count"] != expected_nodes
        or summary["edge_count"] != expected_edges
        or summary["spec_sha256"] != expected_spec_hash
    ):
        raise ValueError("Architecture summary counts do not match the YAML spec")
    expected_pages = [
        {
            "id": page["id"],
            "title": page["title"],
            "mermaid_file": page["mermaid_file"],
            "module_count": sum(node["type"] in PROCESSING_TYPES for node in page["nodes"]),
            "node_count": len(page["nodes"]),
            "edge_count": len(page["edges"]),
            **analyze_page_layout(page),
        }
        for page in spec["pages"]
    ]
    if summary["pages"] != expected_pages:
        raise ValueError("Architecture summary page metrics drifted from the YAML spec")
    all_nodes = [node for page in spec["pages"] for node in page["nodes"]]
    expected_statistics = {
        "criticality_statistics": dict(
            sorted(Counter(node["criticality"] for node in all_nodes).items())
        ),
        "maturity_statistics": dict(
            sorted(Counter(node["maturity"] for node in all_nodes).items())
        ),
        "source_basis_statistics": dict(
            sorted(Counter(node["source_basis"] for node in all_nodes).items())
        ),
    }
    for field, expected in expected_statistics.items():
        if summary[field] != expected:
            raise ValueError(f"Architecture summary {field} drifted from the YAML spec")
    if summary["contract_quality"] != {
        "specific_node_contracts": expected_nodes,
        "placeholder_node_contracts": 0,
        "edge_aligned_interfaces": expected_nodes,
    }:
        raise ValueError("Architecture summary contract-quality counts are invalid")
    if summary["validation_status"] != "PASS":
        raise ValueError("Architecture summary validation_status is not PASS")
    report = report_path.read_text(encoding="utf-8")
    for heading in (
        "Page dimensions",
        "Status migration",
        "Contract quality",
        "Event Graph checks",
        "Task-criticality checks",
        "Ownership distribution",
        "Remaining open questions",
    ):
        if heading not in report:
            raise ValueError(f"Architecture quality report is missing section: {heading}")
    if report != _build_quality_report(summary):
        raise ValueError("Architecture quality report is stale or manually drifted")
    return {
        "pages": EXPECTED_PAGE_COUNT,
        "nodes": expected_nodes,
        "edges": expected_edges,
        "warnings": summary["warnings"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--drawio", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        counts = validate_assets(args.spec, args.drawio)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        ET.ParseError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        LOGGER.error("%s", error)
        return 2
    LOGGER.info(
        "Architecture validation passed: %s pages, %s nodes, %s edges, %s warnings",
        counts["pages"],
        counts["nodes"],
        counts["edges"],
        len(counts["warnings"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
