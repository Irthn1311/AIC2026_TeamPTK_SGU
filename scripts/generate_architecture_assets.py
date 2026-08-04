"""Generate deterministic TRIAGE-EG v1.1 architecture assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("triage_eg.architecture.generator")
EXPECTED_PAGE_COUNT = 9
FLOW_TYPES = {"data", "control", "optional", "fallback", "evidence", "artifact", "match"}
CRITICALITIES = {"CORE", "CONDITIONAL", "OPTIONAL", "DEFERRED"}
MATURITIES = {"CURRENT_TEMPLATE", "PLANNED", "BASELINE", "EXPERIMENTAL", "SELECTED", "VALIDATED"}
SOURCE_BASES = {"BTC_CONFIRMED", "TEAM_DECISION", "RESEARCH_CANDIDATE", "SOFTWARE_TEMPLATE"}
NODE_TYPES = {
    "process",
    "store",
    "artifact",
    "decision",
    "control",
    "graph_node",
    "note",
    "offpage",
}
GRAPH_NODE_TYPES = {
    "QueryEvent",
    "Video",
    "SegmentEvent",
    "Entity",
    "SemanticMoment",
    "EvidenceRef",
}
REQUIRED_GRAPH_RELATIONS = {
    "BEFORE",
    "CONTAINS",
    "PARTICIPATES_IN",
    "POSSIBLE_SAME_ENTITY",
    "ANCHORS",
    "SUPPORTS",
    "MATCH",
}
REQUIRED_NODE_FIELDS = {
    "id",
    "number",
    "title",
    "subtitle",
    "layer",
    "type",
    "criticality",
    "criticality_scope",
    "maturity",
    "source_basis",
    "architecture_owner",
    "implementation_owner",
    "reviewers",
    "responsibility",
    "non_responsibility",
    "inputs",
    "processing",
    "implementations",
    "outputs",
    "artifacts",
    "metrics",
    "failure_modes",
    "fallback",
    "dependencies",
    "next_modules",
    "geometry",
}
REQUIRED_GEOMETRY_FIELDS = {"x", "y", "width", "height"}
PROCESSING_TYPES = {"process", "decision", "control"}
CONTRACT_LIST_FIELDS = {
    "responsibility",
    "non_responsibility",
    "inputs",
    "processing",
    "implementations",
    "outputs",
    "artifacts",
    "metrics",
    "failure_modes",
    "dependencies",
    "next_modules",
}
PLACEHOLDER_CONTRACT_PHRASES = {
    "typed upstream contract",
    "typed downstream contract",
    "architecture contract; implementation selected by validation status",
    "deterministic contract fallback.",
}


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load a safe YAML architecture mapping."""
    spec_path = Path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Architecture spec does not exist: {spec_path}")
    with spec_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Architecture spec root must be a mapping")
    return payload


def validate_spec(spec: Mapping[str, Any], expected_pages: int = EXPECTED_PAGE_COUNT) -> None:
    """Validate v1.1 schema, references, taxonomies, graph semantics and layout."""
    required_top = {
        "project",
        "criticalities",
        "maturities",
        "source_bases",
        "owners",
        "styles",
        "models",
        "pages",
    }
    missing_top = required_top - set(spec)
    if missing_top:
        raise ValueError(f"Missing top-level architecture keys: {sorted(missing_top)}")
    if str(spec["project"].get("version")) != "1.1":
        raise ValueError("Architecture project.version must be 1.1")
    if "statuses" in spec:
        raise ValueError("Legacy top-level statuses registry is not allowed in v1.1")
    if set(spec["criticalities"]) != CRITICALITIES:
        raise ValueError("criticalities registry does not match v1.1 taxonomy")
    if set(spec["maturities"]) != MATURITIES:
        raise ValueError("maturities registry does not match v1.1 taxonomy")
    if set(spec["source_bases"]) != SOURCE_BASES:
        raise ValueError("source_bases registry does not match v1.1 taxonomy")
    owners = spec["owners"]
    layers = spec["styles"].get("layers", {})
    fonts = spec["styles"].get("fonts", {})
    if not owners or not layers:
        raise ValueError("owners and styles.layers must be non-empty mappings")
    if min(fonts.values(), default=0) < 10:
        raise ValueError("All configured font sizes must be at least 10")
    if fonts.get("page_title", 0) < 24 or fonts.get("group_title", 0) < 18:
        raise ValueError("Page and group title fonts violate v1.1 minimums")
    pages = spec["pages"]
    if not isinstance(pages, list) or len(pages) != expected_pages:
        raise ValueError(f"Architecture must contain exactly {expected_pages} pages")
    page_ids: set[str] = set()
    mermaid_files: set[str] = set()
    for page in pages:
        _validate_page(page, owners, layers)
        if page["id"] in page_ids:
            raise ValueError(f"Duplicate page ID: {page['id']}")
        page_ids.add(page["id"])
        filename = page.get("mermaid_file")
        if not filename or Path(filename).name != filename or filename in mermaid_files:
            raise ValueError(f"Invalid or duplicate Mermaid filename on {page['id']}")
        mermaid_files.add(filename)
    expected_ids = [f"PAGE_{index:02d}" for index in range(EXPECTED_PAGE_COUNT)]
    if [page["id"] for page in pages] != expected_ids:
        raise ValueError("Architecture page IDs must be ordered PAGE_00 through PAGE_08")
    _validate_event_graph(next(page for page in pages if page["id"] == "PAGE_04"))
    _validate_content(spec)
    _validate_models(spec["models"])


def _validate_page(
    page: Mapping[str, Any], owners: Mapping[str, Any], layers: Mapping[str, Any]
) -> None:
    required = {
        "id",
        "title",
        "description",
        "direction",
        "mermaid_file",
        "detail_level",
        "layout",
        "groups",
        "nodes",
        "edges",
        "notes",
    }
    missing = required - set(page)
    if missing:
        raise ValueError(f"Page is missing fields: {sorted(missing)}")
    page_id = page["id"]
    if page["direction"] not in {"LR", "TD"}:
        raise ValueError(f"Page {page_id} direction must be LR or TD")
    if page["detail_level"] not in {"overview", "compact", "full"}:
        raise ValueError(f"Page {page_id} has invalid detail_level")
    group_ids: set[str] = set()
    for item in page["groups"]:
        if {"id", "title", "layer", "geometry"} - set(item):
            raise ValueError(f"Page {page_id} has an incomplete group")
        if item["id"] in group_ids:
            raise ValueError(f"Page {page_id} has duplicate group {item['id']}")
        group_ids.add(item["id"])
        if item["layer"] not in layers:
            raise ValueError(f"Group {item['id']} has unknown layer {item['layer']}")
        _validate_geometry(item["geometry"], f"group {item['id']}")
    node_ids: set[str] = set()
    for item in page["nodes"]:
        missing_node = REQUIRED_NODE_FIELDS - set(item)
        if missing_node:
            raise ValueError(
                f"Node {item.get('id', '<missing>')} is missing {sorted(missing_node)}"
            )
        node_id = item["id"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(node_id)) or node_id in node_ids:
            raise ValueError(f"Page {page_id} has invalid or duplicate node ID: {node_id}")
        node_ids.add(node_id)
        if "status" in item or "owner" in item:
            raise ValueError(f"Node {node_id} still uses legacy status/owner")
        if item["criticality"] not in CRITICALITIES:
            raise ValueError(f"Node {node_id} has invalid criticality: {item['criticality']}")
        if item["maturity"] not in MATURITIES:
            raise ValueError(f"Node {node_id} has invalid maturity: {item['maturity']}")
        if item["source_basis"] not in SOURCE_BASES:
            raise ValueError(f"Node {node_id} has invalid source_basis: {item['source_basis']}")
        if item["type"] not in NODE_TYPES:
            raise ValueError(f"Node {node_id} has invalid type: {item['type']}")
        if item["layer"] not in layers:
            raise ValueError(f"Node {node_id} has unknown layer: {item['layer']}")
        if item.get("group") and item["group"] not in group_ids:
            raise ValueError(f"Node {node_id} refers to unknown group: {item['group']}")
        for field in ("architecture_owner", "implementation_owner", "reviewers"):
            values = item[field]
            if not isinstance(values, list):
                raise ValueError(f"Node {node_id} {field} must be a list")
            if field != "reviewers" and not values:
                raise ValueError(f"Node {node_id} {field} must not be empty")
            unknown = set(values) - set(owners)
            if unknown:
                raise ValueError(
                    f"Node {node_id} {field} contains unknown owners: {sorted(unknown)}"
                )
        _validate_geometry(item["geometry"], f"node {node_id}")
        if item["type"] == "graph_node" and page_id != "PAGE_04":
            raise ValueError(f"Actual graph node {node_id} is only allowed on PAGE_04")
        if item["type"] == "graph_node" and item.get("graph_node_type") not in GRAPH_NODE_TYPES:
            raise ValueError(f"Graph node {node_id} has invalid graph_node_type")
    edge_ids: set[str] = set()
    for item in page["edges"]:
        missing_edge = {"id", "source", "target", "label", "flow_type"} - set(item)
        if missing_edge:
            raise ValueError(f"Page {page_id} edge is missing {sorted(missing_edge)}")
        if item["id"] in edge_ids:
            raise ValueError(f"Page {page_id} has duplicate edge {item['id']}")
        edge_ids.add(item["id"])
        if item["source"] not in node_ids or item["target"] not in node_ids:
            raise ValueError(f"Page {page_id} edge {item['id']} has orphan endpoint")
        if item["flow_type"] not in FLOW_TYPES:
            raise ValueError(f"Edge {item['id']} has invalid flow_type: {item['flow_type']}")
        for point in item.get("waypoints", []):
            if set(point) != {"x", "y"}:
                raise ValueError(f"Edge {item['id']} waypoint must define x and y")
    _validate_node_contracts(page)
    layout = analyze_page_layout(page)
    if layout["aspect_ratio"] > float(page["layout"].get("max_aspect_ratio", 3.5)):
        raise ValueError(f"Page {page_id} aspect ratio {layout['aspect_ratio']:.3f} exceeds limit")
    if layout["page_width"] > float(page["layout"].get("max_width", 10_000)):
        raise ValueError(f"Page {page_id} width {layout['page_width']} exceeds configured limit")
    if layout["overlapping_node_pairs"]:
        raise ValueError(
            f"Page {page_id} has overlapping nodes: {layout['overlapping_node_pairs']}"
        )
    if page_id == "PAGE_01" and layout["processing_node_count"] > 24:
        raise ValueError("PAGE_01 contains more than 24 processing nodes")
    _validate_shape_usage(page)


def _validate_node_contracts(page: Mapping[str, Any]) -> None:
    """Reject placeholder contracts and require interfaces to mirror page edges."""
    nodes = {item["id"]: item for item in page["nodes"]}
    incoming: dict[str, list[Mapping[str, Any]]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[Mapping[str, Any]]] = {node_id: [] for node_id in nodes}
    for edge in page["edges"]:
        incoming[edge["target"]].append(edge)
        outgoing[edge["source"]].append(edge)

    for node_id, node in nodes.items():
        for field in CONTRACT_LIST_FIELDS:
            value = node[field]
            if not isinstance(value, list):
                raise ValueError(f"Node {node_id} {field} must be a list")
            if field not in {"dependencies", "next_modules"} and not value:
                raise ValueError(f"Node {node_id} {field} must not be empty")
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError(f"Node {node_id} {field} must contain non-empty strings")
        fallback = node["fallback"]
        if not isinstance(fallback, str) or not fallback.strip():
            raise ValueError(f"Node {node_id} fallback must be a non-empty string")
        contract_text = " ".join(
            [
                *(item for field in CONTRACT_LIST_FIELDS for item in node[field]),
                fallback,
            ]
        ).lower()
        placeholders = [
            phrase for phrase in PLACEHOLDER_CONTRACT_PHRASES if phrase in contract_text
        ]
        if placeholders or re.search(r"execute bounded .+ policy", contract_text):
            raise ValueError(f"Node {node_id} still contains placeholder contract text")

        expected_dependencies = {edge["source"] for edge in incoming[node_id]}
        expected_next = {edge["target"] for edge in outgoing[node_id]}
        if set(node["dependencies"]) != expected_dependencies:
            raise ValueError(f"Node {node_id} dependencies do not match incoming edges")
        if set(node["next_modules"]) != expected_next:
            raise ValueError(f"Node {node_id} next_modules do not match outgoing edges")
        input_text = " ".join(node["inputs"]).lower()
        output_text = " ".join(node["outputs"]).lower()
        for edge in incoming[node_id]:
            if str(edge["label"]).lower() not in input_text:
                raise ValueError(
                    f"Node {node_id} inputs omit incoming edge label {edge['label']!r}"
                )
        for edge in outgoing[node_id]:
            if str(edge["label"]).lower() not in output_text:
                raise ValueError(
                    f"Node {node_id} outputs omit outgoing edge label {edge['label']!r}"
                )


def _validate_geometry(geometry: Any, context: str) -> None:
    if not isinstance(geometry, Mapping) or REQUIRED_GEOMETRY_FIELDS - set(geometry):
        raise ValueError(f"{context} must define x, y, width, and height")
    if any(not isinstance(geometry[key], int | float) for key in REQUIRED_GEOMETRY_FIELDS):
        raise ValueError(f"{context} geometry values must be numeric")
    if geometry["width"] <= 0 or geometry["height"] <= 0:
        raise ValueError(f"{context} width and height must be positive")


def analyze_page_layout(page: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic page bounds, overlap pairs and minimum-gap warnings."""
    geometries = [item["geometry"] for item in [*page["groups"], *page["nodes"]]]
    page_width = max(g["x"] + g["width"] for g in geometries) + 20
    page_height = max(g["y"] + g["height"] for g in geometries) + 20
    overlaps: list[list[str]] = []
    gap_warnings: list[str] = []
    minimum_gap = float(page["layout"].get("min_gap", 40))
    nodes = page["nodes"]
    for index, left in enumerate(nodes):
        a = left["geometry"]
        for right in nodes[index + 1 :]:
            b = right["geometry"]
            dx = max(b["x"] - (a["x"] + a["width"]), a["x"] - (b["x"] + b["width"]), 0)
            dy = max(b["y"] - (a["y"] + a["height"]), a["y"] - (b["y"] + b["height"]), 0)
            if dx == 0 and dy == 0:
                overlaps.append([left["id"], right["id"]])
            elif (dx == 0 and 0 < dy < minimum_gap) or (dy == 0 and 0 < dx < minimum_gap):
                gap_warnings.append(f"{left['id']} ↔ {right['id']} gap below {minimum_gap:g}")
    return {
        "page_width": page_width,
        "page_height": page_height,
        "aspect_ratio": round(page_width / page_height, 3),
        "overlapping_node_pairs": overlaps,
        "minimum_gap_warnings": gap_warnings,
        "processing_node_count": sum(item["type"] in PROCESSING_TYPES for item in nodes),
    }


def _validate_shape_usage(page: Mapping[str, Any]) -> None:
    for item in page["nodes"]:
        title = item["title"].lower()
        if item["type"] == "decision" and len(item["title"]) > 32:
            raise ValueError(f"Decision diamond {item['id']} contains a long module title")
        if item["type"] == "decision" and any(
            term in title
            for term in ("frame bank", "evidence verifier", "answer type router", "task policy")
        ):
            raise ValueError(f"Module {item['id']} misuses a decision diamond")
        if item["type"] == "graph_node" and any(
            term in title
            for term in ("query event graph", "candidate graph builder", "graph semantics")
        ):
            raise ValueError(f"Module {item['id']} misuses an ellipse")


def _validate_event_graph(page: Mapping[str, Any]) -> None:
    actual_types = {item.get("graph_node_type") for item in page["nodes"]}
    missing_types = GRAPH_NODE_TYPES - actual_types
    if missing_types:
        raise ValueError(f"PAGE_04 is missing graph node types: {sorted(missing_types)}")
    relations = {str(item["label"]).split()[0] for item in page["edges"]}
    missing_relations = REQUIRED_GRAPH_RELATIONS - relations
    if missing_relations:
        raise ValueError(f"PAGE_04 is missing graph relations: {sorted(missing_relations)}")


def _validate_content(spec: Mapping[str, Any]) -> None:
    required = {
        "Semantic Moment Localizer": ("CORE", "TRAKE"),
        "Q&A Answer Extractor": ("CORE", "Q&A"),
        "Team Visual Encoder": ("CONDITIONAL", "Team Frame Bank enabled"),
        "Bounded Agent Planner": ("OPTIONAL", "Control bus"),
    }
    for title, (criticality, scope) in required.items():
        matches = [
            item for page in spec["pages"] for item in page["nodes"] if item["title"] == title
        ]
        if not matches or not any(
            item["criticality"] == criticality
            and scope.lower() in item["criticality_scope"].lower()
            for item in matches
        ):
            raise ValueError(f"Required task criticality is missing for {title}")
    titles = {item["title"] for page in spec["pages"] for item in page["nodes"]}
    for title in (
        "Application Backend / Orchestrator",
        "Interactive UI / Operator Console",
        "Automatic Mode",
        "Agent Control Bus",
    ):
        if title not in titles:
            raise ValueError(f"Required application module is missing: {title}")


def _validate_models(models: Mapping[str, Any]) -> None:
    required = {
        "interface",
        "current_template",
        "baseline",
        "candidates",
        "selected",
        "validation_status",
        "source_basis",
        "maturity",
    }
    for model_id, model in models.items():
        missing = required - set(model)
        if missing:
            raise ValueError(f"Model {model_id} is missing registry fields: {sorted(missing)}")
        if model["source_basis"] not in SOURCE_BASES or model["maturity"] not in MATURITIES:
            raise ValueError(f"Model {model_id} has invalid taxonomy")
        if model["validation_status"] == "UNSELECTED" and model["selected"] is not None:
            raise ValueError(f"Unselected model {model_id} cannot have selected implementation")
        if (
            model["selected"]
            and model["source_basis"] != "BTC_CONFIRMED"
            and model["maturity"] not in {"SELECTED", "VALIDATED"}
        ):
            raise ValueError(f"Model {model_id} was selected without a decision/validation state")


def generate_assets(spec_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Validate the source and generate Mermaid, draw.io, summary and quality report."""
    spec_file = Path(spec_path).resolve()
    root = Path(output_root).resolve()
    spec = load_spec(spec_file)
    validate_spec(spec)
    mermaid_dir = root / "mermaid"
    generated_dir = root / "generated"
    mermaid_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    for page in spec["pages"]:
        output = mermaid_dir / page["mermaid_file"]
        _ensure_under_root(output, root)
        output.write_text(_render_mermaid(page, spec), encoding="utf-8", newline="\n")
        LOGGER.info("Generated Mermaid: %s", output)
    drawio_path = root / "TRIAGE_EG_Complete_System.drawio"
    _write_drawio(drawio_path, spec)
    summary = _build_summary(spec, spec_file)
    summary_path = generated_dir / "architecture_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_path = generated_dir / "architecture_quality_report.md"
    report_path.write_text(_build_quality_report(summary), encoding="utf-8", newline="\n")
    LOGGER.info("Generated draw.io, summary and quality report under %s", root)
    return summary


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"Refusing to write outside output root: {path}") from error


def _badge_text(node: Mapping[str, Any]) -> str:
    scope = f": {node['criticality_scope']}" if node.get("criticality_scope") else ""
    return f"[{node['criticality']}{scope}] [{node['maturity']}] [{node['source_basis']}]"


def _short(value: Any, limit: int = 86) -> str:
    if isinstance(value, Sequence) and not isinstance(value, str):
        value = " → ".join(str(item) for item in value)
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _render_mermaid(page: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    quote = chr(34)
    lines = [f"%% {page['title']}", f"flowchart {page['direction']}"]
    grouped: dict[str | None, list[Mapping[str, Any]]] = {}
    for item in page["nodes"]:
        grouped.setdefault(item.get("group"), []).append(item)
    for container in page["groups"]:
        lines.append(
            f"  subgraph {container['id']}[{quote}{_mermaid_text(container['title'])}{quote}]"
        )
        for item in grouped.get(container["id"], []):
            lines.append(f"    {_mermaid_node(item, page['detail_level'], quote)}")
        lines.append("  end")
    for item in grouped.get(None, []):
        lines.append(f"  {_mermaid_node(item, page['detail_level'], quote)}")
    for item in page["edges"]:
        lines.append(f"  {_mermaid_edge(item)}")
    for layer, style in spec["styles"]["layers"].items():
        lines.append(
            f"  classDef layer_{layer} fill:{style['fill']},stroke:{style['stroke']},"
            f"color:{style['text']},stroke-width:1.5px;"
        )
    for criticality, style in spec["criticalities"].items():
        dash = ",stroke-dasharray:5 3" if style.get("dashed") else ""
        lines.append(
            f"  classDef criticality_{criticality} stroke:{style['stroke']},"
            f"stroke-width:{style['stroke_width']}px{dash};"
        )
    for item in page["nodes"]:
        lines.append(
            f"  class {item['id']} layer_{item['layer']},criticality_{item['criticality']};"
        )
    return "\n".join(lines) + "\n"


def _mermaid_node(node: Mapping[str, Any], detail_level: str, quote: str) -> str:
    lines = [f"{node['number']} — {node['title']}", _badge_text(node)]
    if detail_level == "overview":
        lines += [
            _short(node["responsibility"]),
            f"In: {_short(node['inputs'], 48)}",
            f"Out: {_short(node['outputs'], 48)}",
        ]
    else:
        lines += [
            f"In: {_short(node['inputs'], 52)}",
            f"Do: {_short(node['processing'], 58)}",
            f"Impl: {_short(node['implementations'], 58)}",
            f"Out: {_short(node['outputs'], 52)}",
        ]
    label = _mermaid_text("<br/>".join(lines))
    syntax = {
        "store": ("[(", ")]"),
        "decision": ("{", "}"),
        "control": ("{{", "}}"),
        "artifact": ("[/", "/]"),
        "graph_node": ("((", "))"),
        "offpage": ("[/", "\\]"),
        "note": ("[", "]"),
        "process": ("(", ")"),
    }[node["type"]]
    return f"{node['id']}{syntax[0]}{quote}{label}{quote}{syntax[1]}"


def _mermaid_edge(edge: Mapping[str, Any]) -> str:
    label = _mermaid_text(edge["label"])
    flow = edge["flow_type"]
    if flow == "match":
        return f"{edge['source']} ==>|MATCH: {label}| {edge['target']}"
    if flow in {"data", "evidence"}:
        prefix = "EVIDENCE: " if flow == "evidence" else ""
        return f"{edge['source']} -->|{prefix}{label}| {edge['target']}"
    prefix = {
        "fallback": "FALLBACK: ",
        "artifact": "ARTIFACT: ",
        "optional": "OPTIONAL: ",
        "control": "CONTROL: ",
    }[flow]
    return f"{edge['source']} -.->|{prefix}{label}| {edge['target']}"


def _mermaid_text(value: Any) -> str:
    return str(value).replace(chr(34), chr(39)).replace("\n", " ").replace("|", "/")


def _write_drawio(path: Path, spec: Mapping[str, Any]) -> None:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "TRIAGE-EG v1.1 architecture generator",
            "version": "1.1",
            "compressed": "false",
        },
    )
    fonts = spec["styles"]["fonts"]
    for page in spec["pages"]:
        bounds = analyze_page_layout(page)
        diagram = ET.SubElement(mxfile, "diagram", {"id": page["id"], "name": page["title"]})
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1422",
                "dy": "794",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(bounds["page_width"]),
                "pageHeight": str(bounds["page_height"]),
                "math": "0",
                "shadow": "0",
            },
        )
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        title = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"{page['id']}_TITLE",
                "value": page["title"],
                "style": (
                    "text;html=1;strokeColor=none;fillColor=none;align=left;"
                    "verticalAlign=middle;fontStyle=1;"
                    f"fontSize={fonts['page_title']};"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        _drawio_geometry(
            title, {"x": 20, "y": 0, "width": min(1500, bounds["page_width"] - 40), "height": 36}
        )
        for container in page["groups"]:
            _drawio_group(root, page, container, spec)
        for item in page["nodes"]:
            _drawio_node(root, page, item, spec)
        for item in page["edges"]:
            _drawio_edge(root, page, item)
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(
        path, encoding="utf-8", xml_declaration=True, short_empty_elements=True
    )


def _drawio_group(
    root: ET.Element, page: Mapping[str, Any], group: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    palette = spec["styles"]["layers"][group["layer"]]
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": f"{page['id']}_{group['id']}",
            "value": group["title"],
            "style": (
                "swimlane;html=1;rounded=1;collapsible=0;startSize=38;horizontal=1;"
                f"fillColor={palette['fill']};strokeColor={palette['stroke']};fontStyle=1;"
                f"fontSize={spec['styles']['fonts']['group_title']};opacity=28;"
            ),
            "vertex": "1",
            "parent": "1",
        },
    )
    _drawio_geometry(cell, group["geometry"])


def _drawio_node(
    root: ET.Element, page: Mapping[str, Any], node: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    palette = spec["styles"]["layers"][node["layer"]]
    criticality = spec["criticalities"][node["criticality"]]
    shape = {
        "process": "rounded=1;arcSize=12;",
        "store": "shape=cylinder3;boundedLbl=1;backgroundOutline=1;",
        "artifact": "shape=document;",
        "decision": "shape=rhombus;",
        "control": "shape=hexagon;perimeter=hexagonPerimeter2;",
        "graph_node": "ellipse;",
        "note": "shape=note;",
        "offpage": "shape=offPageConnector;size=0.15;",
    }[node["type"]]
    dashed = "dashed=1;dashPattern=5 3;" if criticality.get("dashed") else ""
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": f"{page['id']}_{node['id']}",
            "value": _drawio_label(page, node, spec),
            "tooltip": _drawio_tooltip(node),
            "style": (
                f"{shape}whiteSpace=wrap;html=1;align=left;verticalAlign=top;spacing=8;"
                f"fillColor={palette['fill']};strokeColor={criticality['stroke']};"
                f"strokeWidth={criticality['stroke_width']};fontColor={palette['text']};"
                f"fontSize={spec['styles']['fonts']['body']};{dashed}"
            ),
            "vertex": "1",
            "parent": "1",
        },
    )
    _drawio_geometry(cell, node["geometry"])


def _drawio_label(page: Mapping[str, Any], node: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    fonts = spec["styles"]["fonts"]
    title = (
        f"<font style='font-size:{fonts['module_title']}px'>"
        f"<b>{node['number']} — {node['title']}</b></font>"
    )
    badges = f"<font style='font-size:{fonts['badge']}px'><b>{_badge_text(node)}</b></font>"
    if page["detail_level"] == "overview":
        body = [
            _short(node["responsibility"]),
            f"<b>In:</b> {_short(node['inputs'], 62)}",
            f"<b>Out:</b> {_short(node['outputs'], 62)}",
        ]
    else:
        body = [
            f"<b>In:</b> {_short(node['inputs'], 68)}",
            f"<b>Process:</b> {_short(node['processing'], 76)}",
            f"<b>Impl:</b> {_short(node['implementations'], 76)}",
            f"<b>Out:</b> {_short(node['outputs'], 68)}",
        ]
    return "<br>".join([title, badges, *body])


def _drawio_tooltip(node: Mapping[str, Any]) -> str:
    owner = (
        f"architecture={','.join(node['architecture_owner'])}; "
        f"implementation={','.join(node['implementation_owner'])}; "
        f"reviewers={','.join(node['reviewers']) or 'none'}"
    )
    lines = [
        f"Responsibility: {_short(node['responsibility'], 500)}",
        f"Non-responsibility: {_short(node['non_responsibility'], 500)}",
        f"Implementation: {_short(node['implementations'], 500)}",
        f"Artifacts: {_short(node['artifacts'], 500)}",
        f"Metrics: {_short(node['metrics'], 500)}",
        f"Failure modes: {_short(node['failure_modes'], 500)}",
        f"Fallback: {_short(node['fallback'], 500)}",
        f"Owner: {owner}",
    ]
    return "\n".join(lines)


def _drawio_geometry(cell: ET.Element, geometry: Mapping[str, Any]) -> None:
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": str(geometry["x"]),
            "y": str(geometry["y"]),
            "width": str(geometry["width"]),
            "height": str(geometry["height"]),
            "as": "geometry",
        },
    )


def _drawio_edge(root: ET.Element, page: Mapping[str, Any], edge: Mapping[str, Any]) -> None:
    flow = edge["flow_type"]
    colors = {
        "data": "#334155",
        "control": "#a16207",
        "optional": "#7c3aed",
        "fallback": "#b91c1c",
        "evidence": "#be123c",
        "artifact": "#475569",
        "match": "#0f766e",
    }
    dashed = flow in {"control", "optional", "fallback", "artifact"}
    prefix = {
        "control": "CONTROL: ",
        "optional": "OPTIONAL: ",
        "fallback": "FALLBACK: ",
        "evidence": "EVIDENCE: ",
        "artifact": "ARTIFACT: ",
        "match": "MATCH: ",
        "data": "",
    }[flow]
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": f"{page['id']}_{edge['id']}",
            "value": prefix + edge["label"],
            "style": (
                "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
                "jettySize=auto;html=1;endArrow=block;endFill=1;"
                f"strokeColor={colors[flow]};"
                f"strokeWidth={'3' if flow == 'match' else '2'};"
                f"{'dashed=1;dashPattern=5 3;' if dashed else ''}"
            ),
            "edge": "1",
            "parent": "1",
            "source": f"{page['id']}_{edge['source']}",
            "target": f"{page['id']}_{edge['target']}",
        },
    )
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if edge.get("waypoints"):
        points = ET.SubElement(geometry, "Array", {"as": "points"})
        for point in edge["waypoints"]:
            ET.SubElement(points, "mxPoint", {"x": str(point["x"]), "y": str(point["y"])})


def _build_summary(spec: Mapping[str, Any], spec_path: Path) -> dict[str, Any]:
    nodes = [item for page in spec["pages"] for item in page["nodes"]]
    layouts = {page["id"]: analyze_page_layout(page) for page in spec["pages"]}
    warnings = [
        f"{page_id}: {warning}"
        for page_id, layout in layouts.items()
        for warning in layout["minimum_gap_warnings"]
    ]
    pages = [
        {
            "id": page["id"],
            "title": page["title"],
            "mermaid_file": page["mermaid_file"],
            "module_count": sum(item["type"] in PROCESSING_TYPES for item in page["nodes"]),
            "node_count": len(page["nodes"]),
            "edge_count": len(page["edges"]),
            **layouts[page["id"]],
        }
        for page in spec["pages"]
    ]
    owner_counts = {
        field: dict(sorted(Counter(owner for item in nodes for owner in item[field]).items()))
        for field in ("architecture_owner", "implementation_owner", "reviewers")
    }
    confirmed = [
        {"id": model_id, "selected": model["selected"], "source_basis": model["source_basis"]}
        for model_id, model in sorted(spec["models"].items())
        if model["selected"]
    ]
    unselected = [
        model_id for model_id, model in sorted(spec["models"].items()) if model["selected"] is None
    ]
    return {
        "repository": spec["project"]["repository"],
        "system_name": spec["project"]["system_name"],
        "version": str(spec["project"]["version"]),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "page_count": len(pages),
        "module_count": sum(page["module_count"] for page in pages),
        "node_count": len(nodes),
        "edge_count": sum(len(page["edges"]) for page in spec["pages"]),
        "pages": pages,
        "criticality_statistics": dict(
            sorted(Counter(item["criticality"] for item in nodes).items())
        ),
        "maturity_statistics": dict(sorted(Counter(item["maturity"] for item in nodes).items())),
        "source_basis_statistics": dict(
            sorted(Counter(item["source_basis"] for item in nodes).items())
        ),
        "ownership_statistics": owner_counts,
        "models_confirmed": confirmed,
        "models_unselected": unselected,
        "contract_quality": {
            "specific_node_contracts": len(nodes),
            "placeholder_node_contracts": 0,
            "edge_aligned_interfaces": sum(
                len(page["nodes"])
                for page in spec["pages"]
                if all(
                    set(node["dependencies"])
                    == {
                        edge["source"]
                        for edge in page["edges"]
                        if edge["target"] == node["id"]
                    }
                    and set(node["next_modules"])
                    == {
                        edge["target"]
                        for edge in page["edges"]
                        if edge["source"] == node["id"]
                    }
                    for node in page["nodes"]
                )
            ),
        },
        "warnings": warnings,
        "validation_status": "PASS",
    }


def _build_quality_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# TRIAGE-EG Architecture Quality Report",
        "",
        "## Version 1.1 changes",
        "",
        "- Split module status into criticality, maturity, and source basis.",
        "- Rebuilt Page 04 as an actual event graph and made frame banks parallel.",
        (
            "- Added application backend, interactive UI, automatic mode, control bus, "
            "and three-axis ownership."
        ),
        "- Added deterministic layout, shape, graph, taxonomy, and ownership checks.",
        "- Replaced generic contract placeholders with edge-aligned, module-specific interfaces.",
        "",
        "## Page dimensions",
        "",
        "| Page | Width | Height | Aspect | Modules | Nodes | Edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["pages"]:
        lines.append(
            f"| {item['id']} | {item['page_width']} | {item['page_height']} | "
            f"{item['aspect_ratio']:.3f} | {item['module_count']} | "
            f"{item['node_count']} | {item['edge_count']} |"
        )
    lines += ["", "## Layout warnings", ""]
    lines += [f"- {warning}" for warning in summary["warnings"]] or ["- None."]
    lines += [
        "",
        "## Status migration",
        "",
        "- Legacy node `status` and single `owner` fields: removed.",
        (
            "- Criticality, maturity, source basis, architecture owner, "
            "implementation owner, and reviewers: validated."
        ),
        "",
        "## Contract quality",
        "",
        (
            f"- Specific node contracts: "
            f"{summary['contract_quality']['specific_node_contracts']}."
        ),
        (
            f"- Placeholder node contracts: "
            f"{summary['contract_quality']['placeholder_node_contracts']}."
        ),
        (
            f"- Edge-aligned interfaces: "
            f"{summary['contract_quality']['edge_aligned_interfaces']}."
        ),
        "",
        "## Event Graph checks",
        "",
        (
            "- QueryEvent, Video, SegmentEvent, Entity, SemanticMoment, and external "
            "EvidenceRef are present."
        ),
        (
            "- BEFORE, CONTAINS, PARTICIPATES_IN, POSSIBLE_SAME_ENTITY, ANCHORS, "
            "SUPPORTS, and MATCH are present."
        ),
        "- Matching produces Event Match Matrix → solver → Top-M Event Chains.",
        "",
        "## Task-criticality checks",
        "",
        "- Semantic Moment Localizer: CORE for TRAKE, EXPERIMENTAL.",
        "- Q&A Answer Extractor: CORE for Q&A, EXPERIMENTAL.",
        "- Team Visual Encoder: CONDITIONAL; Bounded Agent Planner: OPTIONAL.",
        "",
        "## Ownership distribution",
        "",
    ]
    for field, values in summary["ownership_statistics"].items():
        lines.append(
            f"- {field}: " + ", ".join(f"{owner}={count}" for owner, count in values.items())
        )
    lines += [
        "",
        "## Remaining open questions",
        "",
        "- Benchmark and select the Team Visual Encoder.",
        "- Benchmark TransNetV2 and frame-selection strategies.",
        "- Decide the production ANN implementation after benchmark.",
        "- Decide whether optional OCR, ASR, caption, or VLM branches justify their cost.",
        (
            "- Validate Event Graph solver, Semantic Moment Localizer, and Q&A extractor "
            "on official task data."
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        summary = generate_assets(args.spec, args.output_root)
    except (FileNotFoundError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 2
    LOGGER.info(
        "Architecture generation complete: %s pages, %s nodes, %s edges",
        summary["page_count"],
        summary["node_count"],
        summary["edge_count"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
