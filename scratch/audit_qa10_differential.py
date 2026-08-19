#!/usr/bin/env python3
"""QA-10 Differential Forensic Audit (Contract A vs Contract B).

Compares QA-10 under:
  - Contract A (874c5f5-compatible): question_en = q_en
  - Contract B (4b64603 canonical): question_en = None

Both with:
  - event_description_en = q_en
  - include_vi_variant = False
  - VisualOntologyConfig.enabled = True
  - qa_unsupported_provider_fallback = True
  - S2D1 + S2E1 frozen
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.visual_ontology import VisualOntologyConfig


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_marks.split())


def resolve_ocr_config() -> OCRAnswerProviderConfig:
    tess_path = shutil.which("tesseract")
    available_langs: list[str] = []
    if tess_path:
        try:
            import subprocess
            res = subprocess.run([tess_path, "--list-langs"], capture_output=True, text=True, check=False)
            available_langs = [l.strip() for l in res.stdout.splitlines()[1:] if l.strip()]
        except Exception:
            pass

    desired = ("eng", "vie")
    supported = tuple(l for l in desired if l in available_langs)
    if not supported:
        supported = tuple(available_langs[:2]) if available_langs else ("eng",)

    if not available_langs:
        return OCRAnswerProviderConfig(enabled=False, languages=("eng",))

    return OCRAnswerProviderConfig(
        enabled=True,
        languages=supported,
        evidence_frame_budget=8,
    )


def resolve_visual_ontology_config() -> VisualOntologyConfig:
    candidates = [
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/l21_150_diagnostic/qa_dev_visual_ontology.json"),
    ]
    for p in candidates:
        if p.exists():
            return VisualOntologyConfig(
                enabled=True,
                ontology_path=p,
                evidence_frame_budget=16,
                max_active_domains=2,
            )
    return VisualOntologyConfig(enabled=False)


def run_audit() -> None:
    print("=" * 115)
    print("QA-10 TARGETED DIFFERENTIAL FORENSIC AUDIT (CONTRACT A vs CONTRACT B)")
    print("=" * 115)

    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}
    
    q_info = next(q for q in bm_data["queries"] if q.get("query_id") == "QA-10")
    qid = "QA-10"
    target_vid = q_info["video_id"]
    s_gt, e_gt = map(int, q_info["proposed_interval"])
    raw_accepted = q_info.get("accepted_answers") or [q_info.get("answer", "")]
    gold_answers = [normalize_text(a) for a in raw_accepted if a]
    q_vi = q_info.get("question_vi", "")
    q_en = en_map.get(qid, "")

    print(f"Query ID             : {qid}")
    print(f"Target Video         : {target_vid}")
    print(f"Ground Truth Interval: [{s_gt}..{e_gt}]")
    print(f"Gold Answers         : {gold_answers}")
    print(f"Question VI          : {q_vi}")
    print(f"Question EN          : {q_en}")
    print("=" * 115)

    session_output = Path("/kaggle/working/qa10_diff_output") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa10_diff_output"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    evidence_config = QAVideoConditionedEvidenceConfig(
        enabled=True,
        selected_video_cap=16,
        anchors_per_video=5,
        video_rrf_constant=60.0,
        preserve_keyframe_evidence=True,
        keyframe_evidence_video_cap=16,
        keyframe_evidence_anchors_per_video=1,
        temporal_refinement_enabled=True,
        temporal_seed_anchors_per_video=2,
        temporal_refinement_video_cap=8,
        temporal_refinement_total_seed_cap=16,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
        count_far_alt_micro=False,
        top1_secondary_refined_rescue_enabled=True,
        top1_secondary_refined_rescue_span_candidateizer=True,
        top1_secondary_refined_rescue_tail_budget=5,
    )

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else Path("scratch/manifest_cache.json"),
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=evidence_config,
        qa_visual_ontology_config=resolve_visual_ontology_config(),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
        qa_unsupported_provider_fallback=True,
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    runtime = OperationalKISRuntime.bootstrap(config)

    contracts = [
        ("Contract A (874c5f5: question_en=q_en)", q_en),
        ("Contract B (4b64603: question_en=None)", None),
    ]

    for label, q_en_val in contracts:
        print("\n" + "#" * 115)
        print(f"RUNNING: {label}")
        print("#" * 115)

        req = QAQueryRequest(
            request_id=f"audit-qa10-{int(time.time()*1000)}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=q_en_val,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )

        t0 = time.time()
        res, timings, diag = runtime.qa_pipeline.process_qa_query(req)
        exec_time = time.time() - t0

        qtype = diag.get("question_type")
        cls_reason = diag.get("question_classification_reason")
        cls_policy = diag.get("question_classifier_policy")
        selected_vids = diag.get("selected_video_ids", [])
        
        print(f"\n1. CLASSIFICATION & CAPABILITY:")
        print(f"   • Runtime QuestionType : {qtype}")
        print(f"   • Classifier Policy    : {cls_policy}")
        print(f"   • Classifier Reason    : {cls_reason}")

        print(f"\n2. VIDEO SELECTION (Nomination Depth: {len(selected_vids)}):")
        vid_in_sel = target_vid in selected_vids
        vid_rank = selected_vids.index(target_vid) + 1 if vid_in_sel else None
        print(f"   • Target Video ({target_vid}) Nominated? : {vid_in_sel} (Nomination Rank: {vid_rank})")
        print(f"   • Selected Videos List: {selected_vids[:8]}")

        print(f"\n3. ANSWER HYPOTHESES & PROVIDER:")
        cand_provider = runtime.qa_pipeline.candidate_provider
        prov_class = cand_provider.__class__.__name__
        answer_text = req.question_en or req.question
        hypotheses = runtime.qa_pipeline._answer_hypotheses(res.question_type, answer_text)
        hyp_ans_list = [h.canonical_answer for h in hypotheses]
        contains_xich_du = any(normalize_text(h) in gold_answers for h in hyp_ans_list)
        print(f"   • Candidate Provider Class : {prov_class}")
        print(f"   • Text fed to Hypotheses   : '{answer_text}'")
        print(f"   • Total Hypotheses Count   : {len(hypotheses)}")
        print(f"   • Hypotheses List          : {hyp_ans_list}")
        print(f"   • Contains 'xích đu'?      : {contains_xich_du}")

        print(f"\n4. EVIDENCE CANDIDATES FOR TARGET VIDEO ({target_vid}):")
        usable_cands = diag.get("usable_evidence_candidates", [])
        target_ev = [c for c in usable_cands if c.get("video_id") == target_vid]
        print(f"   • Total Usable Evidence Candidates : {len(usable_cands)}")
        print(f"   • Evidence Candidates for {target_vid} : {len(target_ev)}")
        for idx, ec in enumerate(target_ev):
            fid = ec.get("frame_id")
            in_gt = s_gt <= fid <= e_gt
            dist = 0 if in_gt else min(abs(fid - s_gt), abs(fid - e_gt))
            print(f"     [{idx+1}] Frame={fid:<6} | Dist to GT={dist:<5} | In GT?={in_gt} | Source={ec.get('evidence_source')} | Rank={ec.get('rank')}")

        print(f"\n5. FINAL PREDICTIONS TOP-100 (Total Predictions: {len(res.predictions)}):")
        target_preds = [p for p in res.predictions if p.video_id == target_vid]
        print(f"   • Predictions for {target_vid} : {len(target_preds)}")
        
        hit_preds = []
        for p in target_preds:
            fid = p.frame_id
            ans = p.answer
            norm_a = normalize_text(ans)
            in_gt = (s_gt <= fid <= e_gt)
            match_a = (norm_a in gold_answers)
            is_hit = in_gt and match_a
            if is_hit:
                hit_preds.append(p)
            print(f"     -> Rank={p.rank:<3} | Frame={fid:<6} (In GT={str(in_gt):<5}) | Answer='{ans}' (Match Gold={str(match_a):<5}) | STRICT HIT={is_hit}")

        print(f"\n6. OFFICIAL VERDICT & FIRST FAILURE TAXONOMY:")
        if hit_preds:
            verdict = f"STRICT HIT @{hit_preds[0].rank} ✅"
            taxonomy = "STRICT_HIT (SUCCESS)"
            detail = f"Hit at Rank {hit_preds[0].rank} | Frame {hit_preds[0].frame_id} | Answer '{hit_preds[0].answer}'"
        elif not vid_in_sel:
            verdict = "NO HIT ❌"
            taxonomy = "VIDEO_ABSENT"
            detail = f"Target video {target_vid} was not nominated in Top {len(selected_vids)} videos"
        elif not target_ev:
            verdict = "NO HIT ❌"
            taxonomy = "EVIDENCE_ABSENT"
            detail = f"Target video nominated but no evidence frames were selected"
        elif not any(s_gt <= ec.get("frame_id", -1) <= e_gt for ec in target_ev):
            verdict = "NO HIT ❌"
            taxonomy = "TEMPORAL_MISS"
            closest_f = min([ec.get("frame_id", -9999) for ec in target_ev], key=lambda f: min(abs(f - s_gt), abs(f - e_gt)))
            detail = f"Evidence frames extracted [{', '.join(str(ec.get('frame_id')) for ec in target_ev)}], closest is {closest_f} (dist={min(abs(closest_f-s_gt), abs(closest_f-e_gt))})"
        elif not contains_xich_du:
            verdict = "NO HIT ❌"
            taxonomy = "ANSWER_MISS"
            detail = f"Gold answer 'xích đu' was missing from candidate hypotheses"
        else:
            verdict = "NO HIT ❌"
            taxonomy = "ALLOCATION_MISS"
            detail = f"Frame in GT and Gold Answer both present, but displaced or rejected in Top-100 constructor"

        print(f"   • Verdict          : {verdict}")
        print(f"   • Failure Taxonomy : >> {taxonomy} <<")
        print(f"   • Failure Detail   : {detail}")
        print(f"   • Latency          : {exec_time:.2f}s")


if __name__ == "__main__":
    run_audit()
