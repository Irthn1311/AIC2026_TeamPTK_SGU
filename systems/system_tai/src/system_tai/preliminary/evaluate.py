import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .evaluation import evaluate_dataset, evaluate_ranked_query
from .matching import NormalizedAliasAnswerMatcher
from .schemas import (
    KISGroundTruth,
    KISPrediction,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
)
from .scoring import (
    score_kis_prediction,
    score_qa_prediction,
    score_trake_prediction,
)
from .validation import validate_ranked_top100


def main() -> None:
    parser = argparse.ArgumentParser(description="system_tai preliminary offline evaluator")
    parser.add_argument("--task", required=True, choices=["kis", "qa", "trake"])
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    # Load GT
    with open(args.ground_truth, encoding="utf-8") as f:
        gt_data = json.load(f)

    # Build GT mapping: query_id -> GT object
    gts = {}
    for entry in gt_data:
        qid = entry["query_id"]
        if args.task == "kis":
            gts[qid] = KISGroundTruth(
                query_id=qid,
                video_id=entry["video_id"],
                start_frame_id=entry["start_frame_id"],
                end_frame_id=entry["end_frame_id"],
            )
        elif args.task == "qa":
            gts[qid] = QAGroundTruth(
                query_id=qid,
                video_id=entry["video_id"],
                start_frame_id=entry["start_frame_id"],
                end_frame_id=entry["end_frame_id"],
                accepted_answers=tuple(entry["accepted_answers"]),
            )
        elif args.task == "trake":
            intervals = tuple((i[0], i[1]) for i in entry["event_intervals"])
            gts[qid] = TRAKEGroundTruth(
                query_id=qid,
                video_id=entry["video_id"],
                event_intervals=intervals,
            )

    # Load Predictions (JSONL)
    preds_by_query = {}
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry["query_id"]
            if qid not in preds_by_query:
                preds_by_query[qid] = []

            if args.task == "kis":
                preds_by_query[qid].append(
                    KISPrediction(
                        query_id=qid,
                        rank=entry["rank"],
                        video_id=entry["video_id"],
                        frame_id=entry["frame_id"],
                    )
                )
            elif args.task == "qa":
                preds_by_query[qid].append(
                    QAPrediction(
                        query_id=qid,
                        rank=entry["rank"],
                        video_id=entry["video_id"],
                        frame_id=entry["frame_id"],
                        answer=entry["answer"],
                    )
                )
            elif args.task == "trake":
                preds_by_query[qid].append(
                    TRAKEPrediction(
                        query_id=qid,
                        rank=entry["rank"],
                        video_id=entry["video_id"],
                        frame_ids=tuple(entry["frame_ids"]),
                    )
                )

    # Evaluate
    reports = []
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)

    for qid, preds in preds_by_query.items():
        if qid not in gts:
            print(f"Warning: predictions for {qid} without GT", file=sys.stderr)
            continue

        gt = gts[qid]

        errors = validate_ranked_top100(preds, args.task, gt)
        if errors:
            print(f"Validation failed for query {qid}:", file=sys.stderr)
            for err in errors:
                print(f" - {err.message}", file=sys.stderr)
            sys.exit(1)

        if args.task == "kis":
            scorer = score_kis_prediction
        elif args.task == "qa":
            def scorer(p, g):
                return score_qa_prediction(p, g, matcher)
        elif args.task == "trake":
            scorer = score_trake_prediction

        report = evaluate_ranked_query(qid, args.task, preds, gt, scorer)
        reports.append(report)

    dataset_report = evaluate_dataset(reports)

    # Output structured deterministic JSON
    output = asdict(dataset_report)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
