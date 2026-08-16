"""Opt-in video-conditioned multi-video evidence grounding for QA."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.retrieval.multi_query import QueryVariant, WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoRestrictedSearchOutcome,
)

QA_VIDEO_CONDITIONED_EVIDENCE_V1 = "QA_VIDEO_CONDITIONED_EVIDENCE_V1"
QA_KEYFRAME_EVIDENCE_BANK_V1 = "QA_KEYFRAME_EVIDENCE_BANK_V1"
QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1 = "QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1"
QA_CANDIDATE_ORDER_ROUND_ROBIN = "round_robin"
QA_CANDIDATE_ORDER_GLOBAL_RESTRICTED_COSINE = "global_restricted_cosine"
QA_CANDIDATE_ORDERING_POLICIES = frozenset(
    {
        QA_CANDIDATE_ORDER_ROUND_ROBIN,
        QA_CANDIDATE_ORDER_GLOBAL_RESTRICTED_COSINE,
    }
)
KEYFRAME_ANCHOR = "KEYFRAME_ANCHOR"
RAW_REFINED = "RAW_REFINED"
TEMPORAL_REFINED = "TEMPORAL_REFINED"


@dataclass(frozen=True, slots=True)
class QAVideoConditionedEvidenceConfig:
    """Small, target-agnostic configuration for the QA-A1 grounding path."""

    enabled: bool = False
    selected_video_cap: int = 32
    anchors_per_video: int = 5
    video_rrf_constant: float = 60.0
    candidate_ordering_policy: str = QA_CANDIDATE_ORDER_ROUND_ROBIN
    preserve_keyframe_evidence: bool = False
    keyframe_evidence_video_cap: int = 32
    keyframe_evidence_anchors_per_video: int = 1
    temporal_refinement_enabled: bool = False
    temporal_seed_anchors_per_video: int = 3
    temporal_refinement_video_cap: int = 32
    temporal_refinement_total_seed_cap: int = 96
    secondary_temporal_micro_budget: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.secondary_temporal_micro_budget) is not bool:
            raise ValueError("secondary_temporal_micro_budget must be a boolean")
        if type(self.selected_video_cap) is not int or self.selected_video_cap < 1:
            raise ValueError("selected_video_cap must be an integer >= 1")
        if type(self.anchors_per_video) is not int or self.anchors_per_video < 1:
            raise ValueError("anchors_per_video must be an integer >= 1")
        if (
            type(self.candidate_ordering_policy) is not str
            or self.candidate_ordering_policy not in QA_CANDIDATE_ORDERING_POLICIES
        ):
            raise ValueError(
                "candidate_ordering_policy must be one of: "
                + ", ".join(sorted(QA_CANDIDATE_ORDERING_POLICIES))
            )
        if type(self.preserve_keyframe_evidence) is not bool:
            raise ValueError("preserve_keyframe_evidence must be a boolean")
        if (
            type(self.keyframe_evidence_video_cap) is not int
            or self.keyframe_evidence_video_cap < 1
        ):
            raise ValueError("keyframe_evidence_video_cap must be an integer >= 1")
        if (
            type(self.keyframe_evidence_anchors_per_video) is not int
            or not 1 <= self.keyframe_evidence_anchors_per_video <= 100
        ):
            raise ValueError(
                "keyframe_evidence_anchors_per_video must be in [1, 100]"
            )
        if (
            self.preserve_keyframe_evidence
            and self.keyframe_evidence_video_cap > self.selected_video_cap
        ):
            raise ValueError(
                "keyframe_evidence_video_cap must not exceed selected_video_cap"
            )
        if self.preserve_keyframe_evidence and not self.enabled:
            raise ValueError(
                "preserve_keyframe_evidence requires video-conditioned evidence"
            )
        if (
            self.keyframe_evidence_anchors_per_video != 1
            and not self.preserve_keyframe_evidence
        ):
            raise ValueError(
                "multi-anchor keyframe evidence requires the keyframe evidence bank"
            )
        if (
            self.preserve_keyframe_evidence
            and self.keyframe_evidence_anchors_per_video > self.anchors_per_video
        ):
            raise ValueError(
                "keyframe_evidence_anchors_per_video must not exceed anchors_per_video"
            )
        if (
            self.preserve_keyframe_evidence
            and self.keyframe_evidence_video_cap
            * self.keyframe_evidence_anchors_per_video
            > 100
        ):
            raise ValueError(
                "multi-anchor keyframe evidence capacity must not exceed 100"
            )
        if type(self.temporal_refinement_enabled) is not bool:
            raise ValueError("temporal_refinement_enabled must be a boolean")
        if (
            type(self.temporal_seed_anchors_per_video) is not int
            or not 1 <= self.temporal_seed_anchors_per_video <= 100
        ):
            raise ValueError(
                "temporal_seed_anchors_per_video must be in [1, 100]"
            )
        if (
            type(self.temporal_refinement_video_cap) is not int
            or not 1 <= self.temporal_refinement_video_cap <= 100
        ):
            raise ValueError(
                "temporal_refinement_video_cap must be in [1, 100]"
            )
        maximum_temporal_seeds = (
            self.temporal_refinement_video_cap
            * self.temporal_seed_anchors_per_video
        )
        if (
            type(self.temporal_refinement_total_seed_cap) is not int
            or not 1
            <= self.temporal_refinement_total_seed_cap
            <= 100
        ):
            raise ValueError(
                "temporal_refinement_total_seed_cap must be in [1, 100]"
            )
        if self.temporal_refinement_enabled:
            if self.keyframe_evidence_anchors_per_video != 1:
                raise ValueError(
                    "multi-anchor keyframe-only evidence cannot be combined with "
                    "temporal refinement"
                )
            if not (self.enabled and self.preserve_keyframe_evidence):
                raise ValueError(
                    "temporal_refinement_enabled requires video-conditioned evidence "
                    "and the keyframe evidence bank"
                )
            if self.temporal_seed_anchors_per_video > self.anchors_per_video:
                raise ValueError(
                    "temporal_seed_anchors_per_video must not exceed anchors_per_video"
                )
            if self.temporal_refinement_video_cap > self.selected_video_cap:
                raise ValueError(
                    "temporal_refinement_video_cap must not exceed selected_video_cap"
                )
            if self.temporal_refinement_video_cap > self.keyframe_evidence_video_cap:
                raise ValueError(
                    "temporal_refinement_video_cap must not exceed "
                    "keyframe_evidence_video_cap"
                )
            if self.temporal_refinement_total_seed_cap > maximum_temporal_seeds:
                raise ValueError(
                    "temporal_refinement_total_seed_cap must not exceed the bounded "
                    "video/anchor capacity"
                )
            if (
                self.keyframe_evidence_video_cap
                * self.temporal_seed_anchors_per_video
                > 100
            ):
                raise ValueError(
                    "multi-seed temporal evidence capacity must not exceed 100"
                )
        if (
            type(self.video_rrf_constant) is bool
            or not isinstance(self.video_rrf_constant, (int, float))
            or not math.isfinite(float(self.video_rrf_constant))
            or self.video_rrf_constant <= 0
        ):
            raise ValueError("video_rrf_constant must be finite and positive")


def select_primary_keyframe_anchors(
    candidates: Sequence[CandidateFrame],
    *,
    video_cap: int,
) -> tuple[CandidateFrame, ...]:
    """Select one deterministic local-rank-one anchor per nominated video."""

    if type(video_cap) is not int or video_cap < 1:
        raise ValueError("video_cap must be an integer >= 1")
    eligible: list[tuple[int, str, int, int, int, CandidateFrame]] = []
    for candidate in candidates:
        metadata = dict(candidate.diagnostic_metadata or {})
        if metadata.get("local_anchor_rank") != 1:
            continue
        nomination_rank = metadata.get("video_nomination_rank")
        if type(nomination_rank) is not int or nomination_rank < 1:
            raise ValueError(
                "primary QA keyframe anchor requires a valid video_nomination_rank"
            )
        eligible.append(
            (
                nomination_rank,
                candidate.video_id,
                candidate.frame_id,
                candidate.clip_row,
                candidate.rank,
                candidate,
            )
        )
    eligible.sort(key=lambda item: item[:-1])
    selected: list[CandidateFrame] = []
    seen_videos: set[str] = set()
    for *_ordering, candidate in eligible:
        if candidate.video_id in seen_videos:
            continue
        seen_videos.add(candidate.video_id)
        selected.append(candidate)
        if len(selected) >= video_cap:
            break
    return tuple(selected)


def select_temporal_seed_anchors(
    candidates: Sequence[CandidateFrame],
    *,
    anchors_per_video: int,
    video_cap: int,
    total_seed_cap: int | None = None,
) -> tuple[CandidateFrame, ...]:
    """Select bounded, target-agnostic multi-seed anchors in deterministic order."""

    if type(anchors_per_video) is not int or anchors_per_video < 1:
        raise ValueError("anchors_per_video must be an integer >= 1")
    if type(video_cap) is not int or video_cap < 1:
        raise ValueError("video_cap must be an integer >= 1")
    if total_seed_cap is not None and (
        type(total_seed_cap) is not int or total_seed_cap < 1
    ):
        raise ValueError("total_seed_cap must be an integer >= 1 when provided")

    eligible: list[tuple[int, int, str, int, int, int, CandidateFrame]] = []
    nomination_ranks: dict[str, int] = {}
    for candidate in candidates:
        metadata = dict(candidate.diagnostic_metadata or {})
        local_rank = metadata.get("local_anchor_rank")
        nomination_rank = metadata.get("video_nomination_rank")
        if type(local_rank) is not int or local_rank < 1:
            raise ValueError("QA temporal seed requires a valid local_anchor_rank")
        if type(nomination_rank) is not int or nomination_rank < 1:
            raise ValueError(
                "QA temporal seed requires a valid video_nomination_rank"
            )
        previous = nomination_ranks.setdefault(candidate.video_id, nomination_rank)
        if previous != nomination_rank:
            raise ValueError("video_nomination_rank must be stable within one video")
        if local_rank > anchors_per_video:
            continue
        eligible.append(
            (
                local_rank,
                nomination_rank,
                candidate.video_id,
                candidate.frame_id,
                candidate.clip_row,
                candidate.rank,
                candidate,
            )
        )

    selected_video_ids = {
        video_id
        for video_id, _rank in sorted(
            nomination_ranks.items(), key=lambda item: (item[1], item[0])
        )[:video_cap]
    }
    eligible.sort(key=lambda item: item[:-1])
    selected = [item[-1] for item in eligible if item[2] in selected_video_ids]
    if total_seed_cap is not None:
        selected = selected[:total_seed_cap]
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class QAVariantVideoRank:
    variant_id: str
    weight: float
    video_rank: int
    best_frame_id: int
    best_frame_cosine: float


@dataclass(frozen=True, slots=True)
class QAVideoNomination:
    video_id: str
    nomination_rank: int
    video_rrf_score: float
    best_individual_variant_rank: int
    per_variant: tuple[QAVariantVideoRank, ...]


def nominate_qa_videos(
    *,
    variants: Sequence[QueryVariant],
    maxima: FullCorpusVideoMaximaOutcome,
    config: QAVideoConditionedEvidenceConfig,
) -> tuple[QAVideoNomination, ...]:
    """Fuse per-variant video ranks without mixing raw cross-language cosine."""

    resolved_variants = tuple(variants)
    if not resolved_variants:
        raise ValueError("at least one localization variant is required")
    expected_ids = {variant.variant_id for variant in resolved_variants}
    if not expected_ids.issubset(maxima.rankings):
        missing = sorted(expected_ids - set(maxima.rankings))
        raise ValueError(f"missing video-maxima rankings: {missing}")

    hits_by_variant = {
        variant.variant_id: {hit.video_id: hit for hit in maxima.rankings[variant.variant_id]}
        for variant in resolved_variants
    }
    video_sets = [set(hits) for hits in hits_by_variant.values()]
    if not video_sets or any(video_ids != video_sets[0] for video_ids in video_sets[1:]):
        raise ValueError("localization variant rankings must cover the same corpus videos")

    unranked: list[QAVideoNomination] = []
    for video_id in sorted(video_sets[0]):
        provenance = tuple(
            QAVariantVideoRank(
                variant_id=variant.variant_id,
                weight=float(variant.weight),
                video_rank=hits_by_variant[variant.variant_id][video_id].rank,
                best_frame_id=hits_by_variant[variant.variant_id][video_id].frame_id,
                best_frame_cosine=float(
                    hits_by_variant[variant.variant_id][video_id].cosine_score
                ),
            )
            for variant in sorted(resolved_variants, key=lambda item: item.variant_id)
        )
        score = sum(
            item.weight / (config.video_rrf_constant + item.video_rank)
            for item in provenance
        )
        unranked.append(
            QAVideoNomination(
                video_id=video_id,
                nomination_rank=0,
                video_rrf_score=float(score),
                best_individual_variant_rank=min(
                    item.video_rank for item in provenance
                ),
                per_variant=provenance,
            )
        )

    ordered = sorted(
        unranked,
        key=lambda item: (
            -item.video_rrf_score,
            item.best_individual_variant_rank,
            item.video_id,
        ),
    )[: config.selected_video_cap]
    return tuple(
        QAVideoNomination(
            video_id=item.video_id,
            nomination_rank=rank,
            video_rrf_score=item.video_rrf_score,
            best_individual_variant_rank=item.best_individual_variant_rank,
            per_variant=item.per_variant,
        )
        for rank, item in enumerate(ordered, start=1)
    )


def _candidate_from_restricted_hit(
    *,
    hit: RestrictedFrameHit,
    variant: QueryVariant,
) -> CandidateFrame:
    return CandidateFrame(
        video_id=hit.video_id,
        frame_id=hit.frame_id,
        clip_row=hit.clip_row,
        keyframe_order=hit.keyframe_order,
        score=float(hit.cosine_score),
        rank=hit.rank,
        source="qa_video_restricted_exact",
        diagnostic_metadata={
            "pts_time": hit.pts_time,
            "variant_hit_count": 1,
            "best_individual_rank": hit.rank,
            "per_variant": [
                {
                    "variant_id": variant.variant_id,
                    "language": variant.language.value,
                    "variant_type": variant.variant_type.value,
                    "weight": float(variant.weight),
                    "rank": hit.rank,
                    "cosine_score": float(hit.cosine_score),
                }
            ],
        },
    )


def build_qa_grounding_result(
    *,
    query_id: str,
    variants: Sequence[QueryVariant],
    nominations: Sequence[QAVideoNomination],
    restricted: VideoRestrictedSearchOutcome,
    weighted_rrf: WeightedRRFRetriever,
    config: QAVideoConditionedEvidenceConfig,
    output_top_k: int,
) -> KISResult:
    """Build rank-slot evidence candidates diversified across nominated videos."""

    resolved_variants = tuple(variants)
    resolved_nominations = tuple(nominations)
    if not query_id.strip():
        raise ValueError("query_id must not be empty")
    if not resolved_variants:
        raise ValueError("at least one localization variant is required")
    if not 1 <= output_top_k <= 100:
        raise ValueError("output_top_k must be in [1, 100]")
    if len({item.video_id for item in resolved_nominations}) != len(
        resolved_nominations
    ):
        raise ValueError("nominated video_id values must be unique")
    if (
        config.candidate_ordering_policy
        == QA_CANDIDATE_ORDER_GLOBAL_RESTRICTED_COSINE
        and len(resolved_variants) != 1
    ):
        raise ValueError(
            "global_restricted_cosine ordering requires exactly one localization variant"
        )

    staged: list[tuple[int, int, CandidateFrame, QAVideoNomination]] = []
    per_video_cap = min(config.anchors_per_video, output_top_k)
    if (
        config.candidate_ordering_policy
        == QA_CANDIDATE_ORDER_GLOBAL_RESTRICTED_COSINE
        and config.preserve_keyframe_evidence
    ):
        # Global ordering must rank the same bounded candidate set that the
        # downstream keyframe evidence bank can admit.  Ranking deeper local
        # anchors first and filtering them later creates sparse output ranks
        # and can reduce an otherwise complete 32 x 3 evidence bank to fewer
        # than 96 records.
        per_video_cap = min(
            per_video_cap,
            config.keyframe_evidence_anchors_per_video,
        )
    for nomination in resolved_nominations:
        if len(resolved_variants) == 1:
            variant = resolved_variants[0]
            hits = restricted.rankings[variant.variant_id][nomination.video_id]
            local_candidates = tuple(
                _candidate_from_restricted_hit(hit=hit, variant=variant)
                for hit in hits[:per_video_cap]
            )
        else:
            rankings: dict[str, KISResult] = {}
            for variant in resolved_variants:
                hits = restricted.rankings[variant.variant_id][nomination.video_id]
                rankings[variant.variant_id] = KISResult(
                    query_id=variant.variant_id,
                    ranked_candidates=tuple(
                        _candidate_from_restricted_hit(hit=hit, variant=variant)
                        for hit in hits
                    ),
                )
            fused = weighted_rrf.fuse_rankings(
                query_id=f"{query_id}::{nomination.video_id}",
                variants=resolved_variants,
                rankings=rankings,
                output_top_k=per_video_cap,
                rrf_constant=config.video_rrf_constant,
            )
            local_candidates = fused.ranked_candidates

        for local_candidate in local_candidates:
            staged.append(
                (
                    local_candidate.rank,
                    nomination.nomination_rank,
                    local_candidate,
                    nomination,
                )
            )

    if config.candidate_ordering_policy == QA_CANDIDATE_ORDER_ROUND_ROBIN:
        staged.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].video_id,
                item[2].frame_id,
                item[2].clip_row,
            )
        )
    else:
        staged.sort(
            key=lambda item: (
                -float(item[2].score),
                item[1],
                item[0],
                item[2].video_id,
                item[2].frame_id,
                item[2].clip_row,
            )
        )
    selected: list[CandidateFrame] = []
    seen: set[tuple[str, int]] = set()
    for local_rank, _video_rank, candidate, nomination in staged:
        identity = (candidate.video_id, candidate.frame_id)
        if identity in seen:
            continue
        seen.add(identity)
        metadata = dict(candidate.diagnostic_metadata or {})
        per_variant = tuple(metadata.get("per_variant", ()))
        metadata.update(
            {
                "grounding_policy": QA_VIDEO_CONDITIONED_EVIDENCE_V1,
                "video_nomination_rank": nomination.nomination_rank,
                "video_nomination_rrf_score": nomination.video_rrf_score,
                "video_best_individual_variant_rank": (
                    nomination.best_individual_variant_rank
                ),
                "local_anchor_rank": local_rank,
                "localization_score": float(candidate.score),
                "localization_score_kind": (
                    "restricted_cosine"
                    if len(resolved_variants) == 1
                    else "weighted_rrf"
                ),
                "candidate_ordering_policy": config.candidate_ordering_policy,
                "source_localization_variant_ids": [
                    str(item["variant_id"]) for item in per_variant
                ],
            }
        )
        selected.append(
            CandidateFrame(
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                clip_row=candidate.clip_row,
                keyframe_order=candidate.keyframe_order,
                score=float(candidate.score),
                rank=len(selected) + 1,
                source=QA_VIDEO_CONDITIONED_EVIDENCE_V1,
                diagnostic_metadata=metadata,
            )
        )
        if len(selected) >= output_top_k:
            break
    return KISResult(query_id=query_id, ranked_candidates=tuple(selected))


def nomination_diagnostics(
    nominations: Sequence[QAVideoNomination],
    *,
    anchor_counts: Mapping[str, int],
) -> list[dict[str, object]]:
    """Return a bounded JSON-safe summary without dumping corpus-wide rankings."""

    return [
        {
            "video_id": item.video_id,
            "video_nomination_rank": item.nomination_rank,
            "video_nomination_rrf_score": item.video_rrf_score,
            "best_individual_variant_rank": item.best_individual_variant_rank,
            "anchor_count": int(anchor_counts.get(item.video_id, 0)),
            "per_variant": [
                {
                    "variant_id": rank.variant_id,
                    "weight": rank.weight,
                    "video_rank": rank.video_rank,
                    "best_frame_id": rank.best_frame_id,
                    "best_frame_cosine": rank.best_frame_cosine,
                }
                for rank in item.per_variant
            ],
        }
        for item in nominations
    ]


def distill_qa_scene_prompt(text: str) -> str:
    """Distill an interrogative QA question into a descriptive visual scene query."""
    if not isinstance(text, str) or not text.strip():
        return ""
    if not re.search(
        r"\b(?:what|which|how|who|where|scene|trong canh|ban tin|mau gi|bao nhieu|la gi)\b|[?]",
        text,
        flags=re.IGNORECASE,
    ):
        return text

    cleaned = text.strip()
    match = re.search(
        r"^(?:In the scene with|In the scene where|In the scene,?)\s+(.*?),"
        r"\s*(?:what|which|how|who|where)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = re.sub(
            r"^(?:In the scene with|In the scene where|In the news report,?|"
            r"In the scene,?|In the frame,?)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:Trong cảnh(?: có)?|Trong bản tin(?: có)?|Trong đoạn clip(?: có)?)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What is the scene displayed on|What scene is displayed on|"
            r"What is displayed on|What is shown on)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^What\s+(crop|animal|tool|device|instrument)\s+is\s+the\s+(.*?)\s+(harvesting|using|riding|holding)\s*(.*?)\?*$",
            r"The \2 \3 \1 \4",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^What color is the\s+(.*?)\s+worn by\s+(?:the\s+)?(.*?)\?*$",
            r"The \2 wearing a \1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^What color is the\s+(female|male)?\s*presenter's\s+(outfit|shirt|suit)\?*$",
            r"The \1 presenter wearing a \2 in the studio",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What color is(?: the)?|Which color is(?: the)?|What color are(?: the)?)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What type of vehicles?|Which type of vehicles?|"
            r"What kind of vehicles?)\s+(?:mainly )?(?:make up|is|are)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What type of vehicle|Which type of vehicle|"
            r"What kind of vehicle)\s+(?:is|are)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What type of|Which type of|What kind of)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What animal|Which animal)\s+(?:does|is|was)\s+(?:the )?",
            "animal ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What crop|Which crop)\s+(?:is|was)\s+(?:the )?",
            "crop ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What device|Which device|What tool|Which tool|"
            r"What instrument)\s+(?:is|was)\s+(?:the )?",
            "tool ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What is the name of(?: the)?|What is the brand of(?: the)?|Which brand is)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What is the person|What is the man|What is the woman)\s+",
            "a person ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What is the|What are the|What was the|What were the)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What is|What are|What was|What were)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:What activity are the|What craft are the)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:How many)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(
        r"\s+(?:are in the frame|are in the studio|are clearly visible|"
        r"can be seen)\?*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[?!.,;:\'\"]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text.strip()
