"""Frozen contracts for the bounded BCF-1 late-fusion experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

EXPERIMENT = "TRIAGE_BCF1_PROTECTED_LATE_FUSION"
POLICY = "A0_TOP5_PROTECTED_EQUAL_RRF60_LATE_FUSION"
SCA1_ANCHOR_COMMIT = "c26d5b1f8ffe96557387ffdf2b8beb6d134bfefa"
PREPARATION_ZIP_SHA256 = "849111c17d24f9232acfe9aeb4b77dfbeba4c0d2b4261b092417e4146baed4e3"
A0_CROSS_SHA256 = "8a774e25aae0d4e23eafa905e468b25baeabc0b2ed74ba16491a1138b099ef9e"
S1_CROSS_SHA256 = "4b31244123c22c435bf945f09692050631e71fb0d4c939f17f8f8b98d4533ca8"
F1_CROSS_SHA256 = "801e9e4a8e33916cb0430c9c391694410972a84b212d0db949d63671be39e2dc"
INDEX_ZIP_SHA256 = "e62ab0e7a92a265794b98ed1a54b57fcbba7e513ef03e49f0e541c707eec01bd"
INDEX_FINGERPRINT = "59302ddb5fb8c4aaacc9d6945dd5e1f7c32705286f06fb0fd56493e177eaaaa3"
VECTOR_SHA256 = "9b7b5d157070a105ae9210420e29dfdba889184ec22e4579cbb2fba437eb9384"
NORM_SHA256 = "b1d3914de250273e6659a973e3be93ac5589f41b5a694e2f8804fec8c5520fd4"
INDEX_ROWS = 177_321
INDEX_SHAPE = [177_321, 768]
INDEX_DTYPE = "float16"

PREPARATION_MEMBER_HASHES = {
    "bcf1_preparation/BCF1_PROTOCOL.md": (
        "52557d88a12ffe178e8e56513fe4dd34263c0a0d1248188035c66d6bff94ef55"
    ),
    "bcf1_preparation/POST_GT_DESIGN_SANITY.json": (
        "20b10cafaa2a58ed9e0be66060a956eeb933ecc5330d575f7e814988a991ac39"
    ),
    "bcf1_preparation/README.md": (
        "a16193fa4f0f40f6a786849423cdb76b2f3e9e6882aa3d3d47638b4ef6871c45"
    ),
    "bcf1_preparation/decision_context.json": (
        "615ce72045438b7a433935e07f0e1b405b6466423fe53a42a5864c395f88e00a"
    ),
    "bcf1_preparation/f1_fusion_provenance.jsonl": (
        "0c7f0a05f01a0f0f0a53e79019c03b46177ed92ee9bcef845833e8981a13566d"
    ),
    "bcf1_preparation/notebook_template_contract.json": (
        "761671fa1da893e4cab222d8091132ea387c036ffcac8fdc0d368605d657eecb"
    ),
    "bcf1_preparation/pre_gt_predictions/a0_cross_g1.jsonl": A0_CROSS_SHA256,
    "bcf1_preparation/pre_gt_predictions/f1_cross_protected_rrf60.jsonl": (F1_CROSS_SHA256),
    "bcf1_preparation/pre_gt_predictions/s1_cross_g1.jsonl": S1_CROSS_SHA256,
    "bcf1_preparation/sca1_complementarity_summary.json": (
        "14e6701c0460ed1d88d492d55b670ba0accf43ee39240697e239cb3723a9a06e"
    ),
    "bcf1_preparation/sca1_index_manifest.json": (
        "956a8f6ef71fc3b46f89699dc91644fa9cc8cbdd9706f449d5284b0963d8a5fe"
    ),
    "bcf1_preparation/sca1_paired_score_delta.json": (
        "70608c304d86a2b62264a91ca5959e4daa41b1d8d636bd3274e81d27f1bd56d1"
    ),
}


@dataclass(frozen=True)
class BCF1Settings:
    """The single predeclared BCF-1 arm; every value is immutable."""

    policy: str = POLICY
    rrf_k: int = 60
    protected_prefix: int = 5
    max_predictions: int = 100
    score_space: str = "FINAL_PREDICTION_LISTS_ONLY"
    raw_score_fusion: bool = False
    weights: bool = False
    parameter_sweep: bool = False
    event_graph: bool = False
    vlm: bool = False
    m1: bool = False
    production_policy_changed: bool = False
    automatic_production_promotion: bool = False

    def __post_init__(self) -> None:
        expected = (
            self.policy == POLICY
            and self.rrf_k == 60
            and self.protected_prefix == 5
            and self.max_predictions == 100
            and self.score_space == "FINAL_PREDICTION_LISTS_ONLY"
        )
        forbidden = any(
            (
                self.raw_score_fusion,
                self.weights,
                self.parameter_sweep,
                self.event_graph,
                self.vlm,
                self.m1,
                self.production_policy_changed,
                self.automatic_production_promotion,
            )
        )
        if not expected or forbidden:
            raise ValueError("BCF-1 settings are frozen and diagnostic-only")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "A0_CROSS_SHA256",
    "BCF1Settings",
    "EXPERIMENT",
    "F1_CROSS_SHA256",
    "INDEX_DTYPE",
    "INDEX_FINGERPRINT",
    "INDEX_ROWS",
    "INDEX_SHAPE",
    "INDEX_ZIP_SHA256",
    "NORM_SHA256",
    "POLICY",
    "PREPARATION_MEMBER_HASHES",
    "PREPARATION_ZIP_SHA256",
    "S1_CROSS_SHA256",
    "SCA1_ANCHOR_COMMIT",
    "VECTOR_SHA256",
]
