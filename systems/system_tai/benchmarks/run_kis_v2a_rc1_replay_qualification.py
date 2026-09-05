#!/usr/bin/env python3
"""
Official In-Tree Qualification Runner for Profile `kis-v2a-rc1-replay`.

This runner implements the strict qualification gate:
1. Fail-closed repository clean worktree verification (0 untracked / modified / deleted files).
2. Pinned OpenAI CLIP commit d05afc43... and official ViT-B-32 checkpoint SHA256.
3. Focused Pytest qualification suite (22 tests in test_kis_v2a_rc1_profiles.py and test_kis_v2a_rc1_e2e_closure.py).
4. Comprehensive CLI flag conflict rejection gate (13 non-default and dormant flag combinations).
5. Two independent official CLI passes (pass_1 and pass_2) with session_manifest.json validation.
6. Bit-exact Top-100 projection digests and ground-truth target coarse/final ranks.
7. Two-pass bit-exact invariance across projections and selected-video sequences.
8. Historical selected-sequence cross-audit (authentically verified against d3b2507 or edfebed; never self-generated).
9. Fail-fast isolation gates (request parameter tampering and unseen query sidecar miss).
10. Qualification manifest provenance chain verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

if not __debug__:
    raise RuntimeError("Qualification runner cannot run with Python optimization (-O) enabled")


def paths_overlap(a: Path, b: Path) -> bool:
    resolved_a = a.resolve()
    resolved_b = b.resolve()
    return resolved_a == resolved_b or resolved_a in resolved_b.parents or resolved_b in resolved_a.parents


EXPECTED_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
EXPECTED_CLIP_CHECKPOINT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

CANONICAL_PORTABLE_CORPUS_FINGERPRINT = "b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4"
CANONICAL_ABSOLUTE_CORPUS_FINGERPRINT = "398bb60c6ea1c8ebbd787c801836ef96a8398795b61fd6808e996f4ef19c0fa2"
ALLOWED_CORPUS_FINGERPRINTS = (
    CANONICAL_PORTABLE_CORPUS_FINGERPRINT,
    CANONICAL_ABSOLUTE_CORPUS_FINGERPRINT,
)
HISTORICAL_RC1_COMMIT = "d3b2507b97af03ae9e7067b97e79ecc8488f551c"
HISTORICAL_REPLAY_TAG_COMMIT = "edfebede48f437479dfb03c7131aae64d863b240"
HISTORICAL_VALID_COMMITS = (HISTORICAL_RC1_COMMIT, HISTORICAL_REPLAY_TAG_COMMIT)

GOLDEN_DIGESTS = {
    "query-p1-1-kis": "1ec8d8c03122de1a9e3083a7addadcbcb4f845b2c18deec16d978baf72e19c3e",
    "query-p1-2-kis": "47a486ec387785ef95552642df7ed05542f2bfe80d37004dcc4a7287aca3219e",
    "query-p1-4-kis": "2d7f5ebacb8f040ed1c252feecfcbdee59e44550101a7754c76ebc1935f9770e",
    "query-p1-5-kis": "0eccbb6d600bb2945cd80fe0126e5a2a5106b414da171ce79fd87ca487e3cb62",
    "query-p1-6-kis": "9d4cd4ef703a106b515f087413d79a3755051aceaaf7be7de460511c9dfda6fb",
}

FROZEN_SELECTED_SEQUENCE_DIGESTS = {
    "query-p1-1-kis": "acf04f853f8907081c4a72db3cfa994e40d371783daf44984b65e46a07d9567b",
    "query-p1-2-kis": "86f875cfd66ecc13007ae580912768e42da8372470335d3fd9599d65281a558c",
    "query-p1-4-kis": "393c06fb91975e473bd2015aeeb6089861404cb67f46d8b3aa04db5a7f01b719",
    "query-p1-5-kis": "80eb8ad5c38b221160031ef26344a65eea1ab1b1842acba8317a46c03cacd770",
    "query-p1-6-kis": "193ca3cc581d484cc4aa121b7dd7f1008727bab9eef7c74fd2656cc71d6451ef",
}

EXPECTED_TARGET_COARSE_RANKS = {
    "query-p1-1-kis": 1,
    "query-p1-2-kis": 25,
    "query-p1-4-kis": 1,
    "query-p1-5-kis": 31,
    "query-p1-6-kis": 19,
}

EXPECTED_TARGET_RANKS = {
    "query-p1-1-kis": 1,
    "query-p1-2-kis": 2,
    "query-p1-4-kis": 1,
    "query-p1-5-kis": 25,
    "query-p1-6-kis": 1,
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def get_installed_clip_commit_info() -> str | None:
    for base_p in [Path(sys.prefix), Path("/usr/local")]:
        for dist_info in base_p.glob("**/clip-*.dist-info"):
            d_url = dist_info / "direct_url.json"
            if d_url.is_file():
                try:
                    info = json.loads(d_url.read_text(encoding="utf-8"))
                    return info.get("vcs_info", {}).get("commit_id")
                except Exception:
                    pass
    return None


def run_qualification(
    *,
    repo_root: Path,
    expected_commit: str,
    input_root: Path,
    manifest_cache: Path,
    output_root: Path,
    historical_manifest: Path | None = None,
    historical_manifest_sha256: str | None = None,
    skip_pip_install: bool = False,
) -> int:
    if (historical_manifest is None) != (historical_manifest_sha256 is None):
        raise ValueError(
            "--historical-manifest and --historical-manifest-sha256 must be provided together"
        )

    print("=" * 110)
    print("🔒 KIS V2-A.3 OFFICIAL CLI QUALIFICATION GATE (PROFILE: kis-v2a-rc1-replay)")
    print("=" * 110)

    # 1. Verify Git Commit & Worktree Cleanliness (Fail-Closed)
    active_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    print(f"Active Commit: {active_commit}")
    assert active_commit == expected_commit, (
        f"Fail-closed: Commit mismatch! Expected {expected_commit}, got {active_commit}"
    )

    status_out = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    assert status_out == "", f"Fail-closed: repository is not clean:\n{status_out}"
    print("✅ Repository worktree is 100% clean (0 untracked / modified / deleted files)")

    # 2. Configure Environment & Paths
    sys_tai_src = repo_root / "systems" / "system_tai" / "src"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(sys_tai_src))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{sys_tai_src}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # 3. Verify OpenAI CLIP Dependencies & Commit
    if not skip_pip_install:
        print("\nEnsuring OpenAI CLIP runtime dependencies (ftfy, regex, tqdm, build) are installed...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy", "regex", "tqdm", "build"], check=True)

    obs_commit = get_installed_clip_commit_info()
    if obs_commit != EXPECTED_CLIP_COMMIT and not skip_pip_install:
        print(f"Installing OpenAI CLIP at commit {EXPECTED_CLIP_COMMIT[:8]}...")
        clip_git_url = f"git+https://github.com/openai/CLIP.git@{EXPECTED_CLIP_COMMIT}"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", clip_git_url], check=True)
        obs_commit = get_installed_clip_commit_info()

    assert obs_commit == EXPECTED_CLIP_COMMIT, (
        f"Fail-closed: Observed CLIP commit mismatch! Expected {EXPECTED_CLIP_COMMIT}, got {obs_commit}"
    )
    print(f"✅ OpenAI CLIP commit verified via direct_url.json: {obs_commit}")

    # Checkpoint verification
    clip_cache_dir = Path.home() / ".cache" / "clip"
    clip_cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = clip_cache_dir / "ViT-B-32.pt"
    if not ckpt_file.is_file() or sha256_file(ckpt_file).lower() != EXPECTED_CLIP_CHECKPOINT_SHA256.lower():
        print(f"Pre-provisioning official CLIP ViT-B-32 checkpoint to {ckpt_file}...")
        ckpt_url = f"https://openaipublic.azureedge.net/clip/models/{EXPECTED_CLIP_CHECKPOINT_SHA256}/ViT-B-32.pt"
        urllib.request.urlretrieve(ckpt_url, ckpt_file)

    actual_ckpt_sha = sha256_file(ckpt_file)
    assert actual_ckpt_sha.lower() == EXPECTED_CLIP_CHECKPOINT_SHA256.lower(), f"Checkpoint SHA mismatch: {actual_ckpt_sha}"
    print(f"✅ Official CLIP ViT-B-32 checkpoint verified: {actual_ckpt_sha}")

    # 4. Run Pytest Focused Suite (22 tests)
    print("\nRunning Pytest Profile & Closure Suites (22 tests)...")
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        str(repo_root / "systems" / "system_tai" / "tests" / "test_kis_v2a_rc1_profiles.py"),
        str(repo_root / "systems" / "system_tai" / "tests" / "test_kis_v2a_rc1_e2e_closure.py"),
        "-v",
    ]
    subprocess.run(pytest_cmd, cwd=repo_root / "systems" / "system_tai", env=env, check=True)
    print("✅ 22/22 focused qualification tests passed cleanly!")

    # 5. Load Benchmarks & Manifests
    from system_tai.retrieval.canonical_projection import canonical_projection_digest

    stress_manifest_path = repo_root / "systems" / "system_tai" / "benchmarks" / "frozen_kis_v2a_stress_manifest.json"
    assert stress_manifest_path.is_file(), f"Missing stress manifest at {stress_manifest_path}"
    stress_manifest = json.loads(stress_manifest_path.read_text(encoding="utf-8"))

    manual_ref_path = repo_root / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json"
    assert manual_ref_path.is_file(), f"Missing manual reference at {manual_ref_path}"
    manual_ref = json.loads(manual_ref_path.read_text(encoding="utf-8"))

    target_videos = {q["query_id"]: q["human_verified_video_id"] for q in manual_ref["queries"]}
    print(f"Loaded ground truth target videos: {target_videos}")

    benchmark_requests = [
        {
            "type": "query",
            "request_id": f"req-{q['query_id']}",
            "query_id": q["query_id"],
            "query_vi": q["query_vi"],
        }
        for q in stress_manifest["queries"]
    ]
    benchmark_requests.append({"type": "shutdown", "request_id": "req-shutdown"})

    # 6. Comprehensive Fail-Fast CLI Rejection Gate (13 test cases)
    print("\n--- Testing Comprehensive Fail-Fast CLI Flag Rejection Gate ---")
    cli_rejection_cases = [
        (["--window-before-seconds", "99"], "strictly requires --window-before-seconds 5.0"),
        (["--coarse-stride-frames", "999"], "strictly requires --coarse-stride-frames 15"),
        (["--image-batch-size", "7"], "strictly requires --image-batch-size 32"),
        (["--coarse-decode-strategy", "sparse-verified"], "strictly requires --coarse-decode-strategy sequential"),
        (["--continue-on-request-error"], "strictly requires fail-fast protocol"),
        (["--translation-allow-model-download"], "strictly requires translation_allow_model_download=False"),
        (["--translation-device", "cpu"], "strictly requires translation_device 'auto'"),
        (["--translation-device", "cuda"], "strictly requires translation_device 'auto'"),
        (["--translation-cache-dir", "/tmp/cache"], "does not allow specifying translation_cache_dir"),
        (["--kis-visual-verifier-allow-model-download"], "strictly requires kis_visual_verifier_allow_model_download=False"),
        (["--kis-anchor-video-rank-cap", "99"], "strictly requires default --kis-anchor-video-rank-cap 20"),
        (["--kis-timeline-max-videos", "99"], "strictly requires default --kis-timeline-max-videos 3"),
        (["--kis-visual-verifier-shortlist-per-video", "99"], "strictly requires default --kis-visual-verifier-shortlist-per-video 32"),
    ]

    rejection_test_dir = output_root / "flag_rejection_test"
    rejection_test_dir.mkdir(parents=True, exist_ok=True)
    dummy_req = rejection_test_dir / "dummy_req.jsonl"
    dummy_req.write_text(json.dumps({"type": "shutdown", "request_id": "req-sd"}) + "\n", encoding="utf-8")

    for flags, expected_err in cli_rejection_cases:
        cmd = [
            sys.executable, "-m", "system_tai.kis.session",
            "--profile", "kis-v2a-rc1-replay",
            "--input-root", str(input_root),
            "--manifest-cache", str(manifest_cache),
            "--output-root", str(rejection_test_dir),
        ] + flags
        with open(dummy_req, "r", encoding="utf-8") as in_f:
            res_test = subprocess.run(cmd, stdin=in_f, capture_output=True, text=True, cwd=repo_root, env=env)
        assert res_test.returncode != 0, f"Expected non-zero exit code for flags {flags}"
        combined_msg = (res_test.stdout or "") + (res_test.stderr or "")
        assert expected_err in combined_msg, f"Expected '{expected_err}' in output for flags {flags}, got:\n{combined_msg}"
    print(f"✅ All {len(cli_rejection_cases)} CLI conflict flag combinations strictly rejected fail-fast!")

    # 7. Execute Two Independent Official CLI Passes (with cwd=repo_root)
    pass_results = {}

    for pass_idx in (1, 2):
        pass_name = f"pass_{pass_idx}"
        out_dir = output_root / pass_name
        out_dir.mkdir(parents=True, exist_ok=True)
        requests_file = out_dir / "requests.jsonl"
        with open(requests_file, "w", encoding="utf-8") as f:
            for req in benchmark_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        print(f"\n--- Launching Official CLI Session [{pass_name}] ---")
        cli_cmd = [
            sys.executable, "-m", "system_tai.kis.session",
            "--profile", "kis-v2a-rc1-replay",
            "--input-root", str(input_root),
            "--manifest-cache", str(manifest_cache),
            "--output-root", str(out_dir),
        ]

        with open(requests_file, "r", encoding="utf-8") as in_f:
            res = subprocess.run(cli_cmd, stdin=in_f, capture_output=True, text=True, cwd=repo_root, env=env)

        if res.returncode != 0:
            print(f"CLI FAILED (returncode {res.returncode}):")
            print("STDOUT:\n", res.stdout)
            print("STDERR:\n", res.stderr)
            raise RuntimeError(f"CLI failed on {pass_name}")

        responses = [json.loads(line) for line in res.stdout.strip().split("\n") if line.strip()]
        query_resps = [r for r in responses if r.get("type") == "query_result"]
        assert len(query_resps) == 5, f"Expected 5 query_result lines, got {len(query_resps)}"

        # Verify session_manifest.json (Blocker 1 Fix: top-level fields)
        manifest_file = out_dir / "session_manifest.json"
        assert manifest_file.is_file(), f"Missing session_manifest.json in {out_dir}"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

        if expected_commit:
            assert manifest_data.get("git_commit_hash") == expected_commit, (
                f"Git commit hash mismatch: {manifest_data.get('git_commit_hash')} != {expected_commit}"
            )
        assert manifest_data.get("profile_name") == "kis-v2a-rc1-replay"
        assert manifest_data.get("translation_provider_mode") == "immutable_sidecar"
        assert manifest_data.get("model_provenance", {}).get("clip_source_commit") == EXPECTED_CLIP_COMMIT
        assert manifest_data.get("model_provenance", {}).get("checkpoint_sha256") == EXPECTED_CLIP_CHECKPOINT_SHA256
        assert manifest_data.get("model_provenance", {}).get("verified_bit_exact") is True
        assert manifest_data.get("manifest_fingerprint") in ALLOWED_CORPUS_FINGERPRINTS, (
            f"Manifest fingerprint mismatch: {manifest_data.get('manifest_fingerprint')} not in {ALLOWED_CORPUS_FINGERPRINTS}"
        )
        assert manifest_data.get("video_count") == 873, f"Expected video_count 873, got {manifest_data.get('video_count')}"
        assert manifest_data.get("feature_row_count") == 177321, f"Expected feature_row_count 177321, got {manifest_data.get('feature_row_count')}"

        pass_results[pass_name] = {}
        for r in query_resps:
            qid = r["query_id"]
            cand_rel = r["artifacts"]["candidates_json"]
            cand_path = out_dir / cand_rel
            assert cand_path.is_file(), f"Missing candidates.json for {qid}"
            cand_payload = json.loads(cand_path.read_text(encoding="utf-8"))
            records = cand_payload["records"]
            proj_digest = canonical_projection_digest(records)

            selected_vids = [
                row["video_id"]
                for row in cand_payload.get("video_first", {}).get("selected_videos", [])
                if isinstance(row, dict) and "video_id" in row
            ]
            selected_seq_digest = hashlib.sha256(json.dumps(selected_vids).encode("utf-8")).hexdigest()

            target_vid = target_videos[qid]

            # 1. Target Coarse Rank
            target_coarse_rank = None
            for idx, sv in enumerate(selected_vids, 1):
                if sv == target_vid:
                    target_coarse_rank = idx
                    break
            expected_coarse_rank = EXPECTED_TARGET_COARSE_RANKS[qid]
            assert target_coarse_rank == expected_coarse_rank, (
                f"Target coarse rank mismatch on {qid}: {target_coarse_rank} != {expected_coarse_rank} (Target: {target_vid})"
            )

            # 2. Target Final Rank
            target_final_rank = None
            for r_item in records:
                if r_item["video_id"] == target_vid:
                    target_final_rank = r_item["rank"]
                    break
            expected_final_rank = EXPECTED_TARGET_RANKS[qid]
            assert target_final_rank == expected_final_rank, (
                f"Target final rank mismatch on {qid}: {target_final_rank} != {expected_final_rank} (Target: {target_vid})"
            )

            # 3. Golden Projection Digest Match
            expected_digest = GOLDEN_DIGESTS[qid]
            assert proj_digest == expected_digest, f"Digest mismatch on {qid}: {proj_digest} != {expected_digest}"

            pass_results[pass_name][qid] = {
                "proj_digest": proj_digest,
                "selected_seq_digest": selected_seq_digest,
                "selected_vids": selected_vids,
                "target_coarse_rank": target_coarse_rank,
                "target_final_rank": target_final_rank,
            }
            print(f"  [{pass_name}] {qid} ({target_vid}): Coarse #{target_coarse_rank} | Final #{target_final_rank} | Proj {proj_digest[:16]}... [MATCH ✅]")

    # 8. Two-Pass Invariance Verification
    print("\nVerifying Bit-Exact Two-Pass Invariance (Pass 1 vs Pass 2)...")
    for qid in GOLDEN_DIGESTS:
        p1 = pass_results["pass_1"][qid]
        p2 = pass_results["pass_2"][qid]
        assert p1["proj_digest"] == p2["proj_digest"], f"Two-pass projection divergence on {qid}!"
        assert p1["selected_seq_digest"] == p2["selected_seq_digest"], f"Two-pass selected sequence divergence on {qid}!"
        assert p1["selected_vids"] == p2["selected_vids"], f"Two-pass video list divergence on {qid}!"
        assert p1["target_coarse_rank"] == p2["target_coarse_rank"], f"Two-pass coarse rank divergence on {qid}!"
        assert p1["target_final_rank"] == p2["target_final_rank"], f"Two-pass final rank divergence on {qid}!"
    print("Two-Pass Determinism (Top-100 & Selected-Video Sequence): BIT-EXACT 100% ✅")

    # 9. Historical Closure Cross-Audit
    # (Require explicit artifact path & SHA256; NO loose recursive filesystem globbing)
    historical_audit_status = "PENDING_EXTERNAL_HISTORICAL_ARTIFACT"
    historical_manifest_info = ""

    if historical_manifest is not None:
        assert historical_manifest_sha256 is not None, "historical_manifest_sha256 must be provided"
        if not historical_manifest.is_file():
            raise ValueError(
                f"Fail-closed: Provided historical manifest does not exist at {historical_manifest}"
            )
        actual_hist_sha = sha256_file(historical_manifest).lower()
        if actual_hist_sha != historical_manifest_sha256.lower():
            raise ValueError(
                f"Fail-closed: Historical manifest SHA256 mismatch!\n"
                f"Expected: {historical_manifest_sha256.lower()}\n"
                f"Actual:   {actual_hist_sha}"
            )
        print(f"✅ Historical manifest file SHA-256 verified: {actual_hist_sha}")

        h_data = json.loads(historical_manifest.read_text(encoding="utf-8"))
        assert h_data.get("release_candidate") == "KIS_V2A_RC1", (
            f"Historical manifest release_candidate is not KIS_V2A_RC1 (got {h_data.get('release_candidate')})"
        )
        assert h_data.get("release_qualified") is True, (
            "Historical manifest is not marked release_qualified=True"
        )
        hist_commit = h_data.get("git_commit_sha")
        assert hist_commit in HISTORICAL_VALID_COMMITS, (
            f"Fail-closed: Historical manifest commit mismatch: expected one of {HISTORICAL_VALID_COMMITS}, got {hist_commit}"
        )
        assert h_data.get("corpus", {}).get("fingerprint") in ALLOWED_CORPUS_FINGERPRINTS, (
            f"Historical manifest corpus fingerprint mismatch: {h_data.get('corpus', {}).get('fingerprint')} not in {ALLOWED_CORPUS_FINGERPRINTS}"
        )

        for qid in GOLDEN_DIGESTS:
            hist_q = h_data.get("queries", {}).get(qid, {})
            hist_seq_digest = hist_q.get("selected_sequence_digest")
            hist_proj_digest = hist_q.get("canonical_projection_digest")
            hist_coarse = hist_q.get("target_coarse_rank")
            hist_final = hist_q.get("target_rank_run_1")

            actual_seq_digest = pass_results["pass_1"][qid]["selected_seq_digest"]
            actual_proj_digest = pass_results["pass_1"][qid]["proj_digest"]
            actual_coarse = pass_results["pass_1"][qid]["target_coarse_rank"]
            actual_final = pass_results["pass_1"][qid]["target_final_rank"]

            assert actual_proj_digest == hist_proj_digest, f"Projection digest divergence vs RC1 closure on {qid}!"
            assert actual_seq_digest == hist_seq_digest, f"Selected sequence digest divergence vs RC1 closure on {qid}!"
            assert actual_coarse == hist_coarse, f"Target coarse rank divergence vs RC1 closure on {qid}!"
            assert actual_final == hist_final, f"Target final rank divergence vs RC1 closure on {qid}!"
            print(f"  [{qid}] Bit-exact match with historical closure ({hist_commit[:8]}): Proj={actual_proj_digest[:16]}... | Seq={actual_seq_digest[:16]}... | Coarse=#{actual_coarse} | Final=#{actual_final} ✅")

        historical_audit_status = "PASS"
        historical_manifest_info = f"commit={hist_commit[:8]}, sha256={actual_hist_sha[:16]}..."
        print(f"✅ Authentic historical closure manifest cross-audit: PASS ({historical_manifest_info})")
    else:
        print("\nℹ️ No external historical closure manifest (--historical-manifest) specified.")
        print("   Evaluating selected-sequence digests against in-tree canonical frozen fixtures...")
        mismatches = []
        for qid, expected_seq_digest in FROZEN_SELECTED_SEQUENCE_DIGESTS.items():
            actual_seq_digest = pass_results["pass_1"][qid]["selected_seq_digest"]
            if actual_seq_digest != expected_seq_digest:
                mismatches.append((qid, actual_seq_digest, expected_seq_digest))
            else:
                print(f"  [{qid}] Selected sequence matches frozen fixture: {actual_seq_digest[:16]}... ✅")
        if mismatches:
            for qid, actual_d, exp_d in mismatches:
                print(f"  ❌ Mismatch on {qid}: actual={actual_d} != expected={exp_d}")
            raise AssertionError(
                f"Selected sequence digest mismatch vs in-tree frozen fixture on {len(mismatches)} queries: "
                f"{[m[0] for m in mismatches]}"
            )
        print("   Audit status: In-tree frozen selected-sequence parity PASS (independent historical cross-audit status: PENDING)")

    # 10. Fail-Fast Isolation Gates: Request Tampering & Unseen Query
    print("\nTesting Request Parameter Tampering Rejection Gate (output_top_k=50)...")
    tamper_dir = output_root / "tamper_test"
    tamper_dir.mkdir(parents=True, exist_ok=True)
    tamper_reqs = tamper_dir / "tamper_reqs.jsonl"
    tamper_reqs.write_text(json.dumps({
        "type": "query",
        "request_id": "req-tampered",
        "query_id": "query-p1-1-kis",
        "query_vi": stress_manifest["queries"][0]["query_vi"],
        "output_top_k": 50,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    cli_tamper_cmd = [
        sys.executable, "-m", "system_tai.kis.session",
        "--profile", "kis-v2a-rc1-replay",
        "--input-root", str(input_root),
        "--manifest-cache", str(manifest_cache),
        "--output-root", str(tamper_dir),
    ]
    with open(tamper_reqs, "r", encoding="utf-8") as in_f:
        tamper_res = subprocess.run(cli_tamper_cmd, stdin=in_f, capture_output=True, text=True, cwd=repo_root, env=env)

    assert tamper_res.returncode == 1, f"Expected returncode 1 on tampered output_top_k, got {tamper_res.returncode}"
    assert "strictly requires request output_top_k=100" in (tamper_res.stdout + tamper_res.stderr)
    print("Request Parameter Tampering (output_top_k=50) Rejected Fail-Closed (Code 1) ✅")

    print("\nTesting Out-of-Benchmark Unseen Query Isolation...")
    unseen_dir = output_root / "unseen_query_test"
    unseen_dir.mkdir(parents=True, exist_ok=True)
    unseen_reqs = unseen_dir / "unseen_reqs.jsonl"
    unseen_reqs.write_text(json.dumps({
        "type": "query",
        "request_id": "req-unseen",
        "query_id": "query-novel-unseen",
        "query_vi": "Một cảnh quay hoàn toàn mới không có trong benchmark",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    cli_unseen_cmd = [
        sys.executable, "-m", "system_tai.kis.session",
        "--profile", "kis-v2a-rc1-replay",
        "--input-root", str(input_root),
        "--manifest-cache", str(manifest_cache),
        "--output-root", str(unseen_dir),
    ]
    with open(unseen_reqs, "r", encoding="utf-8") as in_f:
        unseen_res = subprocess.run(cli_unseen_cmd, stdin=in_f, capture_output=True, text=True, cwd=repo_root, env=env)

    assert unseen_res.returncode == 1, f"Expected returncode 1 on unseen query under fail-fast, got {unseen_res.returncode}"
    unseen_lines = [line for line in unseen_res.stdout.strip().split("\n") if "QUERY_EXECUTION_FAILED" in line]
    assert len(unseen_lines) >= 1, "Expected QUERY_EXECUTION_FAILED line in CLI stdout"
    unseen_err = json.loads(unseen_lines[0])
    assert unseen_err.get("error_type") == "TranslationError", f"Expected TranslationError, got {unseen_err.get('error_type')}"
    assert "Sidecar translation miss" in unseen_err.get("message", ""), f"Expected 'Sidecar translation miss' in message, got {unseen_err.get('message')}"
    assert "query-novel-unseen" in unseen_err.get("message", "")
    print("✅ Unseen Query Rejected Fail-Fast with Sidecar translation miss (TranslationError, Code 1)")

    # 11. In-Tree Qualification Manifest Provenance Verification
    in_tree_manifest_path = repo_root / "systems" / "system_tai" / "benchmarks" / "kis_v2a_rc1_replay_qualification_manifest.json"
    assert in_tree_manifest_path.is_file(), f"Missing qualification manifest at {in_tree_manifest_path}"
    in_tree_manifest = json.loads(in_tree_manifest_path.read_text(encoding="utf-8"))
    assert in_tree_manifest.get("profile_name") == "kis-v2a-rc1-replay"
    provenance_chain = in_tree_manifest.get("provenance_chain", {})
    assert "historical_baseline_run" in provenance_chain
    assert "replay_tag_v1" in provenance_chain
    assert "dormant_tuning_flags_hardening" in provenance_chain
    assert "translation_parameter_lock" in provenance_chain
    assert "comprehensive_fail_fast_head" in provenance_chain
    print("✅ In-tree qualification manifest provenance chain verified!")

    # 12. Concluding Calibrated Banner
    print("\n" + "=" * 110)
    print("🔒 KIS V2-A.3 REPLAY QUALIFICATION: FIVE-QUERY FROZEN REPLAY HARDENING PASS ✅")
    if historical_audit_status == "PASS":
        print(f"Audit Verdict: Hardened five-query replay qualified; historical cross-audit: PASS ({historical_manifest_info}).")
    else:
        print("Audit Verdict: Hardened five-query replay qualified; independent historical cross-audit: PENDING.")
    print("Next Milestones Pending: Live competition queries (kis-v2a-rc1-live), ground-truth interval evaluator (KISFixtureEvaluator), latency & RAM SLA benchmarking.")
    print("=" * 110)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Official In-Tree Qualification Runner for Profile kis-v2a-rc1-replay")
    parser.add_argument("--repo-root", default=None, help="Root path of the repository")
    parser.add_argument("--expected-commit", required=True, help="Expected Git commit SHA (fail-closed check)")
    parser.add_argument("--input-root", required=True, help="Path to corpus dataset root")
    parser.add_argument("--manifest-cache", default="/kaggle/working/kis_manifest_cache.json", help="Path to manifest cache file")
    parser.add_argument("--output-root", default="/kaggle/working/kis_v2a_rc1_cli_qualification", help="Output directory for qualification results")
    parser.add_argument("--historical-manifest", default=None, help="Path to authentic historical closure manifest")
    parser.add_argument("--historical-manifest-sha256", default=None, help="Expected SHA256 hex digest of the historical closure manifest")
    parser.add_argument("--skip-pip-install", action="store_true", help="Skip pip install of CLIP dependencies")

    args = parser.parse_args()

    if (args.historical_manifest is None) != (args.historical_manifest_sha256 is None):
        raise ValueError(
            "--historical-manifest and --historical-manifest-sha256 must be provided together"
        )

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent.parent.parent
    input_root = Path(args.input_root).resolve()
    manifest_cache = Path(args.manifest_cache).resolve()
    output_root = Path(args.output_root).resolve()
    historical_manifest = Path(args.historical_manifest).resolve() if args.historical_manifest else None

    # Strict deletion safety guards
    if output_root.parent == output_root:
        raise ValueError(f"Safety guard: output_root cannot be root filesystem ({output_root})")

    clean_out_str = output_root.as_posix().lower().rstrip("/")
    if (
        clean_out_str in ("/kaggle", "/kaggle/input", "/kaggle/working")
        or clean_out_str.endswith(":/kaggle")
        or clean_out_str.endswith(":/kaggle/input")
        or clean_out_str.endswith(":/kaggle/working")
    ):
        raise ValueError(
            f"Safety guard: output_root cannot be /kaggle, /kaggle/input, or /kaggle/working directly ({output_root})"
        )

    if paths_overlap(output_root, repo_root):
        raise ValueError(f"Safety guard: output_root overlaps repo_root ({repo_root})")
    if paths_overlap(output_root, input_root):
        raise ValueError(f"Safety guard: output_root overlaps input_root ({input_root})")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    return run_qualification(
        repo_root=repo_root,
        expected_commit=args.expected_commit,
        input_root=input_root,
        manifest_cache=manifest_cache,
        output_root=output_root,
        historical_manifest=historical_manifest,
        historical_manifest_sha256=args.historical_manifest_sha256,
        skip_pip_install=args.skip_pip_install,
    )


if __name__ == "__main__":
    sys.exit(main())
