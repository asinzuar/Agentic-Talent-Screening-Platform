"""E15c: the local embedding baseline. Offline tests only: a tiny fake encoder, vectors in a temp store, no download."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_e14 import FREE, TextMatcher, FakeLLM, tiny  # noqa: E402

from agentic_talent_screener.evaluation import embedding, report
from agentic_talent_screener.evaluation.jds import synthetic_jds
from agentic_talent_screener.evaluation.runner import execute
from agentic_talent_screener.evaluation.store import EvalStore, StoreRefused
from agentic_talent_screener.evaluation.systems import (
    EmbeddingBaseline, EmbeddingNotCached, EvalCandidate, jd_query,
)

DIM = 16


class FakeEncoder:
    """Hashed bag of words into 16 dims, L2-normalised. Counts calls so tests can prove nothing is recomputed."""

    model_id = "fake/test-encoder@0"

    def __init__(self) -> None:
        self.texts_encoded = 0

    def encode(self, texts):
        self.texts_encoded += len(texts)
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in t.lower().split():
                v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(embedding.l2_normalise(v))
        return out


@pytest.fixture
def store(tmp_path):
    with EvalStore.for_root(tmp_path) as s:
        yield s


JD = synthetic_jds(1, 0)[0]
CANDS = [EvalCandidate("a", text=jd_query(JD)), EvalCandidate("b", text="gardening poetry sailing")]


def test_cosine_scores_and_vectors_are_stored(store):
    enc = FakeEncoder()
    out = EmbeddingBaseline(store, enc).score(CANDS, JD)
    assert out.scores["a"] == pytest.approx(1.0) and out.scores["b"] < out.scores["a"]
    assert store.get_embedding(enc.model_id, embedding.text_sha256("gardening poetry sailing")) is not None
    assert len(store.get_embedding(enc.model_id, embedding.text_sha256(CANDS[1].text))) == DIM


def test_second_run_recomputes_nothing_and_matches(store):
    enc = FakeEncoder()
    first = EmbeddingBaseline(store, enc).score(CANDS, JD)
    n = enc.texts_encoded
    second = EmbeddingBaseline(store, enc).score(CANDS, JD)
    assert enc.texts_encoded == n and second.scores == first.scores


def test_offline_with_cached_vectors_works_without_any_encoder(store):
    first = EmbeddingBaseline(store, FakeEncoder()).score(CANDS, JD)
    off = EmbeddingBaseline(store, None, offline=True)
    off._encoder = None
    # the model id of an encoder-less baseline is the real one, so seed under it to prove the lookup path
    real = embedding.MODEL_ID
    for c in (*CANDS,):
        store.put_embedding(real, embedding.text_sha256(c.text), store.get_embedding(FakeEncoder.model_id, embedding.text_sha256(c.text)), synthetic=True)
    store.put_embedding(real, embedding.text_sha256(jd_query(JD)), store.get_embedding(FakeEncoder.model_id, embedding.text_sha256(jd_query(JD))), synthetic=True)
    assert off.score(CANDS, JD).scores == pytest.approx(first.scores)


def test_offline_missing_vector_is_not_cached_not_zero(store):
    with pytest.raises(EmbeddingNotCached, match="embedding not cached"):
        EmbeddingBaseline(store, FakeEncoder(), offline=True).score(CANDS, JD)


def test_store_refuses_non_synthetic_vectors(store):
    with pytest.raises(StoreRefused):
        EmbeddingBaseline(store, FakeEncoder(), synthetic=False).score(CANDS, JD)


def test_model_id_changes_with_the_revision_so_vectors_are_not_reused(store):
    a, b = FakeEncoder(), FakeEncoder()
    b.model_id = "fake/test-encoder@1"
    EmbeddingBaseline(store, a).score(CANDS, JD)
    EmbeddingBaseline(store, b).score(CANDS, JD)
    assert b.texts_encoded == a.texts_encoded


def test_helpers():
    assert embedding.l2_normalise([0.0, 0.0]) == [0.0, 0.0]
    assert math.isclose(sum(x * x for x in embedding.l2_normalise([3.0, 4.0])), 1.0)
    assert len(embedding.MODEL_REVISION) == 40 and embedding.MODEL_ID.endswith(embedding.MODEL_REVISION)
    assert not embedding.model_cached("no/such/cache/dir")


def test_hf_home_env_wins_and_default_is_git_ignored_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "x"))
    assert embedding.hf_home() == tmp_path / "x"
    monkeypatch.delenv("HF_HOME")
    assert embedding.hf_home(tmp_path) == tmp_path / ".tmp" / "hf"


def test_ranking_and_robustness_comparator_in_a_tiny_run(tmp_path):
    enc = FakeEncoder()
    d = execute(tiny(systems=("keyword_ats", "embedding", "full_engine")), base=tmp_path, run_id="e1", call=FakeLLM(),
                model=FREE, matcher=TextMatcher(), encoder=enc)
    r = json.loads((d / "results.json").read_text())
    m = r["metrics"]
    assert r["systems"]["embedding"]["status"] == "ok"
    assert "ranking.lift_ndcg10_pct.full_engine.vs.embedding" in m
    assert "ranking.lift_ndcg10_pct.full_engine.vs.keyword_ats" in m
    assert m["robustness.asr.embedding"]["target"] is None and "comparator" in m["robustness.asr.embedding"]["note"]
    assert "robustness.rank_shift_mean.embedding" in m
    row = {x["layer"]: x for x in r["slide9"]}["Matching & ranking"]
    assert "embedding baseline: not" not in row["measured"]
    assert {c["name"] for c in row["components"]} >= {"vs embedding", "vs keyword_ats"}
    # re-run offline: every vector comes from the store, the encoder is never touched
    n = enc.texts_encoded
    d2 = execute(tiny(systems=("embedding",), offline=True), base=tmp_path, run_id="e2", model=FREE, encoder=enc)
    r2 = json.loads((d2 / "results.json").read_text())
    assert r2["systems"]["embedding"]["status"] == "ok" and enc.texts_encoded == n


def test_offline_run_without_cached_vectors_is_not_run_with_reason(tmp_path):
    d = execute(tiny(systems=("keyword_ats", "embedding", "full_engine"), offline=True), base=tmp_path, run_id="e3",
                model=FREE, matcher=TextMatcher(), encoder=FakeEncoder())
    r = json.loads((d / "results.json").read_text())
    rec = r["systems"]["embedding"]
    assert rec["status"] == "not_run" and rec["reason"] == "embedding not cached"
    assert rec["ranking"] is None
    row = {x["layer"]: x for x in r["slide9"]}["Matching & ranking"]
    assert "embedding baseline: not run" in row["measured"]


def test_lift_cannot_pass_without_the_embedding_comparison():
    raw = {"systems": {"embedding": {"reason": "embedding not cached"}}}
    assert report._embedding_reason(raw) == "embedding not cached"
    assert report._embedding_reason({"systems": {}}) == "not selected"


@pytest.mark.skipif(not embedding.model_cached(), reason="MiniLM not in the local HF cache; never downloaded by tests")
def test_real_model_integration(tmp_path):
    enc = embedding.HFEncoder(local_only=True)
    v = enc.encode(["python developer with sql", "python developer with sql", "pastry chef"])
    assert len(v[0]) == 384 and v[0] == pytest.approx(v[1], abs=1e-5)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(v[0], v[1]) > dot(v[0], v[2]) and math.isclose(dot(v[0], v[0]), 1.0, abs_tol=1e-4)


def test_r2_slide9_wording_reflects_the_real_embedding_state():
    n = report._embedding_note
    assert "not selected" in n({"systems": {}})
    assert "not run (embedding not cached)" in n({"systems": {"embedding": {"status": "not_run", "reason": "embedding not cached"}}})
    ran = n({"systems": {"embedding": {"status": "ok", "ranking": {"x": 1}}}})
    assert "ran, but there is no system to compare" in ran and "not run" not in ran


def test_r2_row_says_ran_when_embedding_ran_without_full_engine(tmp_path):
    d = execute(tiny(systems=("keyword_ats", "embedding")), base=tmp_path, run_id="r2", model=FREE, encoder=FakeEncoder())
    r = json.loads((d / "results.json").read_text())
    row = {x["layer"]: x for x in r["slide9"]}["Matching & ranking"]
    assert "ran, but there is no system to compare" in row["measured"] and "not run" not in row["measured"]
    assert any("256 tokens" in c["text"] and "lower bound" in c["text"] for c in r["caveats"])


def test_r3_missing_or_wrong_length_vector_is_an_infrastructure_error(store):
    from agentic_talent_screener.evaluation.systems import InfrastructureError

    class Short(FakeEncoder):
        def encode(self, texts):
            return [v[:-1] if t == CANDS[1].text else v for t, v in zip(texts, super().encode(texts))]

    with pytest.raises(InfrastructureError, match="wrong length"):
        EmbeddingBaseline(store, Short()).score(CANDS, JD)

    class Drops(FakeEncoder):
        def encode(self, texts):
            return [[0.0] * 0 if t == CANDS[1].text else v for t, v in zip(texts, super().encode(texts))]

    with pytest.raises((InfrastructureError, ValueError)):
        EmbeddingBaseline(store, Drops()).score(CANDS, JD)


def test_loader_refuses_pickle_weights():
    import inspect
    assert "use_safetensors" in inspect.getsource(embedding.HFEncoder._load)
