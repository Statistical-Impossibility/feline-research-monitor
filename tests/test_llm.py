"""Tests for model-chain spec parsing and cross-model fallback (no ADK needed)."""

import asyncio

import pytest

import feline_monitor.llm as llm
from feline_monitor.config import Model


def test_model_specs_single():
    m = Model(provider="lmstudio", model_id="x")
    assert llm.model_specs(m) == [("lmstudio", "x")]


def test_model_specs_chain_overrides_single():
    m = Model(
        provider="lmstudio",
        model_id="x",
        chain=[
            {"provider": "openrouter", "model_id": "a"},
            {"provider": "lmstudio", "model_id": "b"},
        ],
    )
    assert llm.model_specs(m) == [("openrouter", "a"), ("lmstudio", "b")]


def test_run_with_fallback_returns_first_success(monkeypatch):
    seen = []

    def fake(agent, text, timeout_s=None):
        seen.append(agent)
        if agent == "bad":
            raise ConnectionError("backend down")
        return "OK"

    monkeypatch.setattr(llm, "run_agent", fake)
    text, label = llm.run_with_fallback(["bad", "good"], "prompt", labels=["b", "g"])
    assert text == "OK"
    assert label == "g"  # reports which model actually answered
    assert seen == ["bad", "good"]  # tried bad, fell through to good


def test_run_with_fallback_all_fail_reraises_last(monkeypatch):
    def fake(agent, text, timeout_s=None):
        raise ConnectionError("down")

    monkeypatch.setattr(llm, "run_agent", fake)
    with pytest.raises(ConnectionError):
        llm.run_with_fallback(["a", "b"], "prompt")


class _FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_unload_lmstudio_posts_correct_url_and_body(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.update(url=url, json=json)
        return _FakeResp(200)

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    llm.unload_lmstudio_model("google/gemma-4-31b")
    # /v1 stripped, native REST path appended; instance_id is the raw model key
    assert calls["url"] == "http://localhost:1234/api/v1/models/unload"
    assert calls["json"] == {"instance_id": "google/gemma-4-31b"}


def test_unload_lmstudio_is_non_fatal(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("no server")

    monkeypatch.setattr(httpx, "post", boom)
    llm.unload_lmstudio_model("x")  # must NOT raise


def test_maybe_unload_only_fires_for_lmstudio(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "unload_lmstudio_model", lambda mid: seen.append(mid))
    llm._maybe_unload("lmstudio/google/gemma-4-31b")
    llm._maybe_unload("openrouter/nvidia/nemotron")  # not local -> no unload
    llm._maybe_unload("model[0]")                     # unlabeled -> no unload
    assert seen == ["google/gemma-4-31b"]


def test_run_with_fallback_unloads_dropped_lmstudio_model(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "unload_lmstudio_model", lambda mid: seen.append(mid))

    def fake(agent, text, timeout_s=None):
        if agent == "lm":
            raise ConnectionError("lm studio oom")
        return "OK"

    monkeypatch.setattr(llm, "run_agent", fake)
    text, label = llm.run_with_fallback(
        ["lm", "or"], "prompt", labels=["lmstudio/google/gemma-4-31b", "openrouter/x"],
        unload_local=True,
    )
    assert text == "OK" and label == "openrouter/x"
    assert seen == ["google/gemma-4-31b"]  # the dropped local model was ejected


def test_run_with_fallback_success_does_not_try_rest(monkeypatch):
    seen = []

    def fake(agent, text, timeout_s=None):
        seen.append(agent)
        return "OK"

    monkeypatch.setattr(llm, "run_agent", fake)
    llm.run_with_fallback(["first", "second"], "prompt")
    assert seen == ["first"]  # stopped after first success


def test_run_with_fallback_skips_invalid_output(monkeypatch):
    # A model that returns 200 with junk (no raise) must still fall through.
    def fake(agent, text, timeout_s=None):
        return "junk" if agent == "bad" else '{"relevant": true}'

    monkeypatch.setattr(llm, "run_agent", fake)
    text, _ = llm.run_with_fallback(
        ["bad", "good"], "p", validate=lambda t: t.startswith("{")
    )
    assert text == '{"relevant": true}'


def test_run_with_fallback_returns_last_when_none_valid(monkeypatch):
    monkeypatch.setattr(llm, "run_agent", lambda a, t, timeout_s=None: "junk")
    text, _ = llm.run_with_fallback(["a", "b"], "p", validate=lambda t: False)
    assert text == "junk"  # best-effort last text; caller's parse check catches it


def test_run_with_fallback_disables_failed_model_across_calls(monkeypatch):
    # A model that rate-limits is added to `disabled` and skipped on the next call.
    seen = []

    def fake(agent, text, timeout_s=None):
        seen.append(agent)
        if agent == "bad":
            raise ConnectionError("429")
        return "OK"

    monkeypatch.setattr(llm, "run_agent", fake)
    disabled: set[int] = set()
    llm.run_with_fallback(["bad", "good"], "p", disabled=disabled)
    assert disabled == {0}  # index 0 ("bad") now disabled
    seen.clear()
    llm.run_with_fallback(["bad", "good"], "p", disabled=disabled)
    assert seen == ["good"]  # "bad" skipped entirely on the second call


def test_run_with_fallback_disables_on_unusable_output(monkeypatch):
    monkeypatch.setattr(llm, "run_agent", lambda a, t, timeout_s=None: "junk" if a == "bad" else "{}")
    disabled: set[int] = set()
    llm.run_with_fallback(["bad", "good"], "p", validate=lambda t: t == "{}", disabled=disabled)
    assert disabled == {0}  # junk-emitting model disabled for the run too


def test_run_with_fallback_raises_when_all_disabled(monkeypatch):
    monkeypatch.setattr(llm, "run_agent", lambda a, t, timeout_s=None: "OK")
    with pytest.raises(RuntimeError):
        llm.run_with_fallback(["a", "b"], "p", disabled={0, 1})


def test_build_model_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(llm.MissingApiKey, match="OPENROUTER_API_KEY"):
        llm.build_model("openrouter", "some/model")


def test_build_model_gemini_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(llm.MissingApiKey, match="GEMINI_API_KEY"):
        llm.build_model("gemini", "some-model")


def test_run_agent_times_out_instead_of_hanging(monkeypatch):
    # Root cause of the K: freeze: a stalled model call had no bound and hung forever.
    # run_agent must convert an over-long call into a TimeoutError so the chain fails over.
    async def _never_returns(agent, text):
        await asyncio.sleep(5)
        return "unreachable"

    monkeypatch.setattr(llm, "_run_async", _never_returns)
    with pytest.raises(TimeoutError):
        llm.run_agent("agent", "text", timeout_s=0.05)


def test_run_with_fallback_threads_timeout_to_run_agent(monkeypatch):
    got = {}

    def fake(agent, text, timeout_s=None):
        got["timeout_s"] = timeout_s
        return "OK"

    monkeypatch.setattr(llm, "run_agent", fake)
    llm.run_with_fallback(["a"], "p", timeout_s=42)
    assert got["timeout_s"] == 42


def test_run_with_fallback_timeout_disables_and_fails_over(monkeypatch):
    # A timed-out model is treated like any transport failure: disabled + fall through.
    def fake(agent, text, timeout_s=None):
        if agent == "slow":
            raise TimeoutError("model stalled")
        return "OK"

    monkeypatch.setattr(llm, "run_agent", fake)
    disabled: set[int] = set()
    text, _ = llm.run_with_fallback(["slow", "good"], "p", disabled=disabled)
    assert text == "OK"
    assert disabled == {0}
