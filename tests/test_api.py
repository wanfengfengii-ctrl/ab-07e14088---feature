"""API-level tests for the delay-plan compiler service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def typical_request() -> dict:
    return {
        "targets": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
        "minDelay": 0,
        "maxDelay": 40,
        "maxStep": 2,
        "maxRamps": 3,
        "anchors": [
            {"element": 0, "delay": 10},
            {"element": 12, "delay": 20},
            {"element": 23, "delay": 9},
        ],
    }


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_compile_typical_feasible():
    req = typical_request()
    resp = client.post("/api/delay-plans/compile", json=req)
    assert resp.status_code == 200
    body = resp.json()
    assert body["feasible"] is True
    delays = body["delays"]
    assert len(delays) == len(req["targets"])
    assert len(body["errors"]) == len(req["targets"])
    assert body["errors"] == [d - t for d, t in zip(delays, req["targets"])]
    for a in req["anchors"]:
        assert delays[a["element"]] == a["delay"]
    assert all(req["minDelay"] <= d <= req["maxDelay"] for d in delays)
    assert all(abs(delays[i + 1] - delays[i]) <= req["maxStep"]
               for i in range(len(delays) - 1))
    diffs = [delays[i + 1] - delays[i] for i in range(len(delays) - 1)]
    runs = 1 + sum(1 for i in range(1, len(diffs)) if diffs[i] != diffs[i - 1])
    assert runs == body["rampCount"] <= req["maxRamps"]
    assert len(body["ramps"]) == runs
    abs_errors = [abs(e) for e in body["errors"]]
    assert body["maxAbsError"] == max(abs_errors)
    assert body["totalAbsError"] == sum(abs_errors)
    # the exact targets are reachable here, so the optimum is error-free
    assert body["maxAbsError"] == 0
    assert body["rampCount"] == 2


def test_compile_is_deterministic_over_http():
    req = typical_request()
    first = client.post("/api/delay-plans/compile", json=req).json()
    second = client.post("/api/delay-plans/compile", json=req).json()
    assert first == second


def test_compile_anchor_conflict_returns_stable_infeasible():
    req = {
        "targets": [0] * 12,
        "minDelay": 0,
        "maxDelay": 40,
        "maxStep": 2,
        "maxRamps": 4,
        "anchors": [
            {"element": 1, "delay": 0},
            {"element": 5, "delay": 20},
        ],
    }
    resp = client.post("/api/delay-plans/compile", json=req)
    assert resp.status_code == 200
    body = resp.json()
    assert body["feasible"] is False
    assert body["reason"] == "anchor-step-conflict"
    # no partial delay table may be emitted
    assert body["delays"] is None
    assert body["errors"] is None
    assert body["ramps"] is None
    conflict = body["conflict"]
    assert conflict["startElement"] == 1
    assert conflict["endElement"] == 5
    assert conflict["requiredChange"] == 20
    assert conflict["allowedChange"] == 8
    # stable across repeated calls
    again = client.post("/api/delay-plans/compile", json=req).json()
    assert again == body


def test_compile_ramp_limit_infeasible():
    req = {
        "targets": [0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        "minDelay": 0,
        "maxDelay": 10,
        "maxStep": 2,
        "maxRamps": 2,
        "anchors": [
            {"element": 0, "delay": 0},
            {"element": 2, "delay": 2},
            {"element": 4, "delay": 0},
            {"element": 6, "delay": 2},
            {"element": 11, "delay": 7},
        ],
    }
    resp = client.post("/api/delay-plans/compile", json=req)
    assert resp.status_code == 200
    body = resp.json()
    assert body["feasible"] is False
    assert body["reason"] == "ramp-limit-exceeded"
    assert body["minRampsRequired"] == 3
    assert body["delays"] is None


# ---------------------------------------------------------------------------
# minRampSteps (minimum ramp hold length)
# ---------------------------------------------------------------------------

def test_legacy_request_shape_is_unchanged():
    # no minRampSteps: request and response keep the exact legacy contract
    req = typical_request()
    assert "minRampSteps" not in req
    body = client.post("/api/delay-plans/compile", json=req).json()
    assert body["feasible"] is True
    for ramp in body["ramps"]:
        assert "stepCount" not in ramp
        assert set(ramp) == {
            "startElement", "endElement", "slope",
            "startValue", "endValue"}


def test_min_ramp_steps_feasible_reports_step_count():
    req = dict(typical_request(), minRampSteps=3)
    resp = client.post("/api/delay-plans/compile", json=req)
    assert resp.status_code == 200
    body = resp.json()
    assert body["feasible"] is True
    delays = body["delays"]
    n = len(delays)
    for ramp in body["ramps"]:
        # stepCount present on every ramp and consistent with boundaries
        assert ramp["stepCount"] == ramp["endElement"] - ramp["startElement"]
        assert ramp["stepCount"] >= 3
        assert ramp["startValue"] == delays[ramp["startElement"]]
        assert ramp["endValue"] == delays[ramp["endElement"]]
        assert (delays[ramp["endElement"]] - delays[ramp["startElement"]]
                == ramp["slope"] * ramp["stepCount"])
    assert body["ramps"][0]["startElement"] == 0
    assert body["ramps"][-1]["endElement"] == n - 1


def test_min_ramp_steps_span_conflict_is_stable_and_partial_free():
    # forced 2 -> 0 drop spans only two steps at the end, so a hold of 3
    # cannot be met jointly with the anchors
    req = {
        "targets": [2] * 12,
        "minDelay": 0,
        "maxDelay": 40,
        "maxStep": 1,
        "maxRamps": 6,
        "minRampSteps": 3,
        "anchors": [
            {"element": 0, "delay": 2},
            {"element": 8, "delay": 2},
            {"element": 11, "delay": 0},
        ],
    }
    resp = client.post("/api/delay-plans/compile", json=req)
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "feasible": False,
        "reason": "ramp-span-conflict",
        "minRampSteps": 3,
        "delays": None,
        "errors": None,
        "ramps": None,
        "maxAbsError": None,
        "totalAbsError": None,
        "rampCount": None,
    }
    again = client.post("/api/delay-plans/compile", json=req).json()
    assert again == body


def test_min_ramp_steps_conflict_even_under_ramp_cap_shortage():
    # three ramps of two steps each would satisfy the hold, but the cap is
    # two; with the hold enabled the joint system still reports span
    # conflict rather than ramp-limit-exceeded
    req = {
        "targets": [0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        "minDelay": 0,
        "maxDelay": 10,
        "maxStep": 2,
        "maxRamps": 2,
        "minRampSteps": 2,
        "anchors": [
            {"element": 0, "delay": 0},
            {"element": 2, "delay": 2},
            {"element": 4, "delay": 0},
            {"element": 6, "delay": 2},
            {"element": 11, "delay": 7},
        ],
    }
    body = client.post("/api/delay-plans/compile", json=req).json()
    assert body["feasible"] is False
    assert body["reason"] == "ramp-span-conflict"
    assert body["delays"] is None


def test_min_ramp_steps_validation_bounds():
    for bad in (1, 9, 0, -2):
        req = dict(typical_request(), minRampSteps=bad)
        assert client.post("/api/delay-plans/compile", json=req).status_code == 422
    for good in (2, 8):
        req = dict(typical_request(), minRampSteps=good)
        assert client.post("/api/delay-plans/compile", json=req).status_code == 200


def _invalid_requests():
    base = typical_request()
    too_few_targets = dict(base, targets=[1] * 11)
    too_many_targets = dict(base, targets=[1] * 49)
    inverted_interval = dict(base, minDelay=30, maxDelay=10)
    negative_step = dict(base, maxStep=-1)
    zero_ramps = dict(base, maxRamps=0)
    one_anchor = dict(base, anchors=[{"element": 0, "delay": 10}])
    nine_anchors = dict(
        base,
        anchors=[{"element": i, "delay": 10} for i in range(9)],
    )
    anchor_out_of_range = dict(
        base, anchors=[{"element": 0, "delay": 10},
                       {"element": 24, "delay": 10}])
    duplicate_anchor = dict(
        base, anchors=[{"element": 0, "delay": 10},
                       {"element": 0, "delay": 10},
                       {"element": 3, "delay": 5}])
    bad_target_type = dict(base, targets=["x"] * 12)
    return [
        too_few_targets, too_many_targets, inverted_interval, negative_step,
        zero_ramps, one_anchor, nine_anchors, anchor_out_of_range,
        duplicate_anchor, bad_target_type,
    ]


def test_validation_errors_return_422():
    for req in _invalid_requests():
        resp = client.post("/api/delay-plans/compile", json=req)
        assert resp.status_code == 422, req
