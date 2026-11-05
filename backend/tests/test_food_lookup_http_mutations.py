"""Tests for food lookup http mutations."""
import json
from pathlib import Path
import subprocess
import shutil

import pytest
from test_manual_data_consistency import api
from test_r013_acceptance_raw_names import food_api, create_meal, rows, update


@pytest.mark.parametrize("source", ["existing", "local", "catalog", "custom", "ai"])
@pytest.mark.parametrize("fault", ["twice", "stale", "pending"])
def test_latest_lookup_intent_and_raw_name_real_sql(food_api, source, fault, record_property):
    api = food_api
    group = create_meal(api); before = rows(api, group)
    root = Path(__file__).resolve().parents[2]
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the actual Vue payload test"
    result = subprocess.run([node, str(root/"frontend/tests/nutrition-lookup-harness.mjs"), source, fault],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert len(evidence["calls"]) == 1
    if fault == "pending": assert evidence["stages"][2] == {"blockedSaveCalls": 0, "ready": False}
    if fault == "stale": assert evidence["stages"][0] == evidence["stages"][1]
    generated = evidence["calls"][0]["body"]
    record_property("actual_component_payload", json.dumps(generated, ensure_ascii=False))
    record_property("lookup_stages", json.dumps(evidence["stages"], ensure_ascii=False))
    # Only translate synthetic row identity and preserve the unrelated existing
    # second item. Names, sources, grams and intent remain component-generated.
    generated["items"][0]["id"] = before[0]["id"]
    generated["items"].append({"id": before[1]["id"], "weight_g": float(before[1]["weight_g"])})
    response = update(api, group, **generated)
    after = rows(api, group)
    record_property("http_status", response.status_code)
    record_property("http_body", response.text)
    record_property("sql_before", json.dumps(before, ensure_ascii=False, default=str))
    record_property("sql_after", json.dumps(after, ensure_ascii=False, default=str))
    assert response.status_code == 200, response.text
    raw, canonical, kcal, protein = ("白饭", "熟大米", 116, 2.6) if fault == "twice" else ("鸡蛋", "鸡蛋", 144, 12.6)
    target = after[0] if source == "existing" else after[2]
    assert (target["raw_food_name"], target["food_name"]) == (raw, canonical)
    assert float(target["total_kcal"]) == pytest.approx(kcal)
    assert float(target["total_protein"]) == pytest.approx(protein)
    assert after[1] == before[1]
    if source == "existing": assert target["id"] == before[0]["id"]
    else: assert after[:2] == before
