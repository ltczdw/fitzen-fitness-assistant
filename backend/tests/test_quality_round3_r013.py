"""Actual component-generated payload into isolated real HTTP and food SQL."""
import json
from pathlib import Path
import subprocess
import shutil
import pytest
from test_manual_data_consistency import api
from test_r013_acceptance_raw_names import food_api, create_meal, rows, update


@pytest.mark.parametrize("mode", ["add_local_then_rename", "add_catalog_then_rename", "add_custom_then_rename", "add_ai_then_rename", "add_local_alias_weight"])
def test_added_then_renamed_component_and_sql_follow_last_intent(food_api, mode, record_property):
    api = food_api; group = create_meal(api); before = rows(api, group)
    root = Path(__file__).resolve().parents[2]
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the actual Vue payload test"
    result = subprocess.run([node, str(root/"frontend/tests/nutrition-edit-harness.mjs"), mode], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    generated = json.loads(result.stdout)["body"]
    record_property("actual_component_payload", json.dumps(generated, ensure_ascii=False))
    # Translate only the synthetic existing row ID; never correct the new name/nutrients.
    generated["items"][0]["id"] = before[0]["id"]
    generated["items"].append({"id": before[1]["id"], "weight_g": float(before[1]["weight_g"])})
    response = update(api, group, **generated)
    record_property("http_status", response.status_code); record_property("http_body", response.text)
    after = rows(api, group)
    record_property("sql_before", json.dumps(before, ensure_ascii=False, default=str))
    record_property("sql_after", json.dumps(after, ensure_ascii=False, default=str))
    assert response.status_code == 200, response.text
    assert after[:2] == before
    added = after[2]
    if mode == "add_local_alias_weight":
        assert (added["raw_food_name"], added["food_name"]) == ("白饭", "熟大米")
        assert float(added["total_kcal"]) == 174
    else:
        assert (added["raw_food_name"], added["food_name"]) == ("鸡蛋", "鸡蛋"), added
        grams = generated["items"][1]["weight_g"]
        assert float(added["total_kcal"]) == pytest.approx(144*grams/100)
        # rows() returns persisted user_food_log columns, whose portion protein
        # column is total_protein; protein_g belongs to the API/catalog shape.
        assert float(added["total_protein"]) == pytest.approx(12.6*grams/100)
