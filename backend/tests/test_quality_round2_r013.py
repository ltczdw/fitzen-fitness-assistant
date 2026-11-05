"""Actual Vue save payload fed to actual HTTP/SQL mutation; all data synthetic."""
import json,subprocess
import shutil
from pathlib import Path
import pytest
from sqlalchemy import event
from test_manual_data_consistency import api,USER
from test_r013_acceptance_raw_names import food_api,create_meal,rows,update


@pytest.mark.parametrize("mode",["init","lookup_unchanged","same_name","rename","add_local_alias","add_custom","add_ai"])
def test_frontend_name_intent_real_mutation(food_api,mode,record_property):
    api=food_api;group=create_meal(api);before=rows(api,group)
    root=Path(__file__).resolve().parents[2]
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the actual Vue payload test"
    result=subprocess.run([node,str(root/"frontend/tests/nutrition-edit-harness.mjs"),mode],cwd=root,capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert result.returncode==0,result.stderr
    body=json.loads(result.stdout)["body"]
    body["items"][0]["id"]=before[0]["id"]
    body["items"].append({"id":before[1]["id"],"weight_g":float(before[1]["weight_g"])})
    record_property("actual_vue_body",json.dumps(body,ensure_ascii=False))
    response=update(api,group,**body);assert response.status_code==200,response.text
    after=rows(api,group)
    expected="白米饭" if mode=="rename" else "米饭"
    assert after[0]["raw_food_name"]==expected
    assert after[0]["food_name"]=="熟大米" and after[0]["id"]==before[0]["id"]
    assert next(x for x in after if x["id"]==before[1]["id"])==before[1]
    if mode.startswith("add_"):
        added=next(x for x in after if x["id"] not in [r["id"] for r in before])
        assert added["raw_food_name"]==("白饭" if mode=="add_local_alias" else "合成自制食物")


def test_missing_resource_404_and_real_storage_503_no_changes(food_api):
    api=food_api;group=create_meal(api);before=rows(api,group)
    assert update(api,"missing-resource",items=[{"food_name":"米饭","weight_g":100}]).status_code==404
    hit=[]
    def fail(conn,cursor,stmt,params,context,many):
        if stmt.lstrip().upper().startswith("UPDATE") and "user_food_log" in stmt:
            hit.append(True);raise RuntimeError("quality-real-storage-fault")
    event.listen(api.engine,"before_cursor_execute",fail)
    try:response=update(api,group,items=[{"id":r["id"],"weight_g":200} for r in before])
    finally:event.remove(api.engine,"before_cursor_execute",fail)
    assert hit and response.status_code==503,response.text
    assert rows(api,group)==before
