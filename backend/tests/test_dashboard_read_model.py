
import unittest
from datetime import date
from unittest.mock import patch
from fastapi import HTTPException
from app.api.dashboard import aggregate, read_dashboard

class DashboardTests(unittest.TestCase):
    def test_missing_and_zero_are_distinct(self):
        meals=[dict(date_key="2026-09-22",calories=0,protein_g=0,carbs_g=0,fat_g=0)]
        days,_=aggregate([],[],meals,date(2026,9,21),2)
        self.assertIsNone(days[0]["calories"])
        self.assertEqual(days[1]["calories"],0)
        self.assertIsNone(days[1]["tdee"])

    def test_partial_nutrients_do_not_understate_total(self):
        meals=[dict(date_key="2026-09-22",calories=100,protein_g=10),
               dict(date_key="2026-09-22",calories=None,protein_g=5)]
        days,_=aggregate([],[],meals,date(2026,9,22),1)
        self.assertIsNone(days[0]["calories"])
        self.assertEqual(days[0]["protein_g"],15)

    def test_completed_sets_and_no_100_record_cap(self):
        sessions=[dict(session_date="2026-09-22",exercises=[dict(exercise_name="test",sets=[
            dict(weight_kg=20,reps=10,completed=True),dict(weight_kg=100,reps=10,completed=False)])]) for _ in range(105)]
        days,best=aggregate([],sessions,[],date(2026,9,22),1)
        self.assertEqual(days[0]["training_volume"],21000)
        self.assertEqual(days[0]["completed_sets"],105)
        self.assertEqual(best[0]["weight_kg"],20)

    def test_legacy_details_take_priority(self):
        workouts=[dict(date="2026-09-22",exercise="test",sets=3,reps=10,weight_kg=50,completed=True,set_details=[])]
        days,_=aggregate(workouts,[],[],date(2026,9,22),1)
        self.assertEqual(days[0]["completed_sets"],0)
        workouts[0]["set_details"]=None
        days,_=aggregate(workouts,[],[],date(2026,9,22),1)
        self.assertEqual(days[0]["training_volume"],1500)

    def test_database_failure_has_no_demo_fallback(self):
        with patch("app.api.dashboard.db.SessionLocal",None):
            with self.assertRaises(HTTPException) as ctx:
                read_dashboard(days=7)
        self.assertEqual(ctx.exception.status_code,503)

if __name__=="__main__":unittest.main()
