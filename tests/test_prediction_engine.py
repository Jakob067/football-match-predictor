import unittest

from prediction_engine import Team, predict
from wc_live_service import _must_decide


class DecisivePredictionTests(unittest.TestCase):
    def test_knockout_prediction_never_returns_draw_score(self):
        result = predict(
            Team("Team A", 1800),
            Team("Team B", 1800),
            neutral=True,
            must_decide=True,
        )

        home_goals, away_goals = result["score"].split("–")
        self.assertNotEqual(home_goals, away_goals)
        self.assertNotEqual(result["prediction"], "Unentschieden")
        self.assertAlmostEqual(
            result["home_advance"] + result["away_advance"],
            100.0,
            places=1,
        )

    def test_group_match_can_still_end_in_a_draw(self):
        self.assertFalse(_must_decide({"stage": "GROUP_STAGE"}))

    def test_world_cup_knockout_stages_require_a_winner(self):
        for stage in ("LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"):
            with self.subTest(stage=stage):
                self.assertTrue(_must_decide({"stage": stage}))


if __name__ == "__main__":
    unittest.main()