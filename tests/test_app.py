import unittest

from app import create_app
from prediction_engine import COMPETITIONS, predict, teams_for


class PredictionTests(unittest.TestCase):
    def test_all_competitions_have_two_teams(self):
        for code in COMPETITIONS:
            self.assertGreaterEqual(len(teams_for(code)), 2, code)

    def test_probabilities_sum_to_one_hundred(self):
        teams = teams_for("WC")
        result = predict(teams[0], teams[1], neutral=True)
        self.assertAlmostEqual(result["home"] + result["draw"] + result["away"], 100, delta=.2)

    def test_interface_and_prediction(self):
        client = create_app("CA").test_client()
        self.assertEqual(client.get("/").status_code, 200)
        response = client.post("/", data={"competition":"CA", "home_team":"Argentina",
            "away_team":"Brazil", "neutral":"on", "action":"predict"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Most likely score", response.data)


if __name__ == "__main__":
    unittest.main()
