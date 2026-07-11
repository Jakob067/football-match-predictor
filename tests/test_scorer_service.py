import unittest
from scorer_service import add_scorer_predictions


SCORERS = [
    {"player":{"name":"Alex Striker"},"team":{"name":"Team A"},"goals":8,"playedMatches":10},
    {"player":{"name":"Ben Forward"},"team":{"name":"Team B"},"goals":4,"playedMatches":10},
]
MATCH = {"home_name":"Team A","away_name":"Team B","home_xg":1.8,"away_xg":.9}


class ScorerTests(unittest.TestCase):
    def test_scorers_are_ranked_and_bounded(self):
        from unittest.mock import patch
        with patch("scorer_service.current_scorers", return_value=SCORERS):
            result = add_scorer_predictions([MATCH.copy()], "PL")[0]["scorers"]
        self.assertEqual(result[0]["name"], "Alex Striker")
        self.assertGreater(result[0]["probability"], result[1]["probability"])
        self.assertLessEqual(result[0]["probability"], 100)


if __name__ == "__main__":
    unittest.main()
