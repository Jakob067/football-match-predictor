import unittest
from unittest.mock import patch

from live_service import build_predictions, calculate_ratings
from live_website import create_app


FINISHED = [{"utcDate":"2026-07-01T18:00:00Z", "homeTeam":{"id":1,"name":"Argentina"},
    "awayTeam":{"id":2,"name":"Brazil"}, "score":{"fullTime":{"home":2,"away":0}}}]
SCHEDULED = [{"id":7,"utcDate":"2099-08-01T18:00:00Z",
    "homeTeam":{"id":1,"name":"Argentina","shortName":"Argentina","crest":"a.svg"},
    "awayTeam":{"id":2,"name":"Brazil","shortName":"Brazil","crest":"b.svg"}}]


class LivePredictionTests(unittest.TestCase):
    def test_finished_results_update_ratings(self):
        ratings, form = calculate_ratings(FINISHED, "WC")
        self.assertGreater(ratings[1], 1940)
        self.assertEqual(form[1], [3])

    def test_scheduled_matches_are_predicted(self):
        rows = build_predictions(FINISHED, SCHEDULED, "WC")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["home"] + rows[0]["draw"] + rows[0]["away"], 100, delta=.2)

    @patch("live_website.next_predictions", return_value=[])
    def test_live_page(self, _mock):
        response = create_app().test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Live-Daten", response.data)


if __name__ == "__main__":
    unittest.main()
