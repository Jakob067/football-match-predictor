import unittest
from unittest.mock import patch

from portal import create_app


MATCH = {"day":"Sa, 12.07.","kickoff":"20:00","home_name":"Team A","away_name":"Team B",
    "home_crest":None,"away_crest":None,"prediction":"Team A win","score":"2–1",
    "home":52.0,"draw":25.0,"away":23.0,"home_xg":1.7,"away_xg":1.0,
    "home_form":10,"away_form":7}


class PortalTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_start_menu(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"N\xc3\xa4chste Spiele ansehen", response.data)

    @patch("portal._fetch", return_value=[MATCH])
    def test_general_matches_menu(self, _fetch):
        response = self.client.get("/?view=matches&competition=PL")
        self.assertIn(b"Team A", response.data)
        self.assertNotIn(b"Modell-Tipp", response.data)

    @patch("portal._fetch", return_value=[MATCH])
    def test_prediction_menu(self, _fetch):
        response = self.client.get("/?view=predictions&competition=PL")
        self.assertIn(b"Modell-Tipp", response.data)
        self.assertIn(b"52.0%", response.data)


if __name__ == "__main__":
    unittest.main()
