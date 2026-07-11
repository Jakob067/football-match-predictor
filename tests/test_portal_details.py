import unittest
from unittest.mock import patch

import portal_details


MATCH = {"day":"Sa, 12.07.","kickoff":"20:00","home_name":"Team A","away_name":"Team B",
    "home_crest":None,"away_crest":None,"prediction":"Team A win","score":"2–1",
    "home":52.0,"draw":25.0,"away":23.0,"home_xg":1.7,"away_xg":1.0,
    "home_form":10,"away_form":7,"confidence":"Medium","scorers":[]}


class DetailPortalTests(unittest.TestCase):
    @patch("portal_details.portal._fetch", return_value=[MATCH])
    def test_clickable_details_render(self, _fetch):
        response = portal_details.app.test_client().get("/?view=matches&competition=PL")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Genaue Analyse anzeigen", response.data)
        self.assertIn(b"Expected Goals", response.data)
        self.assertIn(b"toggleMatch", response.data)


if __name__ == "__main__":
    unittest.main()
