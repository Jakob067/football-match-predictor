import unittest

from wc_live_service import MATCHES_URL, STANDINGS_URL, _standing_adjustments


class WorldCupServiceTests(unittest.TestCase):
    def test_requested_urls_are_configured(self):
        self.assertIn("/WC/standings?stage=GROUP_STAGE", STANDINGS_URL)
        self.assertIn("/WC/matches?status=SCHEDULED", MATCHES_URL)

    def test_standings_adjustment_is_bounded(self):
        payload = {"standings":[{"type":"TOTAL", "table":[{"team":{"id":1},
            "playedGames":3, "points":9, "goalDifference":20}]}]}
        self.assertEqual(_standing_adjustments(payload)[1], 32.0)


if __name__ == "__main__":
    unittest.main()
