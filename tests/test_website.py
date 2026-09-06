import unittest
from website import create_app


class WebsiteTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app("CA").test_client()

    def test_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Matchday", response.data)

    def test_adjusted_prediction_renders(self):
        response = self.client.post("/", data={"competition":"CA", "home_team":"Argentina",
            "away_team":"Brazil", "neutral":"on", "home_form":"15", "away_form":"3",
            "home_missing":"0", "away_missing":"2", "action":"predict"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Argentina gewinnt", response.data)
        self.assertIn(b"data-width", response.data)

    def test_invalid_numbers_are_safe(self):
        response = self.client.post("/", data={"competition":"PL", "home_team":"Arsenal",
            "away_team":"Liverpool", "home_form":"bad", "action":"predict"})
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
