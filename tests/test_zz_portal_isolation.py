"""Base and detailed apps share markup without changing each other's view options."""
import unittest
from unittest.mock import patch

import portal
from tests.test_portal import MATCH


class PortalIsolationTests(unittest.TestCase):
    @patch("portal._fetch", return_value=[MATCH])
    def test_details_option_is_per_app(self, _fetch):
        base = portal.create_app().test_client()
        enhanced = portal.create_app(details_enabled=True).test_client()
        original_page = portal.PAGE
        self.assertIn(b"data-match-details", enhanced.get("/?view=matches").data)
        self.assertNotIn(b"data-match-details", base.get("/?view=matches").data)
        self.assertEqual(portal.PAGE, original_page)
