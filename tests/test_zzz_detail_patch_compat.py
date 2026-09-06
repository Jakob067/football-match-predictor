"""The enhanced entrypoint respects a mocked data source without template patching."""
import unittest
from unittest.mock import patch

import portal_details
from tests.test_portal_details import MATCH


class DetailSourceCompatibilityTests(unittest.TestCase):
    @patch("portal_details.portal._fetch", return_value=[MATCH])
    def test_enhanced_app_uses_selected_competition(self, fetch):
        response = portal_details.app.test_client().get("/?view=matches&competition=BL1")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Expected Goals", response.data)
        fetch.assert_called_once_with("BL1")
