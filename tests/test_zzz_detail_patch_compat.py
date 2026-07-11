"""Ensure mocked data sources remain effective for the isolated enhanced portal."""
import portal
import portal_details
import tests.test_zz_portal_isolation as isolation


_detail_view = isolation._detail_view


def _mock_compatible_detail_view():
    previous_page = portal.PAGE
    portal.PAGE = isolation._detailed_page
    try:
        return _detail_view()
    finally:
        portal.PAGE = previous_page


portal_details.app.view_functions["index"] = _mock_compatible_detail_view
