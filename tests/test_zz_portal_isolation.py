"""Keep the base and enhanced portal templates isolated during unittest discovery."""
import importlib

import portal
import portal_details
import portal_players


_detailed_page = portal.PAGE
_detail_view = portal_details.app.view_functions["index"]
importlib.reload(portal)
_base_page = portal.PAGE


def _isolated_detail_view():
    previous_page, previous_fetch = portal.PAGE, portal._fetch
    portal.PAGE = _detailed_page
    portal._fetch = portal_players._fetch_with_players
    try:
        return _detail_view()
    finally:
        portal.PAGE, portal._fetch = previous_page, previous_fetch


portal_details.app.view_functions["index"] = _isolated_detail_view
