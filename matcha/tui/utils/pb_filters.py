"""
Utility functions for PoseBusters filter extraction.
"""

from typing import Dict, Any, List


def extract_pb_filters(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PoseBusters filters from a sample dictionary.

    Args:
        sample: Sample dictionary containing posebusters_filters_fast key

    Returns:
        Dictionary with extracted filter values:
        - not_too_far_away (bool)
        - no_internal_clash (bool)
        - no_clashes (bool)
        - no_volume_clash (bool)
        - buried_fraction (float)
    """
    pb_filters = sample.get("posebusters_filters_fast", [False] * 5)

    # Ensure we have default values if filter list is shorter than expected
    defaults: List[Any] = [False, False, False, False, 0.0]
    flags = list(pb_filters) + defaults[len(pb_filters):]

    return {
        "not_too_far_away": bool(flags[0]),
        "no_internal_clash": bool(flags[1]),
        "no_clashes": bool(flags[2]),
        "no_volume_clash": bool(flags[3]),
        "buried_fraction": float(flags[4]),
    }
