"""Dual-RAG Query Routing
=========================
Routes queries between Sage (curriculum) and Scout (FAQ) by comparing
retrieval scores from both FAISS indexes. No keyword lists needed.

Both indexes use the same embedding model (text-embedding-3-small) and
L2 distance metric, so scores are directly comparable.

Logic (FAISS L2 distance -- lower = better match):
  CONFIDENT_MATCH = 1.0  : score below this = high confidence match
  POOR_MATCH      = 1.5  : score above this = no useful match found

  Sage: redirect to Scout only if FAQ score is confident AND better than curriculum.
        Otherwise Sage handles (curriculum context or general LLM knowledge).

  Scout: redirect to Sage if curriculum score is confident AND better than FAQ,
         OR if FAQ score is poor (Sage has general knowledge fallback, Scout does not).
"""

from typing import Tuple, Optional, List, Dict

# FAISS L2 distance thresholds (tuned on real queries)
CONFIDENT_MATCH = 1.0   # Below this: high confidence the KB has a good answer
POOR_MATCH = 1.5        # Above this: KB has nothing useful


def _best_score(results: List[Dict]) -> float:
    """Extract best (lowest) FAISS L2 score from results. Returns inf if empty."""
    if not results:
        return float('inf')
    return results[0].get('score', float('inf'))


def route_query(
    current_bot: str,
    curr_results: List[Dict],
    faq_results: List[Dict]
) -> Tuple[bool, Optional[str]]:
    """
    Decide whether to redirect to the other bot based on dual RAG scores.

    Args:
        current_bot: 'sage' or 'scout'
        curr_results: Sage curriculum RAG results (with 'score' field)
        faq_results: Scout FAQ RAG results (with 'score' field)

    Returns:
        (should_redirect, reason)
        reason: 'faq_better' | 'curriculum_better' | 'scout_no_match' | None
    """
    curr_score = _best_score(curr_results)
    faq_score = _best_score(faq_results)

    if current_bot == 'sage':
        # Redirect to Scout only if FAQ is a confident match AND clearly better
        if faq_score < CONFIDENT_MATCH and faq_score < curr_score:
            return (True, 'faq_better')
        # All other cases: Sage handles (curriculum or general knowledge)
        return (False, None)

    else:  # scout
        # Redirect to Sage if curriculum is a confident match AND clearly better
        if curr_score < CONFIDENT_MATCH and curr_score < faq_score:
            return (True, 'curriculum_better')
        # Scout has no general knowledge fallback -- redirect if FAQ can't help
        if faq_score > POOR_MATCH:
            return (True, 'scout_no_match')
        return (False, None)


def generate_redirect_message(current_bot: str, reason: str) -> str:
    """Generate user-friendly redirect message."""
    if current_bot == 'sage':
        return (
            "Hey! This looks like a program logistics question. "
            "I'm Sage -- I specialize in technical curriculum help (code, concepts, debugging).\n\n"
            "For questions about LMS, recordings, Launchpad, Jarvis Labs, or session timings, "
            "try asking **@Scout** -- they handle all program support!"
        )
    else:  # scout
        if reason == 'scout_no_match':
            return (
                "Hmm, I couldn't find a match in the program FAQ for that.\n\n"
                "Try asking **@Sage** -- they can help with technical questions and general queries!"
            )
        return (
            "Hey! This looks like a technical question. "
            "I'm Scout -- I handle program logistics (LMS, recordings, Jarvis Labs, Launchpad).\n\n"
            "For technical help with curriculum concepts, code, or debugging, "
            "try asking **@Sage** -- they're the tech expert!"
        )


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("Testing Dual-RAG Routing:\n" + "=" * 60)

    all_passed = True
    tests = [
        # (current_bot, curr_results, faq_results, expected_redirect, description)

        # --- Sage tests ---
        # FAQ clearly better -> redirect
        ('sage', [{'score': 2.1}], [{'score': 0.4}], True,
            "Sage: FAQ confident + better -> redirect to Scout"),
        # Curriculum better -> no redirect
        ('sage', [{'score': 0.3}], [{'score': 1.8}], False,
            "Sage: curriculum good match -> no redirect"),
        # Both poor -> no redirect (Sage uses general knowledge)
        ('sage', [{'score': 2.0}], [{'score': 1.9}], False,
            "Sage: both poor -> no redirect (general knowledge)"),
        # FAQ slightly better but NOT confident (> 1.0) -> no redirect
        ('sage', [{'score': 1.2}], [{'score': 1.1}], False,
            "Sage: FAQ marginally better but not confident -> no redirect"),
        # Both empty -> no redirect
        ('sage', [], [], False,
            "Sage: no results either side -> no redirect"),
        # FAQ good but curriculum is better -> no redirect
        ('sage', [{'score': 0.5}], [{'score': 0.8}], False,
            "Sage: curriculum better than FAQ -> no redirect"),
        # FAQ exactly at threshold -> no redirect (must be strictly below)
        ('sage', [{'score': 1.5}], [{'score': 1.0}], False,
            "Sage: FAQ at CONFIDENT_MATCH boundary -> no redirect"),

        # --- Scout tests ---
        # Curriculum clearly better -> redirect to Sage
        ('scout', [{'score': 0.5}], [{'score': 2.0}], True,
            "Scout: curriculum confident + better -> redirect to Sage"),
        # FAQ better -> no redirect
        ('scout', [{'score': 1.8}], [{'score': 0.3}], False,
            "Scout: FAQ good match -> no redirect"),
        # Both poor -> redirect to Sage (Scout has no fallback)
        ('scout', [{'score': 2.0}], [{'score': 1.8}], True,
            "Scout: both poor -> redirect to Sage"),
        # FAQ okay (below POOR_MATCH) -> no redirect even if curriculum is also okay
        ('scout', [{'score': 1.3}], [{'score': 1.0}], False,
            "Scout: FAQ okay match -> no redirect"),
        # Scout empty, curriculum good -> redirect
        ('scout', [{'score': 0.6}], [], True,
            "Scout: no FAQ results, curriculum good -> redirect to Sage"),
        # Both empty -> redirect (Scout can't help)
        ('scout', [], [], True,
            "Scout: no results either side -> redirect to Sage"),
    ]

    for bot, curr, faq, expected, desc in tests:
        redirect, reason = route_query(bot, curr, faq)
        status = "[OK]" if redirect == expected else "[FAIL]"
        if redirect != expected:
            all_passed = False
        print(f"{status} {desc}")
        print(f"     redirect={redirect} (expected {expected}), reason={reason}\n")

    print("=" * 60)
    print("[OK] All tests passed!" if all_passed else "[FAIL] Some tests failed - check above")
