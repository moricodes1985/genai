"""
---------
Central place for safety, disclaimers, and lightweight input/output guards
for the License RAG demo.
"""

from typing import Tuple


DISCLAIMER = "⚠️ Educational demo. Not legal advice."


def check_input(message: str) -> Tuple[bool, str]:
    """
    Lightweight input guard:
    - Reject empty queries
    - Warn on suspicious queries (e.g., personal/legal/medical requests)
    Returns (ok, reason_or_empty).
    """
    msg = message.strip().lower()
    if not msg:
        return False, "Empty query not allowed."

    if "lawsuit" in msg or "personal data" in msg or "medical" in msg:
        return False, "Query blocked: unsafe or out of scope."

    return True, ""


def wrap_output(answer: str, sources: list[str]) -> str:
    result = f"{DISCLAIMER}\n\n{answer}"
    if sources:
        result += "\n\nSources:\n" + "\n".join(sorted(set(sources)))
    return result


