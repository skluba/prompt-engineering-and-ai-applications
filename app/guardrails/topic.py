"""Keep the analytics assistant focused on the loaded dataset."""

from __future__ import annotations

import re

_DATAISH = re.compile(
    r"\b("
    r"table|tables|row|rows|column|columns|data|dataset|csv|sql|query|select|join|"
    r"count|sum|avg|average|mean|min|max|group|filter|where|chart|plot|graph|"
    r"aggregate|aggregation|distinct|order|limit|top\s+\d+|how\s+many|breakdown|"
    r"compare|comparison|trend|per\s+\w+|by\s+\w+|list|show|display|calculate"
    r")\b",
    re.IGNORECASE,
)

_STRONG_OFF_TOPIC = re.compile(
    r"\b("
    r"weather\s+in|recipe\s+for|write\s+(me\s+)?(a\s+)?(poem|essay|story)|"
    r"who\s+won\s+the\s+super\s+bowl|lyrics\s+to|torrent\s+download|"
    r"investment\s+advice|medical\s+diagnosis|legal\s+advice"
    r")\b",
    re.IGNORECASE,
)


def detect_off_topic(text: str, table_names: list[str]) -> str | None:
    """Return a short reason if the message is probably off-topic; else ``None``."""
    t = text.strip()
    if len(t) <= 100:
        return None
    tl = t.lower()
    if any(name.lower() in tl for name in table_names if name):
        return None
    if _DATAISH.search(t):
        return None
    if re.search(r"\d", t):
        return None
    if _STRONG_OFF_TOPIC.search(t):
        return (
            "This assistant only answers questions about the dataset you selected "
            "(tables, metrics, and charts). Please rephrase your question in those terms."
        )
    if len(t) > 280:
        return (
            "Your message is long and does not mention data or your tables. "
            "Ask something specific about the loaded dataset "
            "(e.g. counts, filters, or comparisons)."
        )
    return None
