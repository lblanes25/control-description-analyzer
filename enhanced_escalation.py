import re

def enhance_escalation_detection(text, nlp, existing_keywords=None):
    """Enhanced ESCALATION detection using both dependency parsing and pattern-based analysis"""
    if not text or text.strip() == '':
        return {
            "detected": False,
            "score": 0,
            "type": None,
            "phrases": [],
            "suggestions": []
        }

    doc = nlp(text)

    escalation_verbs = existing_keywords or {
        "escalate", "notify", "inform", "report", "raise", "alert", "communicate"
    }

    escalation_targets = {
        "management", "supervisor", "manager", "leadership", "committee",
        "executive", "director", "cfo", "board", "team"
    }

    escalation_phrases = []
    suggestions = []
    matched_types = []

    for token in doc:
        if token.lemma_.lower() in escalation_verbs and token.pos_ in {"VERB", "AUX"}:
            target_found = False
            for child in token.children:
                if child.dep_ in {"dobj", "pobj", "attr", "obl"} and child.text.lower() in escalation_targets:
                    escalation_phrases.append({
                        "text": f"{token.text} {child.text}",
                        "span": [token.idx, child.idx + len(child)],
                        "pattern_type": "dependency"
                    })
                    matched_types.append("dependency")
                    target_found = True
                    break
            if not target_found:
                escalation_phrases.append({
                    "text": token.text,
                    "span": [token.idx, token.idx + len(token)],
                    "pattern_type": "verb_only"
                })
                suggestions.append("Escalation verb detected without a clear recipient — consider specifying a target (e.g., 'to management')")

    patterns = {
        "explicit": {
            "score": 1.0,
            "patterns": [
                r"escalate[ds]?\sto\s(?:the\s)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                r"notif(?:y|ies|ied)\s(?:the\s)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                r"report[eds]?\sto\s(?:the\s)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
            ]
        },
        "process_with_governance": {
            "score": 0.7,
            "patterns": [
                r"(?:through|via|using|per|following)\s(?:the\s)?([a-zA-Z\s]+)\sprocess\s(?:.*?\s)?(?:approv|authoriz|review)",
                r"escalate[ds]?\s(?:via|through|to)\s(?:the\s)?([a-zA-Z\s]+)\sprocess\s(?:.*?)?(?:level|tier|approv)",
                r"(?:using|through|following)\s(?:the\s)?([a-zA-Z\s]+)\sprocedure\s(?:.*?)(?:approv|authoriz|review)"
            ]
        },
        "process_generic": {
            "score": 0.4,
            "patterns": [
                r"(?:through|via|using|per|following)\s(?:the\s)?([a-zA-Z\s]+)\sprocess",
                r"escalate[ds]?\s(?:via|through|to)\s(?:the\s)?([a-zA-Z\s]+)\sprocess",
                r"handle[ds]?\sby\s(?:the\s)?([a-zA-Z\s]+)\sprocess"
            ]
        }
    }

    highest_score = 0
    pattern_type = None

    for p_type, p_info in patterns.items():
        for pattern in p_info["patterns"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                escalation_phrases.append({
                    "text": match.group(0),
                    "span": [match.start(), match.end()],
                    "pattern_type": p_type
                })
                if p_info["score"] > highest_score:
                    highest_score = p_info["score"]
                    pattern_type = p_type
                matched_types.append(p_type)

    detected = len(escalation_phrases) > 0
    base_score = max(highest_score, 0.5 if "dependency" in matched_types else 0)
    score = min(base_score + min(0.2, len(escalation_phrases) * 0.1), 3.0)

    if not detected and re.search(r"exception|issue|error|discrepanc|problem|failure", text, re.IGNORECASE):
        suggestions.append("Control mentions exceptions but doesn't specify how they are handled or escalated")

    return {
        "detected": detected,
        "score": score if detected else 0,
        "type": "hybrid",
        "phrases": escalation_phrases,
        "suggestions": suggestions
    }