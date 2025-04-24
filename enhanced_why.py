import spacy
from typing import Dict, List, Any, Optional, Tuple, Union
import re


def enhance_why_detection(text: str, nlp, risk_description: str = None, existing_keywords: List[str] = None):
    """
    Enhanced WHY detection that evaluates alignment with mapped risks

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        risk_description: The text of the mapped risk (if available)
        existing_keywords: Optional list of existing WHY keywords

    Returns:
        Dictionary with detection results including risk alignment
    """
    if not text or text.strip() == '':
        return {
            "explicit_why": [],
            "implicit_why": [],
            "top_match": None,
            "why_category": None,
            "score": 0,
            "risk_alignment_score": 0 if risk_description else None,
            "feedback": "No control description provided.",
            "extracted_keywords": []
        }

    # Process the text
    doc = nlp(text)

    # Default WHY keywords if none provided
    purpose_markers = existing_keywords or [
        "to ensure", "in order to", "for the purpose of", "designed to",
        "intended to", "so that", "purpose", "objective", "goal",
        "prevent", "detect", "mitigate", "risk", "error", "fraud",
        "misstatement", "compliance", "regulatory", "requirement",
        "accuracy", "completeness", "validity", "integrity"
    ]

    # Pattern-based WHY detection for explicit statements
    explicit_why_candidates = []

    # Look for purpose clauses with "to" + verb pattern
    purpose_patterns = [
        r'to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)\s+([^\.;,]*)',
        r'in\s+order\s+to\s+([^\.;,]*)',
        r'for\s+the\s+purpose\s+of\s+([^\.;,]*)',
        r'designed\s+to\s+([^\.;,]*)',
        r'intended\s+to\s+([^\.;,]*)',
        r'so\s+that\s+([^\.;,]*)'
    ]

    for pattern in purpose_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            purpose_text = match.group(0)
            explicit_why_candidates.append({
                "text": purpose_text,
                "method": "pattern_match",
                "score": 0.9,
                "span": [match.start(), match.end()],
                "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            })

    # Look for keyword-based WHY indicators
    for keyword in purpose_markers:
        if keyword.lower() in text.lower():
            # Find the position in text
            pos = text.lower().find(keyword.lower())

            # Get the sentence containing this keyword
            sentence = None
            for sent in doc.sents:
                if pos >= sent.start_char and pos < sent.end_char:
                    sentence = sent
                    break

            if sentence:
                explicit_why_candidates.append({
                    "text": sentence.text,
                    "method": "keyword_match",
                    "score": 0.7,
                    "span": [sentence.start, sentence.end],
                    "context": text[max(0, sentence.start_char - 10):min(len(text), sentence.end_char + 10)]
                })

    # Detect implicit WHY statements based on control actions
    implicit_why_candidates = []

    # Common control verbs and their implied purposes
    control_verb_purpose = {
        "review": "to ensure accuracy and completeness",
        "reconcile": "to ensure data integrity and accuracy",
        "approve": "to ensure proper authorization",
        "verify": "to confirm accuracy and validity",
        "validate": "to ensure compliance and accuracy",
        "monitor": "to detect anomalies or non-compliance",
        "check": "to identify errors or inconsistencies"
    }

    for token in doc:
        if token.lemma_.lower() in control_verb_purpose:
            # Get the complete verb phrase if possible
            verb_phrase = token.text

            # Try to get the object being acted upon
            obj_text = ""
            for child in token.children:
                if child.dep_ in ("dobj", "pobj"):
                    obj_text = child.text
                    # Try to get the full noun phrase
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            obj_text = chunk.text
                            break
                    break

            implied_purpose = control_verb_purpose[token.lemma_.lower()]

            implicit_why_candidates.append({
                "text": f"{verb_phrase} {obj_text}",
                "implied_purpose": implied_purpose,
                "method": "action_inference",
                "score": 0.5,
                "span": [token.i, token.i + len(obj_text.split()) + 1 if obj_text else 1],
                "context": f"Implied purpose from action: {implied_purpose}"
            })

    # Categorize the WHY (if found)
    categories = {
        "risk_mitigation": ["risk", "prevent", "mitigate", "reduce", "avoid", "minimize"],
        "compliance": ["comply", "compliance", "regulatory", "regulation", "requirement", "policy", "standard"],
        "accuracy": ["accuracy", "accurate", "correct", "error-free", "integrity", "reliable"],
        "completeness": ["complete", "completeness", "all", "comprehensive"],
        "authorization": ["authorize", "approval", "permission", "authorization"]
    }

    def categorize_why(why_text):
        scores = {cat: 0 for cat in categories}
        for cat, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in why_text.lower():
                    scores[cat] += 1

        # Return the category with highest score, or None if all zeros
        max_cat = max(scores.items(), key=lambda x: x[1])
        return max_cat[0] if max_cat[1] > 0 else None

    # Calculate alignment with mapped risk if provided
    risk_alignment_score = None
    risk_alignment_feedback = None

    if risk_description and (explicit_why_candidates or implicit_why_candidates):
        # Process risk description
        risk_doc = nlp(risk_description)

        # Function to calculate semantic similarity
        def calculate_similarity(text1, text2, nlp):
            doc1 = nlp(text1)
            doc2 = nlp(text2)

            # Check if both have vectors
            if not doc1.has_vector or not doc2.has_vector:
                # Fallback to keyword matching
                keywords1 = set([token.lemma_.lower() for token in doc1
                                 if not token.is_stop and not token.is_punct])
                keywords2 = set([token.lemma_.lower() for token in doc2
                                 if not token.is_stop and not token.is_punct])

                # Calculate Jaccard similarity
                if keywords1 and keywords2:
                    return len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
                return 0.0

            return doc1.similarity(doc2)

        # Get best WHY statement
        best_why = None
        if explicit_why_candidates:
            best_why = max(explicit_why_candidates, key=lambda x: x["score"])["text"]
        elif implicit_why_candidates:
            best_why = max(implicit_why_candidates, key=lambda x: x["score"])["implied_purpose"]

        if best_why:
            risk_alignment_score = calculate_similarity(best_why, risk_description, nlp)

            # Generate feedback based on alignment score
            if risk_alignment_score >= 0.7:
                risk_alignment_feedback = "Strong alignment with mapped risk."
            elif risk_alignment_score >= 0.4:
                risk_alignment_feedback = "Moderate alignment with mapped risk."
            else:
                risk_alignment_feedback = f"Weak alignment with mapped risk: '{risk_description}'. Consider clarifying how this control specifically addresses this risk."

    # Sort candidates by score
    explicit_why_candidates.sort(key=lambda x: x["score"], reverse=True)
    implicit_why_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Determine overall WHY score and top match
    if explicit_why_candidates:
        top_match = explicit_why_candidates[0]
        why_score = top_match["score"]
        why_category = categorize_why(top_match["text"])
    elif implicit_why_candidates:
        top_match = implicit_why_candidates[0]
        why_score = top_match["score"] * 0.7  # Implicit scores are discounted
        why_category = categorize_why(top_match["implied_purpose"])
    else:
        top_match = None
        why_score = 0
        why_category = None

    # Generate feedback
    if not explicit_why_candidates and not implicit_why_candidates:
        feedback = "No WHY element detected. Consider adding a clear statement of purpose or risk mitigation."
    elif not explicit_why_candidates and implicit_why_candidates:
        feedback = f"Control has an implied purpose but lacks an explicit WHY statement. Consider adding a clear statement like: '{implicit_why_candidates[0]['implied_purpose']}'."
    elif explicit_why_candidates and why_category == "risk_mitigation":
        feedback = "Control has a clear risk mitigation purpose."
    elif explicit_why_candidates:
        feedback = f"Control has a WHY statement focused on {why_category}."
    else:
        feedback = ""  # default in case none of the above hit

    # Append risk alignment feedback if available
    if risk_alignment_feedback:
        feedback = f"{feedback} {risk_alignment_feedback}".strip()

    return {
        "explicit_why": explicit_why_candidates,
        "implicit_why": implicit_why_candidates,
        "top_match": top_match,
        "why_category": why_category,
        "score": why_score,
        "risk_alignment_score": risk_alignment_score,
        "feedback": feedback,
        "extracted_keywords": [c["text"] for c in explicit_why_candidates]
    }