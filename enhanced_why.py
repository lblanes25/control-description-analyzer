import spacy
from typing import Dict, List, Any, Optional, Tuple, Union
import re
from collections import Counter


def enhance_why_detection(text: str, nlp, risk_description: str = None, existing_keywords: List[str] = None):
    """
    Enhanced WHY detection that evaluates alignment with mapped risks and suggests improvements

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        risk_description: The text of the mapped risk (if available)
        existing_keywords: Optional list of existing WHY keywords

    Returns:
        Dictionary with detection results including risk alignment and suggestions
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
            "extracted_keywords": [],
            "suggested_why_statements": []
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

    # Enhanced control verbs and their implied purposes
    control_verb_purpose = {
        "review": "to ensure accuracy and completeness",
        "reconcile": "to ensure data integrity and accuracy",
        "approve": "to ensure proper authorization",
        "verify": "to confirm accuracy and validity",
        "validate": "to ensure compliance and accuracy",
        "monitor": "to detect anomalies or non-compliance",
        "check": "to identify errors or inconsistencies",
        "compare": "to verify consistency and identify discrepancies",
        "evaluate": "to assess effectiveness and compliance",
        "examine": "to identify potential issues or errors",
        "analyze": "to ensure correctness and identify patterns",
        "audit": "to ensure compliance with policies and regulations",
        "inspect": "to verify adherence to standards and requirements",
        "test": "to confirm proper functioning and identify weaknesses",
        "confirm": "to validate accuracy and completeness",
        "document": "to maintain evidence of control execution",
        "maintain": "to ensure ongoing compliance and accuracy",
        "track": "to ensure completeness and identify anomalies",
        "sign": "to provide evidence of review and approval",
        "authenticate": "to verify identity and authorization",
        "restrict": "to prevent unauthorized access or actions",
        "limit": "to reduce risk of error or fraud"
    }

    # Track all objects of control actions to inform semantic WHY statement generation
    all_action_objects = []

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
                            all_action_objects.append(obj_text)
                            break
                    break

            implied_purpose = control_verb_purpose[token.lemma_.lower()]

            implicit_why_candidates.append({
                "text": f"{verb_phrase} {obj_text}",
                "verb": token.lemma_.lower(),
                "object": obj_text,
                "implied_purpose": implied_purpose,
                "method": "action_inference",
                "score": 0.5,
                "span": [token.i, token.i + len(obj_text.split()) + 1 if obj_text else 1],
                "context": f"Implied purpose from action: {implied_purpose}"
            })

    # Categorize the WHY (if found)
    categories = {
        "risk_mitigation": ["risk", "prevent", "mitigate", "reduce", "avoid", "minimize", "protection", "safeguard"],
        "compliance": ["comply", "compliance", "regulatory", "regulation", "requirement", "policy", "standard", "law",
                       "rule", "guideline"],
        "accuracy": ["accuracy", "accurate", "correct", "error-free", "integrity", "reliable", "validity", "precision"],
        "completeness": ["complete", "completeness", "all", "comprehensive", "exhaustive", "thorough"],
        "authorization": ["authorize", "approval", "permission", "authorization", "access", "authentication"],
        "security": ["secure", "security", "protection", "confidentiality", "privacy", "safeguard"]
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

    # NEW: Improved risk alignment with mapped risk if provided
    risk_alignment_score = None
    risk_alignment_feedback = None
    risk_key_concepts = []

    if risk_description and len(risk_description.strip()) > 0:
        # Process risk description to extract key concepts
        risk_doc = nlp(risk_description.lower())

        # Extract key nouns and verbs from risk description (not stopwords or punctuation)
        risk_key_concepts = [token.lemma_ for token in risk_doc
                             if (token.pos_ in ('NOUN', 'VERB', 'ADJ')
                                 and not token.is_stop and not token.is_punct)]

        # Function to calculate semantic similarity with improved specificity
        def calculate_enhanced_similarity(text1, text2, nlp):
            try:
                doc1 = nlp(text1.lower())
                doc2 = nlp(text2.lower())

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

                # Weight similarity by importance of matching terms
                # Check for key risk terms appearing in why statement
                key_term_matches = 0
                for token in doc1:
                    if token.lemma_.lower() in risk_key_concepts:
                        key_term_matches += 1

                # Blend vector similarity with key term matches
                vector_similarity = doc1.similarity(doc2)
                key_term_factor = min(1.0, key_term_matches / max(1, len(risk_key_concepts)))

                # Combined score (70% vector similarity, 30% key term matches)
                return (0.7 * vector_similarity) + (0.3 * key_term_factor)
            except Exception as e:
                print(f"Error in similarity calculation: {str(e)}")
                return 0.0

        # Get best WHY statement
        best_why = None
        best_why_score = 0
        all_why_statements = explicit_why_candidates + implicit_why_candidates

        for why_stmt in all_why_statements:
            why_text = why_stmt.get("text", "")
            if why_stmt.get("method") == "action_inference":
                why_text = why_stmt.get("implied_purpose", "")

            stmt_score = calculate_enhanced_similarity(why_text, risk_description, nlp)
            if stmt_score > best_why_score:
                best_why = why_text
                best_why_score = stmt_score

        if best_why:
            risk_alignment_score = best_why_score

            # Generate feedback based on alignment score
            if risk_alignment_score >= 0.7:
                risk_alignment_feedback = "Strong alignment with mapped risk."
            elif risk_alignment_score >= 0.4:
                risk_alignment_feedback = "Moderate alignment with mapped risk."
            else:
                risk_alignment_feedback = f"Weak alignment with mapped risk. Consider clarifying how this control specifically addresses the risk: '{risk_description}'."

    # NEW: Generate suggested WHY statements based on context
    suggested_why_statements = []

    if not explicit_why_candidates and implicit_why_candidates:
        # Build suggestions based on the implied purposes
        top_actions = sorted(implicit_why_candidates, key=lambda x: x["score"], reverse=True)[:2]

        for action in top_actions:
            # Generate a more specific WHY statement with the verb and object
            verb = action.get("verb", "")
            obj = action.get("object", "")
            implied = action.get("implied_purpose", "")

            if verb and obj:
                # Base suggestion on the control verb-object pair
                suggestion = f"This control is designed {implied}"

                # If we have a mapped risk, enhance with risk terminology
                if risk_key_concepts:
                    # Find relevant risk concepts that might apply to this control
                    relevant_risk_terms = []
                    obj_doc = nlp(obj.lower())
                    for concept in risk_key_concepts:
                        concept_doc = nlp(concept)
                        if concept_doc.has_vector and obj_doc.has_vector:
                            similarity = concept_doc.similarity(obj_doc)
                            if similarity > 0.4:  # Threshold for relevance
                                relevant_risk_terms.append(concept)

                    # If we found relevant terms, incorporate them
                    if relevant_risk_terms:
                        # Take top 2 most relevant terms
                        top_terms = relevant_risk_terms[:2]
                        risk_phrase = " and ".join(top_terms)
                        suggestion = f"This control is designed {implied} and mitigate risks related to {risk_phrase}"

                suggested_why_statements.append(suggestion)

    # NEW: Add risk-based WHY suggestions if we have a risk description but no WHY statements
    if risk_description and not explicit_why_candidates and not suggested_why_statements:
        risk_doc = nlp(risk_description.lower())

        # Extract potential risk verbs and objects
        risk_verbs = [token.text for token in risk_doc if token.pos_ == "VERB" and not token.is_stop]
        risk_nouns = [token.text for token in risk_doc if token.pos_ == "NOUN" and not token.is_stop]

        # Create prevention and detection focused suggestions
        if risk_verbs and risk_nouns:
            # Use the most common verb and noun for suggestions
            verb_counts = Counter(risk_verbs)
            noun_counts = Counter(risk_nouns)

            top_verb = verb_counts.most_common(1)[0][0] if verb_counts else "occurring"
            top_noun = noun_counts.most_common(1)[0][0] if noun_counts else "issues"

            prevention = f"This control is designed to prevent {top_noun} from {top_verb}"
            detection = f"This control is designed to detect and address {top_noun} if they occur"

            suggested_why_statements.extend([prevention, detection])

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

    # NEW: Enhanced feedback generation
    if not explicit_why_candidates and not implicit_why_candidates:
        feedback = "No WHY element detected. Add a clear statement of purpose using 'to ensure,' 'in order to,' or similar phrasing."
        if suggested_why_statements:
            feedback += f" Consider adding: '{suggested_why_statements[0]}'"
    elif not explicit_why_candidates and implicit_why_candidates:
        # More specific feedback based on control type
        if why_category:
            feedback = f"Control has an implied {why_category} purpose but lacks an explicit WHY statement. Consider adding: '{suggested_why_statements[0] if suggested_why_statements else implicit_why_candidates[0]['implied_purpose']}'"
        else:
            feedback = f"Control has an implied purpose but lacks an explicit WHY statement. Consider adding: '{suggested_why_statements[0] if suggested_why_statements else implicit_why_candidates[0]['implied_purpose']}'"
    elif explicit_why_candidates and why_category == "risk_mitigation":
        feedback = "Control has a clear risk mitigation purpose."

        # Check if the purpose is specific enough
        if len(explicit_why_candidates[0]["text"].split()) < 8:
            feedback += " Consider elaborating on which specific risks are being mitigated."
    elif explicit_why_candidates:
        feedback = f"Control has a WHY statement focused on {why_category}."

        # Additional feedback for compliance controls
        if why_category == "compliance" and "regulation" not in explicit_why_candidates[0]["text"].lower():
            feedback += " Consider specifying which regulations or policies this control supports."
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
        "extracted_keywords": [c["text"] for c in explicit_why_candidates],
        "suggested_why_statements": suggested_why_statements
    }


def extract_risk_themes(risk_description, nlp):
    """
    Extract key themes from a risk description to aid in WHY alignment

    Args:
        risk_description: Text description of the risk
        nlp: spaCy NLP model

    Returns:
        Dictionary with key themes and concepts from the risk
    """
    if not risk_description or len(risk_description.strip()) == 0:
        return {"themes": [], "key_terms": []}

    # Process the risk description
    doc = nlp(risk_description.lower())

    # Extract key terms (nouns, verbs, adjectives)
    key_terms = [token.lemma_ for token in doc
                 if (token.pos_ in ('NOUN', 'VERB', 'ADJ')
                     and not token.is_stop and not token.is_punct)]

    # Extract noun phrases as potential risk themes
    noun_phrases = [chunk.text for chunk in doc.noun_chunks
                    if not all(token.is_stop for token in chunk)]

    # Count term frequencies to identify main themes
    term_freq = Counter(key_terms)
    top_terms = [term for term, freq in term_freq.most_common(5)]

    # Identify top themes (longer noun phrases that contain top terms)
    themes = []
    for phrase in noun_phrases:
        if any(term in phrase for term in top_terms):
            themes.append(phrase)

    # Take top 3 most representative themes
    top_themes = themes[:3] if themes else []

    return {
        "themes": top_themes,
        "key_terms": top_terms
    }


def generate_why_statement(control_actions, risk_themes=None):
    """
    Generate a WHY statement based on control actions and optional risk themes

    Args:
        control_actions: List of control action dictionaries
        risk_themes: Optional dictionary of risk themes and terms

    Returns:
        List of suggested WHY statements
    """
    if not control_actions:
        return []

    suggestions = []

    # Basic control purpose templates
    templates = [
        "This control is designed to {purpose}",
        "The objective of this control is to {purpose}",
        "This control exists to {purpose}"
    ]

    # Get the top control action
    top_action = max(control_actions, key=lambda x: x.get("score", 0))
    purpose = top_action.get("implied_purpose", "").strip()

    if not purpose.startswith("to "):
        purpose = "to " + purpose

    # Generate basic suggestion
    base_suggestion = templates[0].format(purpose=purpose)
    suggestions.append(base_suggestion)

    # If we have risk themes, create more specific suggestions
    if risk_themes and risk_themes.get("themes"):
        top_theme = risk_themes["themes"][0] if risk_themes["themes"] else ""
        if top_theme:
            risk_suggestion = f"This control is designed to {purpose} and address risks related to {top_theme}"
            suggestions.append(risk_suggestion)

    return suggestions