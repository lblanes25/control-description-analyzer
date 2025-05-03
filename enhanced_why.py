from typing import List, Dict, Any
import re


def extract_risk_aspects(risk_text: str) -> List[str]:
    """
    Extract different aspects or components from a risk description
    to enable partial matching when a risk is mitigated by multiple controls.

    Args:
        risk_text: The risk description text

    Returns:
        List of risk aspects/components
    """
    if not risk_text:
        return []

    aspects = []

    # Split complex risks with multiple components
    if " and " in risk_text:
        parts = risk_text.split(" and ")
        for part in parts:
            if len(part.split()) > 3:  # Only consider substantial phrases
                aspects.append(part.strip())

    # Look for risks with multiple impacts
    impact_markers = ["resulting in", "leading to", "causing", "which may cause", "which could result in"]
    for marker in impact_markers:
        if marker in risk_text.lower():
            parts = risk_text.lower().split(marker)
            if len(parts) > 1:
                # Add the cause
                aspects.append(parts[0].strip())
                # Add the effect/impact
                aspects.append(parts[1].strip())

    # If no structural breakdown, try sentence-based splitting
    if not aspects:
        sentences = [s.strip() for s in re.split(r'[.;]', risk_text) if len(s.strip()) > 10]
        aspects.extend(sentences)

    # If still no aspects or just one, use the entire risk as a single aspect
    if not aspects:
        aspects = [risk_text]

    return aspects


def extract_key_concepts(doc) -> List[Dict[str, Any]]:
    """
    Extract key concepts from text including actions, objects, and modifiers.
    This improves semantic understanding beyond individual words.

    Args:
        doc: spaCy doc object

    Returns:
        List of concept dictionaries
    """
    concepts = []

    # Extract action-object pairs
    for token in doc:
        # Find verbs (actions)
        if token.pos_ == "VERB":
            # Create basic action concept
            action = {
                "type": "action",
                "verb": token.lemma_,
                "text": token.text,
                "modifiers": [],
                "objects": []
            }

            # Find objects and modifiers of this verb
            for child in token.children:
                # Objects (what is being acted upon)
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    # Get the complete noun phrase if possible
                    obj_text = child.text
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            obj_text = chunk.text
                            break

                    action["objects"].append({
                        "text": obj_text,
                        "lemma": child.lemma_,
                        "has_modifiers": any(c.dep_ in ["amod", "compound"] for c in child.children)
                    })

                # Modifiers (how the action is performed)
                elif child.dep_ in ["advmod", "amod", "aux"]:
                    action["modifiers"].append({
                        "text": child.text,
                        "lemma": child.lemma_
                    })

            # Only add if it has objects or modifiers
            if action["objects"] or action["modifiers"]:
                concepts.append(action)

    # Extract negations and important modifiers
    for token in doc:
        # Find negations
        if token.dep_ == "neg" or token.text.lower() in ["without", "no", "lack", "missing"]:
            head = token.head
            concepts.append({
                "type": "negation",
                "target": head.lemma_,
                "text": f"{token.text} {head.text}",
                "negates": "approval" if "approv" in head.lemma_ else head.lemma_
            })

        # Find important attribute modifiers
        elif token.dep_ == "amod" and token.text.lower() in ["appropriate", "proper", "unauthorized", "authorized"]:
            head = token.head
            concepts.append({
                "type": "attribute",
                "modifier": token.lemma_,
                "target": head.lemma_,
                "text": f"{token.text} {head.text}"
            })

    return concepts


def identify_concept_relationships(control_concepts, risk_concepts) -> List[Dict[str, Any]]:
    """
    Identify meaningful relationships between control concepts and risk concepts

    Args:
        control_concepts: Concepts extracted from control description
        risk_concepts: Concepts extracted from risk description

    Returns:
        List of relationship dictionaries
    """
    relationships = []

    # Define relationship patterns
    patterns = [
        # Pattern 1: Control addresses a negation in risk (e.g., control has "approval" while risk has "without approval")
        {
            "control_type": "action",
            "risk_type": "negation",
            "match_fn": lambda c, r: c["verb"] == r["target"] or any(
                obj["lemma"] == r["target"] for obj in c["objects"]),
            "score": 0.9,
            "relationship": "mitigates_negation"
        },
        # Pattern 2: Control implements attribute in risk (e.g., "appropriate approval" vs "appropriate authorization")
        {
            "control_type": "attribute",
            "risk_type": "attribute",
            "match_fn": lambda c, r: c["modifier"] == r["modifier"] or c["target"] == r["target"],
            "score": 0.8,
            "relationship": "implements_attribute"
        },
        # Pattern 3: Control verb directly addresses risk verb (e.g., "prevent" vs "occur")
        {
            "control_type": "action",
            "risk_type": "action",
            "match_fn": lambda c, r: c["verb"] == r["verb"] or any(
                c["verb"] == obj["lemma"] for obj in r.get("objects", [])),
            "score": 0.7,
            "relationship": "verb_match"
        },
        # Pattern 4: Approval-authorization relationship (special case)
        {
            "special_case": "approval",
            "match_fn": lambda c, r: (("approv" in c["text"].lower() and "approv" in r["text"].lower()) or
                                      ("authoriz" in c["text"].lower() and "authoriz" in r["text"].lower())),
            "score": 0.85,
            "relationship": "approval_authorization"
        }
    ]

    # Check each control concept against each risk concept
    for c_concept in control_concepts:
        for r_concept in risk_concepts:
            # Check regular patterns
            for pattern in patterns:
                if "special_case" in pattern:
                    # Handle special cases
                    if pattern["match_fn"](c_concept, r_concept):
                        relationships.append({
                            "control_concept": c_concept,
                            "risk_concept": r_concept,
                            "relationship": pattern["relationship"],
                            "score": pattern["score"],
                            "description": f"Control {pattern['relationship']} in risk"
                        })
                elif c_concept.get("type") == pattern["control_type"] and r_concept.get("type") == pattern["risk_type"]:
                    if pattern["match_fn"](c_concept, r_concept):
                        relationships.append({
                            "control_concept": c_concept,
                            "risk_concept": r_concept,
                            "relationship": pattern["relationship"],
                            "score": pattern["score"],
                            "description": f"Control {pattern['relationship']} in risk"
                        })

    return relationships


def detect_implicit_purpose(control_actions, text, nlp):
    """
    Detect implicit purpose based on control actions and their context

    Args:
        control_actions: List of action concepts from the control
        text: The control description text
        nlp: spaCy model

    Returns:
        List of implicit purpose dictionaries
    """
    # Enhanced mapping of control verbs to implied purposes
    control_verb_purpose = {
        "review": {
            "default": "to ensure accuracy and completeness",
            "approval": "to ensure proper authorization",
            "changes": "to prevent unauthorized changes",
            "access": "to ensure appropriate access"
        },
        "approve": {
            "default": "to ensure proper authorization",
            "changes": "to prevent unauthorized changes",
            "transactions": "to ensure authorized transactions",
            "documents": "to ensure document validity"
        },
        "reconcile": {
            "default": "to ensure data integrity and accuracy",
        },
        "verify": {
            "default": "to confirm accuracy and validity",
            "approval": "to ensure proper authorization"
        },
        "validate": {
            "default": "to ensure compliance and accuracy",
            "approval": "to validate authorization"
        },
        "monitor": {
            "default": "to detect anomalies or non-compliance",
            "changes": "to identify unauthorized changes"
        },
        "check": {
            "default": "to identify errors or inconsistencies",
            "approval": "to verify authorization"
        }
    }

    implicit_purposes = []

    for action in control_actions:
        verb = action["verb"]

        if verb in control_verb_purpose:
            # Try to determine the most appropriate purpose based on objects
            purpose_key = "default"

            # Look for specific objects to refine purpose
            for obj in action["objects"]:
                obj_text = obj["text"].lower()
                for key in control_verb_purpose[verb]:
                    if key != "default" and key in obj_text:
                        purpose_key = key
                        break

            # For approval verbs with "before" or "prior to", strengthen prevention aspect
            if verb == "approve" and re.search(r"(before|prior to)\s+\w+ing", text, re.IGNORECASE):
                purpose = "to prevent unauthorized actions"
            else:
                purpose = control_verb_purpose[verb][purpose_key]

            implicit_purposes.append({
                "text": f"{verb} {' '.join([obj['text'] for obj in action['objects']])}",
                "implied_purpose": purpose,
                "score": 0.7 if purpose_key != "default" else 0.5,  # Higher score for context-specific purposes
                "context": "actions"
            })

    return implicit_purposes


def enhance_why_detection(text: str, nlp, risk_description: str = None, existing_keywords: List[str] = None,
                          control_id: str = None):
    """
    Enhanced WHY detection that evaluates alignment with mapped risks with improved
    semantic understanding and recognition of opposites in control-risk relationships.

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        risk_description: The text of the mapped risk (if available)
        existing_keywords: Optional list of existing WHY keywords
        control_id: Optional control identifier for reference in feedback

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
            "extracted_keywords": [],
            "has_success_criteria": False,
            "vague_why_phrases": [],
            "is_actual_mitigation": False,
            "control_type_mismatch": False
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
        r'to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)\s+([^\.;,]{5,50})',
        r'in\s+order\s+to\s+([^\.;,]{5,50})',
        r'for\s+the\s+purpose\s+of\s+([^\.;,]{5,50})',
        r'designed\s+to\s+([^\.;,]{5,50})',
        r'intended\s+to\s+([^\.;,]{5,50})',
        r'so\s+that\s+([^\.;,]{5,50})'
    ]

    for pattern in purpose_patterns:
        matches = list(re.finditer(pattern, text.lower()))

        for match in matches:
            purpose_text = match.group(0)

            # Only add if the match is valid (not too similar to full description)
            if len(purpose_text) / len(text) < 0.7:  # Simple validation check
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

            if sentence and len(sentence.text) / len(text) < 0.7:
                # Additional check: make sure the keyword is used as purpose indicator
                keyword_position = sentence.text.lower().find(keyword.lower())
                if keyword_position == 0 or sentence.text[keyword_position - 1] in " .,;:(":
                    explicit_why_candidates.append({
                        "text": sentence.text,
                        "method": "keyword_match",
                        "score": 0.7,
                        "span": [sentence.start, sentence.end],
                        "context": text[max(0, sentence.start_char - 10):min(len(text), sentence.end_char + 10)]
                    })

    # NEW: Extract concepts for semantic understanding
    control_doc = nlp(text.lower())
    control_concepts = extract_key_concepts(control_doc)

    # Extract implicit purposes based on control actions
    action_concepts = [c for c in control_concepts if c["type"] == "action"]
    implicit_why_candidates = detect_implicit_purpose(action_concepts, text, nlp)

    # IMPROVED: Check for temporal indicators of prevention
    # If control has "before" or "prior to", it suggests preventive action
    if re.search(r"(before|prior to)\s+\w+ing", text, re.IGNORECASE):
        prevention_score = 0.6
        implicit_why_candidates.append({
            "text": re.search(r"(before|prior to)\s+\w+ing", text, re.IGNORECASE).group(0),
            "implied_purpose": "to prevent unauthorized actions",
            "score": prevention_score,
            "context": "temporal_prevention"
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

    # IMPROVED: Risk alignment analysis
    risk_alignment_score = None
    risk_alignment_feedback = None
    risk_concepts = []

    if risk_description:
        # Process risk text
        risk_doc = nlp(risk_description.lower())
        risk_concepts = extract_key_concepts(risk_doc)

        # Extract risk aspects for partial matching
        risk_aspects = extract_risk_aspects(risk_description)

        # Identify relationships between control and risk concepts
        relationships = identify_concept_relationships(control_concepts, risk_concepts)

        # Special case for approval/changes patterns
        # "Changes are made without appropriate approval" should match with
        # "Changes to risk ratings are reviewed and approved by appropriate personnel"
        approval_terms = ["approv", "authoriz", "review"]
        change_terms = ["chang", "modif", "updat"]

        # Check for approval pattern in control
        control_has_approval = any(term in text.lower() for term in approval_terms)
        control_has_changes = any(term in text.lower() for term in change_terms)

        # Check for approval pattern in risk
        risk_has_approval = any(term in risk_description.lower() for term in approval_terms)
        risk_has_changes = any(term in risk_description.lower() for term in change_terms)

        # Check for negation pattern in risk
        risk_has_negation = "without" in risk_description.lower() or "no " in risk_description.lower()

        # Handle the specific case of approval controls addressing unauthorized changes
        if control_has_approval and control_has_changes and risk_has_approval and risk_has_changes and risk_has_negation:
            # This is a strong match - control implements approval to address unauthorized changes
            relationships.append({
                "relationship": "addresses_unauthorized_changes",
                "score": 0.95,
                "description": "Control implements approval process to address unauthorized changes"
            })

            # Add an explicit purpose candidate based on the risk
            explicit_why_candidates.append({
                "text": f"to prevent {risk_description.lower()}",
                "method": "derived_from_risk",
                "score": 0.85,
                "span": [0, len(text)],
                "context": "Derived from risk description"
            })

        # Calculate total alignment score from relationships
        if relationships:
            # Use the maximum relationship score as the base
            max_rel_score = max(rel["score"] for rel in relationships)

            # Calculate term-level alignment for more specific matching
            control_terms = set(t.lemma_ for t in control_doc if not t.is_stop and t.pos_ in ("NOUN", "VERB", "ADJ"))
            risk_terms = set(t.lemma_ for t in risk_doc if not t.is_stop and t.pos_ in ("NOUN", "VERB", "ADJ"))

            # Calculate term overlap
            if risk_terms:
                term_overlap = len(control_terms.intersection(risk_terms)) / len(risk_terms)
            else:
                term_overlap = 0

            # Combined score with higher weight on relationships
            risk_alignment_score = (0.7 * max_rel_score) + (0.3 * term_overlap)

            # Generate feedback based on alignment score and relationships
            if risk_alignment_score >= 0.7:
                # Strong alignment
                top_rel = max(relationships, key=lambda x: x["score"])
                risk_alignment_feedback = f"Strong alignment with mapped risk. Control {top_rel.get('description', 'addresses the risk directly')}."
            elif risk_alignment_score >= 0.4:
                # Moderate alignment
                risk_alignment_feedback = "Moderate alignment with mapped risk."
                if len(risk_aspects) > 1:
                    risk_alignment_feedback += f" Primarily addresses: '{risk_aspects[0]}'."
            else:
                # Weak alignment
                risk_alignment_feedback = f"Weak alignment with mapped risk: '{risk_description}'. Consider explicitly addressing how this control mitigates this specific risk."

                # Suggest improvements
                if "approval" in risk_description.lower() and "approval" not in text.lower():
                    risk_alignment_feedback += " Consider explicitly mentioning the approval process."
                elif "change" in risk_description.lower() and "change" not in text.lower():
                    risk_alignment_feedback += " Consider explicitly addressing the change management aspect."

                # Add reference to control ID if provided
                if control_id:
                    risk_alignment_feedback += f" (Control {control_id})"

    # Detect vague WHY phrases
    vague_why_terms = [
        "proper functioning", "appropriate", "adequately", "properly",
        "as needed", "as required", "as appropriate", "correct functioning",
        "effective", "efficient", "functioning", "operational", "successful",
        "appropriate action", "necessary action", "properly functioning"
    ]

    vague_why_phrases = []
    for candidate in explicit_why_candidates:
        why_text = candidate["text"].lower()
        for vague_term in vague_why_terms:
            if vague_term in why_text:
                vague_why_phrases.append({
                    "text": why_text,
                    "vague_term": vague_term,
                    "suggested_replacement": "specific risk, impact, or compliance requirement"
                })
                # Penalize score for vague phrases
                candidate["score"] *= 0.7

    # Sort candidates by score
    explicit_why_candidates.sort(key=lambda x: x["score"], reverse=True)
    implicit_why_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Determine top match and score
    if explicit_why_candidates:
        top_match = explicit_why_candidates[0]
        base_score = top_match["score"]
        why_category = categorize_why(top_match["text"])
    elif implicit_why_candidates:
        top_match = implicit_why_candidates[0]
        base_score = top_match["score"] * 0.7  # Implicit scores are discounted
        why_category = categorize_why(top_match.get("implied_purpose", ""))
    else:
        top_match = None
        base_score = 0
        why_category = None

    # If we have a strong risk alignment but no explicit WHY, boost the score
    if risk_alignment_score and risk_alignment_score > 0.7 and base_score < 0.5:
        base_score = max(base_score, 0.5)  # Minimum 0.5 score for strong risk alignment

        # Add an implicit purpose derived from the risk if none exists
        if not top_match:
            derived_purpose = f"to prevent {risk_description}"
            top_match = {
                "text": derived_purpose,
                "implied_purpose": derived_purpose,
                "method": "derived_from_risk",
                "score": 0.5
            }

    return {
        "explicit_why": explicit_why_candidates,
        "implicit_why": implicit_why_candidates,
        "top_match": top_match,
        "why_category": why_category,
        "score": base_score,
        "risk_alignment_score": risk_alignment_score,
        "feedback": risk_alignment_feedback,
        "extracted_keywords": [c["text"] for c in explicit_why_candidates],
        "has_success_criteria": any(re.search(pattern, text, re.IGNORECASE) for pattern in [
            r'(\$\d+[,\d]*|\d+\s*%|\d+\s*percent)',
            r'greater than|less than|at least|at most|minimum|maximum',
            r'threshold of|limit of|tolerance of',
            r'within \d+\s*(day|hour|minute|week|month)',
            r'criteria|criterion|standard|benchmark'
        ]),
        "vague_why_phrases": vague_why_phrases,
        "is_actual_mitigation": any(verb.lower() in text.lower() for verb in [
            "resolve", "correct", "address", "remediate", "fix", "prevent",
            "block", "stop", "deny", "restrict", "escalate", "alert",
            "notify", "disable", "lockout", "report"
        ]),
        "control_type_mismatch": False  # Simplified from original
    }