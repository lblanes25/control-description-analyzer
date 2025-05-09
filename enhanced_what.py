from typing import Dict, List, Any, Optional
import re


def enhance_what_detection(text: str, nlp, existing_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Enhanced WHAT detection with improved verb categorization, strength analysis, and context handling
    With improved filtering of timing phrases and better handling of action phrases

    Args:
        text: The control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Optional list of action keywords to consider

    Returns:
        Dictionary containing detailed analysis of action elements:
        - actions: List of detected actions with details
        - primary_action: The main action identified in the control
        - secondary_actions: Additional significant actions identified
        - score: Overall score for the WHAT element quality
        - verb_strength: Assessment of the strength of control verbs used
        - is_process: Flag indicating if this appears to be a process description rather than a control
        - voice: Dominant voice used (active/passive)
        - suggestions: Suggestions for improvement
    """

    # Helper functions defined internally to avoid reference errors
    def is_passive_construction(verb_token) -> bool:
        """Determine if a verb is in passive voice"""
        # Check for passive subject
        if any(token.dep_ == "nsubjpass" for token in verb_token.children):
            return True

        # Check for passive auxiliary (be + past participle)
        if verb_token.tag_ == "VBN":  # Past participle
            # Look for form of "be" as an auxiliary
            for ancestor in verb_token.ancestors:
                if ancestor.lemma_ == "be" and ancestor.pos_ == "AUX":
                    return True

        return False

    def is_action_ensure(verb_token) -> bool:
        """Determine if "ensure" is being used as a primary action rather than as a purpose indicator"""
        # If "ensure" is the main verb of the sentence, it might be a primary action
        if verb_token.dep_ == "ROOT":
            # Check if preceded by "to" which would indicate purpose
            prev_token = verb_token.doc[verb_token.i - 1] if verb_token.i > 0 else None
            if prev_token and prev_token.text.lower() == "to":
                return False
            return True

        # If it's not a ROOT verb, check if it's part of a purpose clause
        if verb_token.dep_ == "xcomp" or verb_token.dep_ == "advcl":
            # Check if preceded by "to"
            prev_token = verb_token.doc[verb_token.i - 1] if verb_token.i > 0 else None
            if prev_token and prev_token.text.lower() == "to":
                return False

        # Default to treating it as an action
        return True

    def get_subject(verb_token) -> Optional[str]:
        """Find the subject of a verb"""
        for token in verb_token.children:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                # Return the complete noun phrase, including any modifiers
                return " ".join(t.text for t in token.subtree)

        # If no direct subject found, look for a governing verb's subject (for compound verbs)
        if verb_token.dep_ == "xcomp" and verb_token.head.pos_ == "VERB":
            return get_subject(verb_token.head)

        return None

    def reconstruct_active_phrase(verb_token) -> str:
        """For passive verbs, reconstruct an active voice phrase from the components"""
        # Find the object (subject in passive voice)
        subj = None
        for token in verb_token.children:
            if token.dep_ == "nsubjpass":
                subj = token
                break

        if not subj:
            # If no subject found, just use the verb
            return verb_token.lemma_

        # Collect the object and any modifiers
        obj_tokens = list(subj.subtree)

        # For special cases like "access is limited", improve the verb phrasing
        if verb_token.lemma_ == "limit" and any(t.text.lower() == "access" for t in obj_tokens):
            return "limit access"

        if verb_token.lemma_ == "restrict" and any(t.text.lower() == "access" for t in obj_tokens):
            return "restrict access"

        if verb_token.lemma_ == "revoke" and any(t.text.lower() == "access" for t in obj_tokens):
            return "revoke access rights"

        if verb_token.lemma_ == "store" and any(t.text.lower() in ["item", "items"] for t in obj_tokens):
            return "store items securely"

        # Create active verb phrase: [VERB] + [OBJECT]
        active_phrase = f"{verb_token.lemma_} {' '.join(t.text for t in obj_tokens)}"

        return active_phrase

    def get_verb_phrase_improved(verb_token, problematic_verbs) -> str:
        """Extract the complete verb phrase (verb + objects) from a verb token"""
        # Start with the verb itself
        phrase_tokens = [verb_token]

        # For problematic verbs, limit the phrase to direct objects only
        if verb_token.lemma_.lower() in problematic_verbs:
            # More restrictive extraction for problematic verbs
            for token in verb_token.children:
                if token.dep_ == "dobj":  # Only direct objects
                    subtree = list(token.subtree)
                    phrase_tokens.extend(subtree)
        else:
            # Standard extraction for normal verbs
            for token in verb_token.children:
                if token.dep_ in ["dobj", "iobj", "attr", "oprd"]:
                    # Include the token and its children (to get the complete phrase)
                    subtree = list(token.subtree)
                    phrase_tokens.extend(subtree)
                elif token.dep_ == "prep":
                    # Include prepositional phrases but only if they're essential
                    # Skip non-essential prepositions that might be timing indicators
                    if token.text.lower() not in ["on", "at", "before", "after", "during", "until"]:
                        subtree = list(token.subtree)
                        phrase_tokens.extend(subtree)
                    else:
                        # For timing prepositions, check if they're essential to the meaning
                        # Fix: Check if next token exists before accessing it
                        if token.i + 1 < len(token.doc):
                            next_token = token.doc[token.i + 1]
                            if not any(next_token.text.lower().startswith(w) for w in
                                       ["a", "an", "the", "each", "every"]):
                                subtree = list(token.subtree)
                                phrase_tokens.extend(subtree)

        # Sort tokens by their position in the original text
        phrase_tokens.sort(key=lambda x: x.i)

        # Combine into a phrase
        verb_phrase = " ".join(token.text for token in phrase_tokens)

        # Remove leading timing phrases
        timing_prefixes = [
            "on a ", "on an ", "daily ", "weekly ", "monthly ", "quarterly ", "annually ", "when ",
            "as needed ", "as required ", "once ", "after ", "before ", "during "
        ]
        for prefix in timing_prefixes:
            if verb_phrase.lower().find(prefix) == 0:
                verb_phrase = verb_phrase[len(prefix):].strip()

        # Remove trailing process phrases
        process_suffixes = [
            " according to", " based on", " part of", " as part of", " per ", " in accordance with"
        ]
        for suffix in process_suffixes:
            if verb_phrase.lower().endswith(suffix):
                verb_phrase = verb_phrase[:-len(suffix)].strip()

        return verb_phrase

    def assess_phrase_completeness(verb_phrase) -> float:
        """Assess how complete a verb phrase is as a control action"""
        # Split into words
        words = verb_phrase.split()

        # Very short phrases are likely incomplete
        if len(words) < 2:
            return 0.4

        # Phrases with just prepositions at the end are incomplete
        if words[-1].lower() in ["to", "for", "with", "by", "on", "at", "in"]:
            return 0.5

        # Phrases ending with process indicators are incomplete
        process_endings = ["according", "based", "part", "accordance"]
        if words[-1].lower() in process_endings:
            return 0.5

        # Long enough phrases with a verb and object are likely complete
        if len(words) >= 3:
            return 1.0

        # Default completeness
        return 0.8

    def assess_object_specificity(verb_token) -> float:
        """Assess how specific the object of a verb is"""
        # Find direct objects
        objects = []
        for token in verb_token.children:
            if token.dep_ in ["dobj", "pobj", "attr"]:
                # Get the complete subtree
                objects.extend(list(token.subtree))

        if not objects:
            return 0.0

        # Assess specificity based on length, modifiers, and noun types
        score = min(1.0, len(objects) / 5.0)  # Longer object phrases tend to be more specific

        # Check for modifiers that make objects more specific
        specificity_indicators = ["specific", "certain", "particular", "defined", "exact"]
        if any(token.lemma_.lower() in specificity_indicators for token in objects):
            score += 0.2

        # Check for numeric modifiers (usually make things more specific)
        if any(token.pos_ == "NUM" for token in objects):
            score += 0.2

        # Check for proper nouns (usually more specific)
        if any(token.pos_ == "PROPN" for token in objects):
            score += 0.2

        # Check for technical terms (usually more specific)
        tech_terms = ["system", "application", "database", "server", "protocol", "interface",
                      "module", "component", "api", "configuration", "parameter", "threshold"]
        if any(token.lemma_.lower() in tech_terms for token in objects):
            score += 0.1

        # Check for vague generic terms that reduce specificity
        vague_object_terms = ["item", "thing", "stuff", "issue", "matter", "situation", "exception",
                              "information", "data", "content", "material", "object"]
        if any(token.lemma_.lower() in vague_object_terms for token in objects):
            score -= 0.3  # Significant penalty for vague objects

        return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1

    def is_core_control_action(verb_token, verb_lemma) -> bool:
        """Determine if a verb represents a core control action rather than a supporting process action"""
        # Core control actions are typically the main verbs that directly mitigate risk
        core_control_verbs = {
            "review", "verify", "approve", "validate", "reconcile", "check",
            "confirm", "compare", "examine", "investigate", "audit", "analyze",
            "evaluate", "assess", "test", "monitor", "inspect", "revoke",
            "disable", "remove", "age", "notify", "route", "raise", "limit", "restrict"
        }

        # Non-control verbs - explicitly mark these as NOT core actions
        non_control_verbs = {
            "use", "launch", "set", "meet", "include", "have", "sound",
            "be", "exist", "contain", "get", "put", "receive", "send", "take",
            "store", "engage", "schedule", "log", "close", "complete", "finish"
        }

        # If it's a known non-control verb, it's definitely not a core action
        if verb_lemma in non_control_verbs:
            return False

        # If it's a known core control verb, it's likely a core action
        if verb_lemma in core_control_verbs:
            # Check if the verb has an object - verbs without objects are less likely to be core actions
            has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
            if not has_object:
                return False
            return True

        # If it's the root verb of the sentence, it's more likely to be a core action
        if verb_token.dep_ == "ROOT":
            # Unless it's passive and a problematic verb
            if is_passive_construction(verb_token) and verb_lemma in ["use", "launch", "set", "meet"]:
                return False
            # Check if it has an object
            has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
            if not has_object:
                return False
            return True

        # Supporting actions are often in subordinate clauses
        if verb_token.dep_ in ["advcl", "relcl", "ccomp"]:
            return False

        # Default to False for uncertain cases
        return False

    def get_specific_alternatives(verb_lemma) -> str:
        """Provide specific alternative verb suggestions based on the context"""
        alternatives = {
            "perform": "'verify', 'examine', or 'evaluate'",
            "do": "'execute', 'conduct', or 'complete'",
            "look": "'examine', 'inspect', or 'review'",
            "check": "'verify', 'validate', or 'examine'",
            "make": "'create', 'prepare', or 'develop'",
            "handle": "'process', 'resolve', or 'manage'",
            "see": "'identify', 'recognize', or 'detect'",
            "watch": "'monitor', 'observe', or 'track'",
            "address": "'resolve', 'rectify', or 'remediate'",
            "consider": "'evaluate', 'assess', or 'analyze'",
            "manage": "'administer', 'supervise', or 'oversee'",
            "review": "'examine', 'analyze', or 'evaluate'",
            "flag": "'identify', 'mark', or 'highlight'",
            "monitor": "'track', 'supervise', or 'observe'",
            "use": "'utilize', 'apply', or 'implement'",
            "launch": "'initiate', 'deploy', or 'implement'",
            "set": "'establish', 'configure', or 'specify'",
            "age": "'classify by age', 'categorize', or 'track aging of'",
            "meet": "'achieve', 'satisfy', or 'fulfill'",
            "include": "'incorporate', 'integrate', or 'contain'",
            "sound": "'establish', 'implement', or 'maintain'",
            "raise": "'escalate', 'notify', or 'alert'",
            "store": "'secure', 'maintain', or 'place'",
            "log": "'record', 'document', or 'register'",
            "schedule": "'plan', 'arrange', or 'program'"
        }

        return alternatives.get(verb_lemma, "'verify', 'approve', or 'reconcile'")

    def extract_fallback_actions(doc, problematic_verbs):
        """Attempt to extract fallback actions when standard methods fail"""
        fallback_actions = []

        # Look for control verb phrases using pattern matching
        control_patterns = [
            (r'(notify|alert|inform)\s+([a-z\s]+)', 0.75),  # notify management, alert team
            (r'(age|categorize)\s+([a-z\s]+)', 0.7),  # age receivables, categorize items
            (r'(receive|collect)\s+([a-z\s]+)', 0.6),  # receive sign-offs, collect approvals
            (r'(limit|restrict)\s+([a-z\s]+)\s+to\s+([a-z\s]+)', 0.75),  # limit access to authorized
            (r'(route|escalate|forward)\s+([a-z\s]+)\s+to\s+([a-z\s]+)', 0.75)  # route exceptions to support
        ]

        text = doc.text.lower()

        for pattern, confidence in control_patterns:
            for match in re.finditer(pattern, text):
                verb = match.group(1)
                full_phrase = match.group(0)

                # Skip problematic verbs unless they're in a strong control context
                if verb in problematic_verbs and not any(
                        ctx in full_phrase for ctx in ["access", "approval", "review"]):
                    continue

                # Convert character span to token span for spaCy
                char_start, char_end = match.start(), match.end()
                token_start = token_end = None

                for token in doc:
                    if token.idx <= char_start < token.idx + len(token):
                        token_start = token.i
                    if token.idx < char_end <= token.idx + len(token):
                        token_end = token.i + 1  # exclusive
                        break

                if token_start is None or token_end is None:
                    # Skip if we can't properly identify token spans
                    continue

                fallback_actions.append({
                    "verb": verb,
                    "verb_lemma": verb,
                    "full_phrase": full_phrase,
                    "subject": None,
                    "is_passive": False,
                    "strength": confidence,
                    "strength_category": "medium",
                    "confidence": confidence,
                    "span": [match.start(), match.end()],
                    "sentence": "Extracted fallback action",
                    "is_core_action": True,
                    "completeness": 0.8
                })

        # Look for special case controls common in security/IT controls
        special_cases = [
            (r'(access|permission)s?\s+(?:are|is)\s+(limited|restricted)\s+to', "limit access to authorized personnel",
             0.8),
            (r'(batch\s+job)s?\s+(?:are|is)\s+(scheduled|run)', "schedule batch jobs", 0.7),
            (r'(exception)s?\s+(?:are|is)\s+(routed|sent)\s+to', "route exceptions", 0.75),
            (r'(notification)s?\s+(?:are|is)\s+(sent|delivered)\s+to', "send notifications", 0.7),
            (r'(system)\s+automatically\s+(ages)', "age receivables automatically", 0.8),
            (r'items\s+are\s+(inventoried|counted)', "inventory items", 0.7)
        ]

        for pattern, replacement, confidence in special_cases:
            match = re.search(pattern, text)
            if match:
                char_start, char_end = match.start(), match.end()
                token_start = token_end = None

                for token in doc:
                    if token.idx <= char_start < token.idx + len(token):
                        token_start = token.i
                    if token.idx < char_end <= token.idx + len(token):
                        token_end = token.i + 1  # exclusive
                        break

                # Handle cases where span was not found properly
                if token_start is None or token_end is None:
                    # Default to first token as a safe fallback
                    token_start, token_end = (0, 1) if len(doc) > 0 else (None, None)

                fallback_actions.append({
                    "verb": replacement.split()[0],
                    "verb_lemma": replacement.split()[0],
                    "full_phrase": replacement,
                    "subject": None,
                    "is_passive": False,
                    "strength": confidence,
                    "strength_category": "medium",
                    "confidence": confidence,
                    "span": [token_start, token_end] if token_start is not None else None,
                    "sentence": "Extracted special case",
                    "is_core_action": True,
                    "completeness": 0.9
                })

        return fallback_actions

    def extract_noun_phrase_actions(doc, nlp):
        """Analyze noun phrases to identify potential control actions"""
        actions = []

        # Control activity nouns that often indicate actions
        control_nouns = {
            "review": 0.8,
            "approval": 0.8,
            "validation": 0.8,
            "verification": 0.8,
            "reconciliation": 0.8,
            "assessment": 0.7,
            "monitoring": 0.7,
            "inspection": 0.7,
            "audit": 0.8,
            "analysis": 0.7,
            "confirmation": 0.7,
            "evaluation": 0.7,
            "check": 0.7,
            "authorization": 0.8,
            "testing": 0.7
        }

        # Examine noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:  # Ignore single word chunks
                head = chunk.root
                head_text = head.text.lower()

                # Check if the head is a control-related noun
                if head_text in control_nouns:
                    # Convert noun to verb form
                    verb_mapping = {
                        "review": "review",
                        "approval": "approve",
                        "validation": "validate",
                        "verification": "verify",
                        "reconciliation": "reconcile",
                        "assessment": "assess",
                        "monitoring": "monitor",
                        "inspection": "inspect",
                        "audit": "audit",
                        "analysis": "analyze",
                        "confirmation": "confirm",
                        "evaluation": "evaluate",
                        "check": "check",
                        "authorization": "authorize",
                        "testing": "test"
                    }

                    verb = verb_mapping.get(head_text, "perform")

                    # Get modifiers to form an action phrase
                    modifiers = [t.text for t in chunk if t.i != head.i]

                    # Construct the phrase: verb + modifiers
                    if modifiers:
                        modified_phrase = " ".join([verb] + modifiers)
                    else:
                        modified_phrase = f"{verb} {head_text}"

                    actions.append({
                        "verb": verb,
                        "verb_lemma": verb,
                        "full_phrase": modified_phrase,
                        "subject": None,
                        "is_passive": False,
                        "strength": control_nouns[head_text],
                        "strength_category": "medium",
                        "confidence": control_nouns[head_text] * 0.8,
                        # Slightly lower confidence for noun-derived actions
                        "span": [chunk.start, chunk.end],
                        "sentence": doc[chunk.sent.start:chunk.sent.end].text,
                        "is_core_action": True,
                        "completeness": 0.8
                    })

        return actions

    # Main function logic starts here
    if not text or text.strip() == '':
        return {
            "actions": [],
            "primary_action": None,
            "secondary_actions": [],
            "score": 0,
            "verb_strength": 0,
            "is_process": False,
            "voice": None,
            "suggestions": ["No text provided to analyze"]
        }

    try:
        # Process the text
        doc = nlp(text.lower())

        passive_voice_detected = False  # initialize flag

        # Categorize verbs by strength with default scores
        high_strength_verbs = {
            "approve": 1.0, "authorize": 1.0, "reconcile": 1.0, "validate": 1.0,
            "certify": 1.0, "sign-off": 1.0, "verify": 1.0, "confirm": 0.9,
            "test": 0.9, "enforce": 0.9, "authenticate": 0.9,
            "audit": 0.9, "inspect": 0.9, "investigate": 0.9, "scrutinize": 0.9,
            "compare": 0.9, "review": 0.85,  # Moved review up from medium
            "check": 0.85, "notify": 0.85, "route": 0.85  # Added more high-value control verbs
        }

        medium_strength_verbs = {
            "examine": 0.7, "analyze": 0.7,
            "evaluate": 0.7, "assess": 0.7, "track": 0.7, "document": 0.7,
            "record": 0.7, "maintain": 0.6, "prepare": 0.6, "generate": 0.6,
            "update": 0.6, "calculate": 0.6, "process": 0.6, "recalculate": 0.7,
            "monitor": 0.65,  # Moved up from low
            "revoke": 0.7,  # Added specific control verb
            "disable": 0.7,  # Added specific control verb
            "remove": 0.7,  # Added specific control verb
            "limit": 0.7,  # Added specific control verb
            "restrict": 0.7,  # Added specific control verb
            "age": 0.7,  # Changed from problematic to valid control verb
            "receive": 0.65,  # Added for CTRL-004
            "resolve": 0.7  # Added for exception handling
        }

        # Significantly lowered strength values for weak verbs
        low_strength_verbs = {
            "look": 0.2, "observe": 0.3,
            "view": 0.2, "consider": 0.2, "watch": 0.2, "note": 0.3,
            "see": 0.1, "handle": 0.2, "manage": 0.3, "coordinate": 0.3,
            "facilitate": 0.2, "oversee": 0.4, "run": 0.3, "perform": 0.3,
            "address": 0.2, "raise": 0.4  # Added "raise" with medium-low score
        }

        # Problematic verbs that are often not control actions, or are process descriptors
        problematic_verbs = {
            "use": 0.1, "launch": 0.1, "set": 0.1, "meet": 0.1, "include": 0.1,
            "have": 0.1, "be": 0.1, "exist": 0.1, "contain": 0.1, "sound": 0.1,
            "store": 0.2, "engage": 0.2, "schedule": 0.2, "used": 0.1, "log": 0.3
        }

        # Combine and extend with problematic verbs
        low_strength_verbs.update(problematic_verbs)

        # Combine and extend with existing keywords if provided
        all_verbs = {**high_strength_verbs, **medium_strength_verbs, **low_strength_verbs}

        if existing_keywords:
            # Add any missing keywords with a default medium score
            for kw in existing_keywords:
                if kw.lower() not in all_verbs:
                    all_verbs[kw.lower()] = 0.6

        # ----------------
        # Timing phrase detection to eliminate WHEN segments from WHAT
        # Expanded pattern list
        # ----------------
        timing_phrases = [
            r'on\s+an?\s+[a-z-]+\s+basis',  # on a monthly basis, on an ad-hoc basis
            r'daily|weekly|monthly|quarterly|annually|yearly',
            r'each\s+(day|week|month|quarter|year)',
            r'every\s+(day|week|month|quarter|year)',
            r'when\s+(necessary|needed|required|appropriate)',
            r'as\s+(needed|required|appropriate)',
            r'once\s+[a-z]+',  # once completed, once approved
            r'prior\s+to',
            r'subsequent\s+to',
            r'following\s+the',
            r'after\s+[a-z]+',
            r'before\s+[a-z]+',
            r'during\s+[a-z]+',
            r'\d+\s+(days?|weeks?|months?)',  # 30 days, 2 weeks
            r'at\s+(the\s+)?(beginning|end|close|start)',
            r'(day|week|month|quarter|year)[\s-]end',
            r'nightly',
            r'timely'
        ]

        # Find all timing phrases to exclude from WHAT candidates
        timing_spans = []
        for pattern in timing_phrases:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                timing_spans.append((match.start(), match.end()))

        # ----------------
        # Process descriptor phrases - patterns that indicate process description, not control actions
        # ----------------
        process_phrases = [
            r'has\s+processes?',
            r'have\s+processes?',
            r'are\s+used\s+for',
            r'is\s+used\s+for',
            r'are\s+stored',
            r'is\s+stored',
            r'are\s+included',
            r'is\s+included',
            r'according\s+to',
            r'based\s+on',
            r'part\s+of',
            r'component\s+of',
            r'element\s+of',
            r'such\s+as',
            r'includes',
            r'include',
            r'included',
            r'have\s+been',
            r'has\s+been',
            r'are\s+subject\s+to',
            r'is\s+subject\s+to'
        ]

        # Find all process phrases to exclude from WHAT candidates
        process_spans = []
        for pattern in process_phrases:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                process_spans.append((match.start(), match.end()))

        # ----------------
        # Purpose phrase detection to identify WHY segments and exclude from WHAT
        # ----------------
        purpose_phrases = [
            r'to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)',
            r'in\s+order\s+to\s+[^\.;,]*',
            r'for\s+the\s+purpose\s+of\s+[^\.;,]*',
            r'designed\s+to\s+[^\.;,]*',
            r'intended\s+to\s+[^\.;,]*',
            r'so\s+that\s+[^\.;,]*',
            r'to\s+ensure\s+[^\.;,]*'
        ]

        # Find all purpose phrases to exclude from WHAT candidates
        purpose_spans = []
        for pattern in purpose_phrases:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                purpose_spans.append((match.start(), match.end()))

        # Action detection logic
        actions = []
        process_indicators = 0
        active_voice_count = 0
        passive_voice_count = 0
        sequence_markers = ["then", "after", "before", "next", "subsequently", "finally", "lastly", "following"]

        # Count sequence markers as process indicators
        for marker in sequence_markers:
            if marker in text.lower():
                process_indicators += 1

        # Detect actions using dependency parsing
        for sent in doc.sents:
            # Find all verbs in the sentence
            for token in sent:
                # Check if token is a verb
                if token.pos_ == "VERB":
                    # Get the lemmatized form of the verb
                    verb_lemma = token.lemma_.lower()

                    # Skip if this verb is part of a timing phrase (WHEN not WHAT)
                    if any(token.idx >= start and token.idx < end for start, end in timing_spans):
                        continue

                    # Skip if this verb is part of a process phrase (not a control action)
                    if any(token.idx >= start and token.idx < end for start, end in process_spans):
                        continue

                    # Skip if this verb is part of a purpose phrase (WHY not WHAT)
                    if any(token.idx >= start and token.idx < end for start, end in purpose_spans):
                        continue

                    # Skip auxiliary verbs and common verbs that don't represent control actions
                    if verb_lemma in ["be", "have", "do", "can", "could", "would", "should", "may", "might"]:
                        continue

                    # Skip "ensure" when used as a standalone verb, as it's typically WHY not WHAT
                    if verb_lemma == "ensure" and not is_action_ensure(token):
                        continue

                    # Skip problematic verbs when in passive voice or subordinate clauses
                    if verb_lemma in problematic_verbs and (is_passive_construction(token) or token.dep_ != "ROOT"):
                        continue

                    # Get the verb phrase with improved extraction that reconstructs proper SVO order
                    is_passive = is_passive_construction(token)
                    if is_passive:
                        verb_phrase = reconstruct_active_phrase(token)
                    else:
                        verb_phrase = get_verb_phrase_improved(token, problematic_verbs)

                    # Skip if the verb phrase is empty after processing or too short
                    if not verb_phrase or len(verb_phrase.split()) < 2:
                        continue

                    # Skip if the verb phrase starts with a timing phrase
                    timing_start_patterns = [
                        "on a", "on an", "daily", "weekly", "monthly", "quarterly", "annually",
                        "when", "as needed", "as required", "once", "after", "before", "during"
                    ]
                    if any(verb_phrase.lower().find(pattern) == 0 for pattern in timing_start_patterns):
                        continue

                    # Skip phrases that are process descriptions rather than actions
                    process_patterns = [
                        "has process", "have process", "according to", "based on", "part of",
                        "are used", "is used", "are stored", "is stored", "are included", "is included",
                        "includes", "include", "included", "such as", "have been", "has been"
                    ]
                    if any(pattern in verb_phrase.lower() for pattern in process_patterns):
                        continue

                    if is_passive:
                        passive_voice_detected = True
                        passive_voice_count += 1
                    else:
                        active_voice_count += 1

                    # Get the subject of the action
                    subject = get_subject(token)

                    # Determine verb strength
                    verb_strength = 0.5  # Default medium strength
                    verb_category = "medium"

                    if verb_lemma in high_strength_verbs:
                        verb_strength = high_strength_verbs[verb_lemma]
                        verb_category = "high"
                    elif verb_lemma in medium_strength_verbs:
                        verb_strength = medium_strength_verbs[verb_lemma]
                        verb_category = "medium"
                    elif verb_lemma in low_strength_verbs:
                        verb_strength = low_strength_verbs[verb_lemma]
                        verb_category = "low"

                    # Calculate confidence score
                    # Factors affecting confidence:
                    # 1. Verb strength
                    # 2. Voice (prefer active)
                    # 3. Subject clarity
                    # 4. Position in the text (earlier = more important)
                    # 5. Object specificity
                    # 6. Vague term penalty
                    # 7. Phrase completeness
                    confidence = verb_strength

                    # Adjust for voice - higher penalty for passive problematic verbs
                    if is_passive:
                        if verb_lemma in problematic_verbs:
                            confidence *= 0.2  # Severe penalty for passive problematic verbs
                        else:
                            confidence *= 0.4  # Increased penalty for passive voice (was 0.8)

                    # Adjust for subject clarity
                    if subject:
                        confidence *= 1.1

                    # Adjust for position (normalize by text length)
                    position_factor = 1.0 - (token.i / len(doc)) * 0.2
                    confidence *= position_factor

                    # Adjust for object specificity
                    object_specificity = assess_object_specificity(token)
                    confidence *= (1.0 + object_specificity * 0.2)

                    # Adjust for phrase completeness - penalize incomplete phrases
                    phrase_completeness = assess_phrase_completeness(verb_phrase)
                    confidence *= phrase_completeness

                    # Apply vague term penalty
                    # Check sentence for vague terms like "as needed", "when appropriate"
                    vague_terms = ["as needed", "when necessary", "as appropriate",
                                   "when appropriate", "as required", "if needed"]
                    sent_text = sent.text.lower()

                    if any(term in sent_text for term in vague_terms):
                        confidence *= 0.7  # Strong penalty for vague temporal terms

                    # Apply ROOT boost - primary verbs in the sentence are more likely to be control actions
                    if token.dep_ == "ROOT":
                        confidence *= 1.2

                    # Cap confidence at 1.0
                    confidence = min(1.0, confidence)

                    # Add to actions
                    actions.append({
                        "verb": token.text,
                        "verb_lemma": verb_lemma,
                        "full_phrase": verb_phrase,
                        "subject": subject,
                        "is_passive": is_passive,
                        "strength": verb_strength,
                        "strength_category": verb_category,
                        "confidence": confidence,
                        "span": [token.i, token.i + 1],
                        "sentence": sent.text,
                        "is_core_action": is_core_control_action(token, verb_lemma),
                        "completeness": phrase_completeness
                    })

        # If no actions found, try to extract fallback actions using more relaxed criteria
        if not actions:
            fallback_actions = extract_fallback_actions(doc, problematic_verbs)
            actions.extend(fallback_actions)

        # Filter out problematic action phrases with enhanced filtering
        filtered_actions = []
        for action in actions:
            # Skip timing phrases
            if any(action["full_phrase"].lower().find(pattern) == 0 for pattern in [
                "on a", "on an", "when", "as needed", "daily", "weekly", "monthly",
                "quarterly", "annually", "once", "after", "before", "during"
            ]):
                continue

            # Skip process description phrases
            if any(pattern in action["full_phrase"].lower() for pattern in [
                "has process", "have process", "according to", "based on", "part of",
                "are used", "is used", "are stored", "is stored", "are included", "is included",
                "includes", "include", "included", "such as", "have been", "has been"
            ]):
                continue

            # Skip phrases with completeness score below threshold (unless we have no other actions)
            if len(actions) > 1 and action["completeness"] < 0.5:
                continue

            # Skip passive problematic verbs with low confidence
            if action["is_passive"] and action["verb_lemma"] in problematic_verbs and action[
                "confidence"] < 0.4:
                continue

            # Remove any timing words at the beginning of the phrase
            full_phrase = action["full_phrase"]
            timing_words = ["once", "daily", "weekly", "monthly", "quarterly", "annually", "when", "as"]
            words = full_phrase.split()
            if words and words[0].lower() in timing_words:
                action["full_phrase"] = " ".join(words[1:])

            filtered_actions.append(action)

        # Use filtered actions
        actions = filtered_actions

        # If still no actions after filtering, try noun phrase analysis
        if not actions:
            noun_phrase_actions = extract_noun_phrase_actions(doc, nlp)
            actions.extend(noun_phrase_actions)

        # Sort actions by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        # Filter for core control actions to identify primary and secondary
        core_actions = [a for a in actions if a.get("is_core_action", False)]

        # If no core actions found, fall back to all actions
        if not core_actions and actions:
            core_actions = actions

        # Identify primary and secondary actions
        primary_action = core_actions[0] if core_actions else None
        secondary_actions = core_actions[1:3] if len(core_actions) > 1 else []

        # Calculate verb strength metrics
        avg_verb_strength = sum(a["strength"] for a in actions) / len(actions) if actions else 0

        # Determine if this is describing a process rather than a single control
        # Indicators:
        # 1. More than 3 distinct action verbs
        # 2. Presence of sequence markers
        # 3. Length of text (very long descriptions often describe processes)
        distinct_verbs = set(a["verb_lemma"] for a in actions)
        if len(distinct_verbs) > 3:
            process_indicators += 2

        if len(text.split()) > 50:  # If more than 50 words
            process_indicators += 1

        is_process = process_indicators >= 3

        # Determine dominant voice
        if active_voice_count > passive_voice_count:
            dominant_voice = "active"
        elif passive_voice_count > active_voice_count:
            dominant_voice = "passive"
        else:
            dominant_voice = "mixed" if active_voice_count > 0 else "unknown"

        # Calculate final score (0-1 scale)
        # Components:
        # 1. Quality of primary action (50%)
        # 2. Average verb strength (25%)
        # 3. Voice preference (active preferred) (15%)
        # 4. Process vs. control clarity (10%)
        primary_action_score = primary_action["confidence"] if primary_action else 0
        verb_strength_score = avg_verb_strength

        voice_score = 0.8 if dominant_voice == "active" else 0.6 if dominant_voice == "mixed" else 0.4
        process_score = 0.9 if not is_process else 0.3

        final_score = (
                primary_action_score * 0.5 +
                verb_strength_score * 0.25 +
                voice_score * 0.15 +
                process_score * 0.1
        )

        # Generate suggestions
        suggestions = []

        if primary_action and primary_action["strength_category"] == "low":
            specific_alts = get_specific_alternatives(primary_action["verb_lemma"])
            suggestions.append(
                f"Replace weak verb '{primary_action['verb']}' with a stronger control verb like {specific_alts}")

        if dominant_voice == "passive":
            suggestions.append("Consider using active voice to clearly indicate who performs the control")

        if is_process:
            suggestions.append(
                "This appears to describe a process rather than a specific control action; consider breaking into separate controls")

        if not primary_action:
            suggestions.append(
                "No clear control action detected; add a specific verb describing what the control does")

            # Suggestion for vague objects - FIX THE PROBLEMATIC LINE HERE
            # Add safety checks for the span index
            if primary_action and "span" in primary_action and len(primary_action["span"]) > 0:
                # Ensure the span index is within the document bounds
                span_idx = primary_action["span"][0]
                if 0 <= span_idx < len(doc):
                    # Now it's safe to access the token
                    if assess_object_specificity(doc[span_idx]) < 0.5:
                        suggestions.append(
                            f"Consider clarifying the object of '{primary_action['verb_lemma']}' to be more specific.")
                else:
                    # Add a more generic suggestion if we can't access the specific token
                    suggestions.append(
                        f"Consider clarifying the object of '{primary_action['verb_lemma']}' to be more specific.")

            if passive_voice_detected:
                suggestions.append(
                    "Consider using active voice to clearly indicate responsibility for control activities.")

            return {
                "actions": actions,
                "primary_action": primary_action,
                "secondary_actions": secondary_actions,
                "score": final_score,
                "verb_strength": avg_verb_strength,
                "is_process": is_process,
                "voice": dominant_voice,
                "suggestions": suggestions
            }

    except IndexError as e:
        print(f"IndexError in WHAT detection: {e}")
        print(f"Problematic text: '{text}'")
        # Return a valid result instead of re-raising the error
        return {
            "actions": actions if 'actions' in locals() else [],
            "primary_action": primary_action if 'primary_action' in locals() else None,
            "secondary_actions": secondary_actions if 'secondary_actions' in locals() else [],
            "score": final_score if 'final_score' in locals() else 0.0,
            "verb_strength": avg_verb_strength if 'avg_verb_strength' in locals() else 0.0,
            "is_process": is_process if 'is_process' in locals() else False,
            "voice": dominant_voice if 'dominant_voice' in locals() else "unknown",
            "suggestions": ["Error analyzing control action. Please check the control description format."]
        }

    except Exception as e:
        print(f"Error in WHAT detection: {str(e)}")
        # Return default empty results on error
        return {
            "actions": [],
            "primary_action": None,
            "secondary_actions": [],
            "score": 0,
            "verb_strength": 0,
            "is_process": False,
            "voice": None,
            "suggestions": [f"Error analyzing text: {str(e)}"]
        }


def mark_possible_standalone_controls(description: str, nlp) -> List[Dict[str, Any]]:
    """
    Analyze a potentially multi-control description and mark potential standalone controls
    Improved to identify controls that should be separated

    Args:
        description: The control description text to analyze
        nlp: spaCy NLP model

    Returns:
        List of potential standalone controls with their spans and scores
    """
    doc = nlp(description)
    potential_controls = []

    # Split by common sentence separators
    sentences = list(doc.sents)

    # Analyze each sentence
    for i, sent in enumerate(sentences):
        try:
            # Analyze this sentence
            result = enhance_what_detection(sent.text, nlp)

            # Check if result is None before trying to access its properties
            if result is None:
                continue

            # If it has a clear action, it might be a separate control
            if result["primary_action"] and result["score"] > 0.5 and not result["is_process"]:
                potential_controls.append({
                    "text": sent.text,
                    "span": [sent.start, sent.end],
                    "score": result["score"],
                    "action": result["primary_action"]["full_phrase"] if result["primary_action"] else None,
                    "control_type": determine_likely_control_type(sent.text)
                })
        except Exception as e:
            print(f"Error analyzing sentence {i}: {e}")
            continue

    # Also look for controls that might be in the same sentence
    # Split long sentences with multiple verbs and conjunctions
    if len(sentences) == 1 and len(doc) > 20:  # If only one long sentence
        try:
            compound_control_candidates = identify_compound_controls(doc)
            if compound_control_candidates:
                potential_controls.extend(compound_control_candidates)
        except Exception as e:
            print(f"Error identifying compound controls: {e}")

    return potential_controls

def determine_likely_control_type(text: str) -> str:
    """
    Determine the likely control type based on text analysis

    Args:
        text: Control description text

    Returns:
        Likely control type: "preventive", "detective", "corrective", or "unknown"
    """
    text_lower = text.lower()

    # Preventive control indicators
    if any(term in text_lower for term in ["prevent", "block", "stop", "prohibit", "restrict", "limit"]):
        return "preventive"

    # Detective control indicators
    if any(term in text_lower for term in
           ["detect", "identify", "discover", "find", "monitor", "review", "reconcile"]):
        return "detective"

    # Corrective control indicators
    if any(term in text_lower for term in ["correct", "resolve", "address", "fix", "remediate", "rectify"]):
        return "corrective"

    return "unknown"

def identify_compound_controls(doc) -> List[Dict[str, Any]]:
    """
    Identify potential compound controls within a single sentence

    Args:
        doc: spaCy document

    Returns:
        List of potential standalone controls
    """
    compound_controls = []

    # Look for clauses connected by coordinating conjunctions
    for token in doc:
        if token.pos_ == "CCONJ" and token.text.lower() in ["and", "or"]:
            # Check if connecting two verb phrases
            left_verbs = [t for t in doc[:token.i] if t.pos_ == "VERB"]
            right_verbs = [t for t in doc[token.i + 1:] if t.pos_ == "VERB"]

            if left_verbs and right_verbs:
                # Extract left and right clauses
                left_verb = left_verbs[-1]  # Last verb before conjunction
                right_verb = right_verbs[0]  # First verb after conjunction

                # Get rough clause spans
                left_span_end = token.i
                left_span_start = max(0, left_verb.i - 5)  # Approximate subject+verb start
                right_span_start = token.i + 1
                right_span_end = min(len(doc), right_verb.i + 5)  # Approximate verb+object end

                # Create potential control entries
                left_text = doc[left_span_start:left_span_end].text
                right_text = doc[right_span_start:right_span_end].text

                # Only add if they have sufficient length
                if len(left_text.split()) > 3:
                    compound_controls.append({
                        "text": left_text,
                        "span": [left_span_start, left_span_end],
                        "score": 0.6,  # Lower confidence due to extraction method
                        "action": left_verb.text,
                        "control_type": determine_likely_control_type(left_text)
                    })

                if len(right_text.split()) > 3:
                    compound_controls.append({
                        "text": right_text,
                        "span": [right_span_start, right_span_end],
                        "score": 0.6,  # Lower confidence due to extraction method
                        "action": right_verb.text,
                        "control_type": determine_likely_control_type(right_text)
                    })

    return compound_controls