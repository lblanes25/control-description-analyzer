"""
Enhanced WHAT Detection Module - COMPLETE FIXED VERSION

This module implements a sophisticated detection system for the WHAT element in control descriptions.
It identifies actions being performed in controls using a layered approach of detection methods
with explicit fallback strategies, and also handles WHERE components within action phrases.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import re


class WhatDetectionConfig:
    """Configuration object for WHAT detection parameters"""

    def __init__(self, spacy_doc, existing_keywords: Optional[List[str]] = None,
                 control_type: Optional[str] = None, config: Optional[Dict] = None):
        self.spacy_doc = spacy_doc
        self.existing_keywords = existing_keywords or []
        self.control_type = control_type
        self.config = config or {}
        self.debug_mode = self._get_config_value("debug_mode", False)

    def _get_config_value(self, key: str, default_value: Any) -> Any:
        """Helper function to safely get configuration values with defaults"""
        if not self.config:
            return default_value

        what_config = self.config.get("elements", {}).get("WHAT", {})
        if key in what_config:
            return what_config[key]

        return self.config.get(key, default_value)


class VerbAnalyzer:
    """Handles verb categorization and strength analysis"""

    def __init__(self, config: WhatDetectionConfig):
        self.config = config
        self.verb_categories = self._setup_verb_categories()
        # Flatten verb categories for easier lookup
        self.all_verbs = {}
        for category, verbs in self.verb_categories.items():
            self.all_verbs.update(verbs)

    def _setup_verb_categories(self) -> Dict[str, Dict[str, float]]:
        """Load verb strength categories from config with fallbacks to default values"""
        default_categories = {
            "high_strength_verbs": {
                "approve": 1.0, "authorize": 1.0, "reconcile": 1.0, "validate": 1.0,
                "certify": 1.0, "verify": 1.0, "confirm": 0.9, "test": 0.9, "enforce": 0.9,
                "authenticate": 0.9, "audit": 0.9, "inspect": 0.9, "review": 0.85,
                "check": 0.85, "notify": 0.85, "route": 0.85, "complete": 0.9, "conduct": 0.9
            },
            "medium_strength_verbs": {
                "examine": 0.7, "analyze": 0.7, "evaluate": 0.7, "assess": 0.7,
                "track": 0.7, "document": 0.7, "record": 0.7, "maintain": 0.6,
                "prepare": 0.6, "generate": 0.6, "update": 0.6, "calculate": 0.6,
                "process": 0.6, "monitor": 0.65, "revoke": 0.7, "disable": 0.7,
                "remove": 0.7, "limit": 0.7, "restrict": 0.7, "resolve": 0.7
            },
            "low_strength_verbs": {
                "look": 0.2, "observe": 0.3, "view": 0.2, "consider": 0.2,
                "watch": 0.2, "note": 0.3, "see": 0.1, "handle": 0.2,
                "manage": 0.3, "coordinate": 0.3, "oversee": 0.4, "perform": 0.3
            },
            "problematic_verbs": {
                "use": 0.1, "launch": 0.1, "set": 0.1, "meet": 0.1, "include": 0.1,
                "have": 0.1, "be": 0.1, "exist": 0.1, "contain": 0.1, "store": 0.2
            }
        }

        # Merge with config if available
        if self.config.config and "WHAT" in self.config.config.get("elements", {}):
            config_categories = self.config.config["elements"]["WHAT"]
            for category in default_categories:
                if category in config_categories:
                    if isinstance(config_categories[category], dict):
                        default_categories[category].update(config_categories[category])

        # Integrate existing keywords
        for keyword in self.config.existing_keywords:
            if " " not in keyword.lower():
                if keyword.lower() not in any(cat.keys() for cat in default_categories.values()):
                    default_categories["medium_strength_verbs"][keyword.lower()] = 0.6

        return default_categories

    def get_verb_strength(self, verb_lemma: str) -> float:
        """Get the strength score for a verb"""
        for category, verbs in self.verb_categories.items():
            if verb_lemma in verbs:
                return verbs[verb_lemma]
        return 0.5

    def get_verb_category(self, verb_lemma: str) -> str:
        """Get the category for a verb"""
        for category, verbs in self.verb_categories.items():
            if verb_lemma in verbs:
                return category
        return "unknown"


class PhraseBuilder:
    """Handles verb phrase construction and cleaning"""

    # Timing exclusion words for phrase cleaning
    TIMING_EXCLUSIONS = [
        "annually", "monthly", "quarterly", "weekly", "daily",
        "periodically", "regularly", "basis", "ad", "hoc"
    ]

    # Purpose clause starters to exclude
    PURPOSE_CLAUSE_STARTERS = [
        "to ensure", "to manage", "to maintain", "to provide", "to comply",
        "to achieve", "to support", "to enable", "to prevent", "to detect"
    ]

    def __init__(self, config: WhatDetectionConfig):
        self.config = config

    def build_verb_phrase(self, token, spacy_doc, is_passive=False) -> Tuple[str, Optional[Dict]]:
        """Enhanced phrase building with stricter purpose clause boundaries"""
        try:
            # Start with the verb itself
            phrase_tokens = [token]
            where_component = None

            # Handle future tense auxiliaries
            future_auxiliaries = ["will", "shall", "can", "may", "must", "should"]

            # Look for auxiliary verbs before this verb
            for child in token.children:
                if (child.dep_ == "aux" and
                    child.lemma_.lower() in future_auxiliaries and
                    child.i < token.i):
                    phrase_tokens.insert(0, child)

            # Look for auxiliaries that govern this verb
            if (token.head.lemma_.lower() in future_auxiliaries and
                token.head.i < token.i and
                token.dep_ in ["xcomp", "ccomp"]):
                phrase_tokens.insert(0, token.head)

            # Handle passive constructions
            if is_passive:
                passive_tokens = self._handle_passive_construction_enhanced(token)
                if passive_tokens:
                    phrase_tokens = passive_tokens

            # Process children with enhanced filtering
            for child in token.children:
                # ALWAYS include direct objects (essential to action)
                if child.dep_ in ["dobj", "iobj", "attr", "oprd"]:
                    obj_tokens = self._get_core_noun_phrase(child)
                    phrase_tokens.extend(obj_tokens)

                # STRICT purpose clause exclusion
                elif self._is_child_purpose_clause_enhanced(child):
                    continue

                # EXCLUDE timing modifiers
                elif self._is_child_timing_modifier(child):
                    continue

                # Include essential prepositional phrases (but limit)
                elif child.dep_ == "prep" and child.text.lower() in ["from", "by", "with", "using"]:
                    prep_tokens = list(child.subtree)
                    if len(prep_tokens) <= 4:  # Limit length to avoid over-capture
                        phrase_tokens.extend(prep_tokens)
                        # Check for WHERE component
                        where_component = self._detect_where_in_prep_phrase(child)

                # Include close adverbs that modify the action
                elif (child.dep_ == "advmod" and
                      child.pos_ == "ADV" and
                      abs(child.i - token.i) <= 2 and
                      child.text.lower() not in self.TIMING_EXCLUSIONS):
                    phrase_tokens.append(child)

            # Sort tokens by position and build phrase
            phrase_tokens.sort(key=lambda x: x.i if hasattr(x, 'i') else 0)
            verb_phrase = " ".join(t.text for t in phrase_tokens if hasattr(t, 'text'))

            # Enhanced phrase cleaning
            verb_phrase = self._clean_verb_phrase_enhanced(verb_phrase)

            return verb_phrase, where_component

        except Exception as e:
            print(f"Error building verb phrase for '{token.text}': {str(e)}")
            return token.text, None

    def _handle_passive_construction_enhanced(self, token):
        """Enhanced passive construction phrase building"""
        if token.tag_ != "VBN":
            return None

        aux_be = None
        passive_subject_tokens = []

        # Find the auxiliary "be"
        for ancestor in token.ancestors:
            if ancestor.lemma_ == "be" and ancestor.dep_ in ["ROOT", "aux"]:
                aux_be = ancestor
                break

        if aux_be:
            # Find passive subject
            for child in aux_be.children:
                if child.dep_ == "nsubjpass":
                    subj_tokens = self._get_core_noun_phrase(child)
                    passive_subject_tokens.extend(subj_tokens)
                    break

            # Rebuild phrase: [passive_subject] [aux_be] [main_verb]
            if passive_subject_tokens:
                return passive_subject_tokens + [aux_be] + [token]

        return None

    def _get_core_noun_phrase(self, noun_token):
        """Extract core noun phrase without excessive modifiers"""
        if hasattr(noun_token, '__iter__') and not hasattr(noun_token, 'text'):
            return noun_token

        core_tokens = [noun_token]

        # Include essential modifiers only
        for child in noun_token.children:
            if child.dep_ in ["det", "amod"] and abs(child.i - noun_token.i) <= 2:
                core_tokens.append(child)
            elif child.dep_ == "compound":
                core_tokens.append(child)
            elif child.dep_ == "poss":
                core_tokens.append(child)

        core_tokens.sort(key=lambda x: x.i)
        return core_tokens

    def _is_child_purpose_clause_enhanced(self, child_token) -> bool:
        """Enhanced purpose clause detection for child tokens"""
        try:
            # Standard "to + purpose verb" pattern
            if child_token.text.lower() == "to":
                if child_token.i + 1 < len(child_token.doc):
                    next_token = child_token.doc[child_token.i + 1]
                    purpose_verbs = ["ensure", "manage", "maintain", "provide", "comply",
                                   "achieve", "support", "enable", "prevent", "detect",
                                   "identify", "validate", "confirm", "demonstrate"]
                    if next_token.lemma_.lower() in purpose_verbs:
                        return True

            # Direct purpose verb children (infinitive clauses)
            if child_token.dep_ in ["xcomp", "advcl"]:
                purpose_verbs = ["manage", "ensure", "maintain", "provide", "comply"]
                if child_token.lemma_.lower() in purpose_verbs:
                    return True

            # Check if entire subtree contains purpose indicators
            subtree_text = " ".join(t.text.lower() for t in child_token.subtree)
            if any(purpose_pattern in subtree_text for purpose_pattern in [
                "to manage", "to ensure", "to maintain", "to provide", "to comply"
            ]):
                return True

            return False
        except Exception:
            return False

    def _is_child_timing_modifier(self, child_token) -> bool:
        """Check if a child token is a timing modifier to exclude"""
        if child_token.text.lower() in self.TIMING_EXCLUSIONS:
            return True

        if child_token.dep_ in ["npmod", "advmod"]:
            subtree_text = " ".join(t.text.lower() for t in child_token.subtree)
            timing_phrases = ["on an ad hoc basis", "on a monthly basis", "on a quarterly basis"]
            return any(phrase in subtree_text for phrase in timing_phrases)

        return False

    def _detect_where_in_prep_phrase(self, prep_token) -> Optional[Dict]:
        """Detect WHERE components with better patterns"""
        try:
            where_systems = self.config._get_config_value("where_systems", [
                "system", "application", "database", "platform", "sharepoint", "sap", "oracle"
            ])

            prep_text = " ".join(t.text.lower() for t in prep_token.subtree)

            for indicator in where_systems:
                if indicator in prep_text:
                    return {
                        "text": prep_text,
                        "type": "system"
                    }

            return None
        except Exception:
            return None

    def _clean_verb_phrase_enhanced(self, phrase: str) -> str:
        """Enhanced phrase cleaning with better purpose clause removal"""
        # Remove extra whitespace
        phrase = re.sub(r'\s+', ' ', phrase).strip()

        # Remove trailing prepositions
        phrase = re.sub(r'\s+(to|for|with|by|on|at|in)$', '', phrase)

        # ENHANCED purpose clause removal
        purpose_patterns = [
            r'\s+to\s+manage\s+.*$',
            r'\s+to\s+ensure\s+.*$',
            r'\s+to\s+maintain\s+.*$',
            r'\s+to\s+provide\s+.*$',
            r'\s+to\s+comply\s+.*$',
            r'\s+to\s+achieve\s+.*$',
            r'\s+to\s+support\s+.*$',
            r'\s+to\s+enable\s+.*$',
            r'\s+to\s+prevent\s+.*$',
            r'\s+to\s+detect\s+.*$'
        ]

        for pattern in purpose_patterns:
            phrase = re.sub(pattern, '', phrase, flags=re.IGNORECASE)

        # Remove timing artifacts
        timing_patterns = [
            r'\b(on\s+an?\s+ad\s+hoc\s+basis)\b',
            r'\b(annually|monthly|quarterly|weekly|daily)\b',
            r'\b(periodically|regularly)\b'
        ]

        for pattern in timing_patterns:
            phrase = re.sub(pattern, '', phrase, flags=re.IGNORECASE)

        # Clean up multiple spaces
        phrase = re.sub(r'\s+', ' ', phrase).strip()

        return phrase

    def is_purpose_clause_enhanced(self, phrase: str) -> bool:
        """Enhanced purpose clause detection with better patterns"""
        phrase_lower = phrase.lower().strip()

        # Expanded purpose clause starters
        purpose_starters = [
            "to ensure", "to manage", "to maintain", "to provide", "to comply",
            "to achieve", "to support", "to enable", "to prevent", "to detect",
            "to identify", "to validate", "to confirm", "to demonstrate",
            "to address", "to handle", "to control", "to monitor", "to oversee"
        ]

        # Direct starter match
        if any(phrase_lower.startswith(starter) for starter in purpose_starters):
            return True

        # Check for infinitive purpose patterns
        infinitive_purpose_patterns = [
            r'^to\s+\w+\s+(the|any|all)\s+\w+',  # "to manage the eventual..."
            r'^in\s+order\s+to\s+',
            r'^for\s+the\s+purpose\s+of\s+',
            r'^designed\s+to\s+',
            r'^intended\s+to\s+'
        ]

        for pattern in infinitive_purpose_patterns:
            if re.search(pattern, phrase_lower):
                return True

        return False

    @staticmethod
    def is_purpose_clause(phrase: str) -> bool:
        """Detect if a phrase is a purpose clause (WHY, not WHAT) - kept for compatibility"""
        phrase_lower = phrase.lower().strip()

        purpose_starters = [
            "to ensure", "to manage", "to maintain", "to provide", "to comply",
            "to achieve", "to support", "to enable", "to prevent", "to detect",
            "to identify", "to validate", "to confirm", "to demonstrate"
        ]

        return any(phrase_lower.startswith(starter) for starter in purpose_starters)


class ConfidenceCalculator:
    """Handles confidence scoring for verb candidates"""

    # Confidence multipliers moved from magic numbers
    FUTURE_TENSE_BOOST = 1.15
    HIGH_STRENGTH_PASSIVE_PENALTY = 0.9
    PROBLEMATIC_PASSIVE_PENALTY = 0.3
    STANDARD_PASSIVE_PENALTY = 0.7
    SUBJECT_BOOST = 1.1
    OBJECT_SPECIFICITY_BOOST = 0.2
    WHERE_COMPONENT_BOOST = 1.1
    ROOT_VERB_BOOST = 1.2
    POSITION_FACTOR_REDUCTION = 0.1

    def __init__(self, config: WhatDetectionConfig):
        self.config = config

    def calculate_verb_confidence(self, token, verb_strength: float, is_passive: bool,
                                  has_future_aux: bool, has_subject: bool,
                                  object_specificity: float, completeness: float,
                                  verb_category: str, has_where_component: bool) -> float:
        """Calculate confidence with improved scoring logic"""
        confidence = verb_strength

        if self.config.debug_mode:
            print(f"\nConfidence calculation for '{token.text}':")
            print(f"  Base verb strength: {confidence:.2f}")

        # Future tense boost (they're valid controls)
        if has_future_aux:
            confidence *= self.FUTURE_TENSE_BOOST
            if self.config.debug_mode:
                print(f"  After future tense boost: {confidence:.2f}")

        # Less harsh passive penalty for strong verbs
        if is_passive:
            if verb_category == "high_strength_verbs":
                confidence *= self.HIGH_STRENGTH_PASSIVE_PENALTY
            elif verb_category == "problematic_verbs":
                confidence *= self.PROBLEMATIC_PASSIVE_PENALTY
            else:
                confidence *= self.STANDARD_PASSIVE_PENALTY
            if self.config.debug_mode:
                print(f"  After passive adjustment: {confidence:.2f}")

        # Subject clarity boost
        if has_subject:
            confidence *= self.SUBJECT_BOOST
            if self.config.debug_mode:
                print(f"  After subject boost: {confidence:.2f}")

        # Position factor (earlier verbs often more important)
        position_factor = 1.0 - (token.i / len(token.doc)) * self.POSITION_FACTOR_REDUCTION
        confidence *= position_factor
        if self.config.debug_mode:
            print(f"  After position adjustment: {confidence:.2f}")

        # Object specificity boost
        confidence *= (1.0 + object_specificity * self.OBJECT_SPECIFICITY_BOOST)
        if self.config.debug_mode:
            print(f"  After object specificity: {confidence:.2f}")

        # Completeness factor
        confidence *= completeness
        if self.config.debug_mode:
            print(f"  After completeness: {confidence:.2f}")

        # WHERE component boost
        if has_where_component:
            confidence *= self.WHERE_COMPONENT_BOOST
            if self.config.debug_mode:
                print(f"  After WHERE boost: {confidence:.2f}")

        # Root verb boost
        if token.dep_ == "ROOT":
            confidence *= self.ROOT_VERB_BOOST
            if self.config.debug_mode:
                print(f"  After ROOT boost: {confidence:.2f}")

        confidence = min(1.0, confidence)
        if self.config.debug_mode:
            print(f"  Final confidence: {confidence:.2f}")

        return confidence

    def assess_object_specificity(self, token) -> float:
        """Assess object specificity with better scoring"""
        try:
            # Find direct objects
            objects = []
            phrase_builder = PhraseBuilder(self.config)
            for child in token.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    objects.extend(phrase_builder._get_core_noun_phrase(child))

            if not objects:
                return 0.0

            # Get config values
            tech_terms = self.config._get_config_value("technical_terms", [
                "system", "application", "database", "server", "report", "document",
                "transaction", "account", "balance", "reconciliation"
            ])

            vague_object_terms = self.config._get_config_value("vague_object_terms", [
                "item", "thing", "stuff", "issue", "matter", "information", "data"
            ])

            # Start with base score
            score = min(1.0, len(objects) / 3.0)

            # Check for specific indicators
            object_text = " ".join(t.text.lower() for t in objects)

            # Boost for technical terms
            if any(term in object_text for term in tech_terms):
                score += 0.2

            # Boost for numbers/specifics
            if any(t.pos_ == "NUM" for t in objects):
                score += 0.2

            # Boost for proper nouns
            if any(t.pos_ == "PROPN" for t in objects):
                score += 0.2

            # Penalty for vague terms
            if any(term in object_text for term in vague_object_terms):
                score -= 0.3

            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5

    def assess_phrase_completeness(self, verb_phrase: str) -> float:
        """Assess phrase completeness with better criteria"""
        words = verb_phrase.split()

        # Very short phrases are incomplete
        if len(words) < 2:
            return 0.3

        # Check for incomplete endings
        if words[-1].lower() in ["to", "for", "with", "by", "on", "at", "in"]:
            return 0.4

        # Check for purpose clause artifacts
        if "to ensure" in verb_phrase.lower() or "to manage" in verb_phrase.lower():
            return 0.2

        # Good length with verb and object
        if len(words) >= 3:
            return 1.0

        return 0.7


class VerbCandidateExtractor:
    """Extracts verb candidates using different detection methods - ENHANCED"""

    def __init__(self, config: WhatDetectionConfig, verb_analyzer: VerbAnalyzer,
                 phrase_builder: PhraseBuilder, confidence_calc: ConfidenceCalculator):
        self.config = config
        self.verb_analyzer = verb_analyzer
        self.phrase_builder = phrase_builder
        self.confidence_calc = confidence_calc

    def extract_control_verb_candidates(self, spacy_doc) -> List[Dict]:
        """Extract action candidates with improved purpose clause filtering"""
        candidates = []

        try:
            # Process each sentence separately
            for sent in spacy_doc.sents:
                sent_verbs = self._find_sentence_verbs_enhanced(sent)

                # Process each verb in the sentence
                for verb in sent_verbs:
                    candidate = self._process_verb_token_enhanced(verb, spacy_doc)
                    if candidate:
                        candidates.append(candidate)

        except Exception as e:
            print(f"Error in control verb extraction: {str(e)}")

        return candidates

    def _find_sentence_verbs_enhanced(self, sent):
        """Find relevant verbs with early purpose clause filtering"""
        sent_verbs = []

        for token in sent:
            if token.pos_ == "VERB" and token.lemma_.lower() in self.verb_analyzer.all_verbs:
                # EARLY PURPOSE CLAUSE FILTERING
                if self._is_purpose_clause_verb(token):
                    continue

                # Skip auxiliary verbs unless they're part of passive construction
                if token.dep_ == "aux" and not self._is_passive_construction(token):
                    continue

                # Skip "to be" verbs unless they're part of a valid construction
                if token.lemma_ == "be" and not self._is_valid_be_construction(token):
                    continue

                sent_verbs.append(token)

        return sent_verbs

    def _is_purpose_clause_verb(self, token) -> bool:
        """Enhanced purpose clause detection at token level"""
        try:
            # Check if this verb is preceded by "to" (infinitive purpose clause)
            if token.i > 0:
                prev_token = token.doc[token.i - 1]
                if prev_token.text.lower() == "to":
                    # Common purpose verbs
                    purpose_verbs = {
                        "manage", "ensure", "maintain", "provide", "comply",
                        "achieve", "support", "enable", "prevent", "detect",
                        "identify", "validate", "confirm", "demonstrate"
                    }
                    if token.lemma_.lower() in purpose_verbs:
                        return True

            # Check if this verb is in a purpose clause context
            # Look for dependency patterns that indicate purpose
            if token.dep_ in ["advcl", "xcomp"] and token.head.pos_ == "VERB":
                # This verb is subordinate to another verb - likely purpose
                if token.lemma_.lower() in ["manage", "ensure", "maintain", "provide"]:
                    return True

            return False
        except Exception:
            return False

    def _process_verb_token_enhanced(self, verb, spacy_doc):
        """Enhanced verb token processing with better action prioritization"""
        try:
            # Get verb properties
            verb_strength = self.verb_analyzer.get_verb_strength(verb.lemma_.lower())
            verb_category = self.verb_analyzer.get_verb_category(verb.lemma_.lower())
            is_passive = self._is_passive_construction(verb)

            # Check for future tense constructions
            has_future_aux = self._has_future_auxiliary(verb)

            # Build verb phrase with improved logic
            verb_phrase, where_info = self.phrase_builder.build_verb_phrase(
                verb, spacy_doc, is_passive
            )

            # Skip if the verb phrase is empty or too short
            if not verb_phrase or len(verb_phrase.split()) < 1:
                return None

            # ENHANCED purpose clause check on the built phrase
            if self.phrase_builder.is_purpose_clause_enhanced(verb_phrase):
                if self.config.debug_mode:
                    print(f"Skipping purpose clause: {verb_phrase}")
                return None

            # Get the subject
            subject = self._get_subject(verb)

            # PRIORITIZE MAIN CLAUSE VERBS OVER SUBORDINATE CLAUSES
            main_clause_boost = self._calculate_main_clause_boost(verb)

            # Assess object specificity and completeness
            object_specificity = self.confidence_calc.assess_object_specificity(verb)
            completeness = self.confidence_calc.assess_phrase_completeness(verb_phrase)

            # Calculate confidence with improved logic
            confidence = self.confidence_calc.calculate_verb_confidence(
                verb, verb_strength, is_passive, has_future_aux,
                subject is not None, object_specificity, completeness,
                verb_category, where_info is not None
            )

            # Apply main clause boost
            confidence *= main_clause_boost

            # Create candidate
            candidate = {
                "verb": verb.text,
                "verb_lemma": verb.lemma_.lower(),
                "full_phrase": verb_phrase,
                "subject": subject,
                "is_passive": is_passive,
                "has_future_aux": has_future_aux,
                "strength": verb_strength,
                "strength_category": verb_category,
                "object_specificity": object_specificity,
                "completeness": completeness,
                "score": confidence,
                "position": verb.i,
                "detection_method": "dependency_parsing_enhanced",
                "is_core_action": self._is_core_control_action_enhanced(verb, verb.lemma_.lower(), verb_category),
                "main_clause_boost": main_clause_boost
            }

            # Add WHERE component if detected
            if where_info:
                candidate["has_where_component"] = True
                candidate["where_text"] = where_info["text"]
                candidate["where_type"] = where_info["type"]
            else:
                candidate["has_where_component"] = False

            return candidate

        except Exception as e:
            if self.config.debug_mode:
                print(f"Error processing verb {verb.text}: {str(e)}")
            return None

    def _calculate_main_clause_boost(self, verb_token) -> float:
        """Calculate boost for main clause verbs vs subordinate clauses"""
        try:
            # ROOT verbs get highest priority
            if verb_token.dep_ == "ROOT":
                return 1.5

            # Main clause verbs (not in subordinate clauses)
            if verb_token.dep_ in ["conj"]:  # Coordinated main verbs
                return 1.3

            # Subordinate clause verbs get penalties
            if verb_token.dep_ in ["advcl", "relcl", "ccomp", "xcomp"]:
                # Check if this is likely a purpose clause
                if verb_token.lemma_.lower() in ["manage", "ensure", "maintain", "provide"]:
                    return 0.3  # Heavy penalty for purpose verbs
                return 0.7  # General subordinate clause penalty

            # Passive main verbs
            if self._is_passive_construction(verb_token) and verb_token.dep_ != "ROOT":
                # Look for the main passive construction
                for ancestor in verb_token.ancestors:
                    if ancestor.dep_ == "ROOT" and ancestor.lemma_ == "be":
                        return 1.4  # This is the main passive action

            return 1.0  # Default
        except Exception:
            return 1.0

    def _is_core_control_action_enhanced(self, verb_token, verb_lemma: str, verb_category: str) -> bool:
        """Enhanced determination of core control actions"""
        try:
            # Purpose clause verbs are never core actions
            if self._is_purpose_clause_verb(verb_token):
                return False

            # High strength verbs in main clauses are likely core actions
            if verb_category == "high_strength_verbs" and verb_token.dep_ in ["ROOT", "conj"]:
                return True

            # Problematic verbs are definitely not core actions
            if verb_category == "problematic_verbs":
                return False

            # Passive constructions of control verbs
            if self._is_passive_construction(verb_token):
                passive_control_verbs = ["completed", "reviewed", "approved", "verified",
                                       "reconciled", "tested", "validated", "examined", "analyzed"]
                if verb_lemma in passive_control_verbs:
                    return True

            # Root verbs with objects are likely core actions
            if verb_token.dep_ == "ROOT":
                has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
                return has_object

            # Medium strength verbs in main clauses (not subordinate)
            if (verb_category == "medium_strength_verbs" and
                verb_token.dep_ not in ["advcl", "relcl", "ccomp", "xcomp"]):
                return True

            return False
        except Exception:
            return False

    def _is_passive_construction(self, verb_token) -> bool:
        """Enhanced passive voice detection"""
        try:
            # Method 1: Explicit passive subject
            if any(token.dep_ == "nsubjpass" for token in verb_token.children):
                return True

            # Method 2: Past participle with "be" auxiliary
            if verb_token.tag_ == "VBN":
                # Look for "be" auxiliary
                for ancestor in verb_token.ancestors:
                    if ancestor.lemma_ == "be" and ancestor.pos_ in ["AUX", "VERB"]:
                        return True

                # Also check head
                if verb_token.head.lemma_ == "be":
                    return True

            # Method 3: This IS the "be" verb with past participle child
            if verb_token.lemma_ == "be":
                for child in verb_token.children:
                    if child.tag_ == "VBN" and child.pos_ == "VERB":
                        return True

            return False
        except Exception:
            return False

    def _is_valid_be_construction(self, be_token) -> bool:
        """Check if "be" verb is part of a valid control construction"""
        # Valid if it has past participle child (passive)
        for child in be_token.children:
            if child.tag_ == "VBN" and child.pos_ == "VERB":
                return True

        # Valid if followed by "responsible for" or similar
        responsible_indicators = ["responsible", "accountable", "required"]
        for child in be_token.children:
            if child.lemma_.lower() in responsible_indicators:
                return True

        return False

    def _has_future_auxiliary(self, verb_token) -> bool:
        """Check if verb has future auxiliary (will, shall, etc.)"""
        future_auxiliaries = ["will", "shall", "can", "may", "must", "should"]

        # Check children for auxiliary
        for child in verb_token.children:
            if (child.dep_ == "aux" and
                child.lemma_.lower() in future_auxiliaries):
                return True

        # Check if this verb is governed by a future auxiliary
        if (verb_token.head.lemma_.lower() in future_auxiliaries and
            verb_token.dep_ in ["xcomp", "ccomp"]):
            return True

        return False

    def _get_subject(self, verb_token) -> Optional[str]:
        """Find the subject of a verb with better handling"""
        try:
            for token in verb_token.children:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    # Get core noun phrase
                    core_tokens = self.phrase_builder._get_core_noun_phrase(token)
                    return " ".join(t.text for t in core_tokens)

            # For compound verbs, look at governing verb's subject
            if verb_token.dep_ == "xcomp" and verb_token.head.pos_ == "VERB":
                return self._get_subject(verb_token.head)

            return None
        except Exception:
            return None


def analyze_control_actions(text: str, nlp, existing_keywords: Optional[List[str]] = None,
                           control_type: Optional[str] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced WHAT detection with comprehensive purpose clause filtering.

    KEY FIXES:
    1. Early purpose clause filtering at verb detection
    2. Enhanced main clause prioritization
    3. Better passive voice detection for actual actions
    4. Stricter phrase boundary detection
    5. Improved candidate ranking

    Args:
        text: The control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Optional list of action keywords to consider
        control_type: Optional control type (preventive, detective, corrective) for validation
        config: Optional configuration dictionary (overrides default settings)

    Returns:
        Dictionary containing detailed analysis of action elements
    """
    if not text or text.strip() == '':
        return _create_empty_result("No text provided to analyze")

    try:
        # Process the text with spaCy
        spacy_doc = nlp(text.lower())

        # Initialize configuration and analyzers with enhanced classes
        detection_config = WhatDetectionConfig(spacy_doc, existing_keywords, control_type, config)
        verb_analyzer = VerbAnalyzer(detection_config)
        phrase_builder = PhraseBuilder(detection_config)
        confidence_calc = ConfidenceCalculator(detection_config)
        extractor = VerbCandidateExtractor(detection_config, verb_analyzer, phrase_builder, confidence_calc)

        # Initialize tracking variables
        active_voice_count = 0
        passive_voice_count = 0
        action_candidates = []

        # Phase 1: Enhanced primary detection with purpose clause filtering
        verb_candidates = extractor.extract_control_verb_candidates(spacy_doc)

        # Update voice tracking
        for candidate in verb_candidates:
            if candidate["is_passive"]:
                passive_voice_count += 1
            else:
                active_voice_count += 1

        action_candidates.extend(verb_candidates)

        # Phase 2: Pattern-based detection (only if no strong actions found)
        if not _has_strong_primary_action(verb_candidates):
            pattern_candidates = _extract_action_patterns_enhanced(text, spacy_doc, detection_config)
            action_candidates.extend(pattern_candidates)

        # Phase 3: Noun chunk fallback (only if still no viable actions)
        if len(action_candidates) == 0:
            noun_chunk_candidates = _extract_from_noun_chunks(spacy_doc, nlp, detection_config)
            action_candidates.extend(noun_chunk_candidates)

        # Enhanced filtering and ranking
        filtered_candidates = _filter_action_candidates_enhanced(action_candidates, phrase_builder)

        # Determine primary and secondary actions with better logic
        primary_action, secondary_actions = _determine_primary_secondary_actions_enhanced(filtered_candidates, phrase_builder)

        # Determine voice
        voice = _determine_voice(active_voice_count, passive_voice_count)

        # Evaluate control type alignment if control_type is provided
        control_type_alignment = _evaluate_control_type_alignment(
            primary_action, secondary_actions, control_type, text, detection_config
        )

        # Calculate final score
        final_score = _calculate_final_score(
            primary_action, secondary_actions, text,
            control_type_alignment["is_aligned"] if control_type else True
        )

        # Generate suggestions
        suggestions = _generate_what_suggestions(
            filtered_candidates, text, control_type, control_type_alignment, detection_config
        )

        # Determine if this is describing a process rather than a single control
        is_process = _determine_if_process(filtered_candidates, text)

        return {
            "primary_action": primary_action,
            "secondary_actions": secondary_actions,
            "actions": filtered_candidates,
            "score": final_score,
            "voice": voice,
            "suggestions": suggestions,
            "is_process": is_process,
            "control_type_alignment": control_type_alignment
        }

    except Exception as e:
        print(f"Error in enhanced WHAT detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return _create_error_result(str(e))


def enhance_what_detection(text: str, nlp, existing_keywords: Optional[List[str]] = None,
                          control_type: Optional[str] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    MAIN ENHANCED WHAT DETECTION FUNCTION - FIXED VERSION

    This is the main entry point that should be called instead of analyze_control_actions.
    It includes all the fixes for the purpose clause issue.
    """

    # Use the enhanced analysis function with all fixes
    return analyze_control_actions(text, nlp, existing_keywords, control_type, config)


def _extract_action_patterns_enhanced(text: str, spacy_doc, config: WhatDetectionConfig) -> List[Dict]:
    """Enhanced pattern extraction with purpose clause filtering"""
    action_candidates = []
    text_lower = text.lower()

    try:
        # Enhanced control patterns with purpose clause exclusion
        control_patterns = [
            # Focus on passive constructions (main actions)
            (r'(strategy|review|reconciliation|analysis|assessment)\s+is\s+(completed|performed|conducted)', 0.9),
            (r'(will\s+)?(obtain|collect|gather|receive)\s+([a-z\s]+)', 0.8),
            (r'(notify|alert|inform)\s+([a-z\s]+)', 0.75),
            (r'(validate|verify|confirm)\s+that\s+([a-z\s]+)', 0.8),
            (r'(age|categorize|classify)\s+([a-z\s]+)', 0.7),
            (r'(limit|restrict)\s+([a-z\s]+)\s+to\s+([a-z\s]+)', 0.75),
            (r'(route|escalate|forward)\s+([a-z\s]+)\s+to\s+([a-z\s]+)', 0.75)
        ]

        # Apply patterns with purpose clause filtering
        for pattern, confidence in control_patterns:
            for match in re.finditer(pattern, text_lower):
                verb_phrase = match.group(0)

                # ENHANCED purpose clause exclusion
                if _contains_purpose_indicators(verb_phrase):
                    continue

                # Skip if it contains purpose clause indicators
                if any(purpose in verb_phrase for purpose in [
                    "to ensure", "to manage", "to maintain", "to provide", "to comply"
                ]):
                    continue

                # Extract main verb
                verb = _extract_main_verb_from_pattern(match)

                # Clean the phrase
                phrase_builder = PhraseBuilder(config)
                clean_phrase = phrase_builder._clean_verb_phrase_enhanced(verb_phrase)

                where_info = _detect_where_in_prep_phrase_simple(clean_phrase, config)

                action_candidates.append({
                    "verb": verb,
                    "verb_lemma": verb,
                    "full_phrase": clean_phrase,
                    "subject": None,
                    "is_passive": "is" in verb_phrase,
                    "has_future_aux": "will" in verb_phrase,
                    "strength": confidence,
                    "strength_category": "medium",
                    "object_specificity": 0.6,
                    "completeness": 0.8,
                    "score": confidence,
                    "position": match.start(),
                    "detection_method": "pattern_matching_enhanced",
                    "is_core_action": True,
                    "has_where_component": where_info is not None,
                    "where_text": where_info["text"] if where_info else None,
                    "where_type": where_info["type"] if where_info else None
                })

    except Exception as e:
        print(f"Error in enhanced pattern extraction: {str(e)}")

    return action_candidates


def _extract_from_noun_chunks(spacy_doc, nlp, config: WhatDetectionConfig) -> List[Dict]:
    """Noun phrase fallback with better filtering"""
    action_candidates = []

    try:
        # Control activity nouns
        control_nouns = {
            "review": 0.8, "approval": 0.8, "validation": 0.8, "verification": 0.8,
            "reconciliation": 0.8, "assessment": 0.7, "monitoring": 0.7,
            "inspection": 0.7, "audit": 0.8, "analysis": 0.7, "testing": 0.7
        }

        verb_mapping = {
            "review": "review", "approval": "approve", "validation": "validate",
            "verification": "verify", "reconciliation": "reconcile",
            "assessment": "assess", "monitoring": "monitor", "inspection": "inspect",
            "audit": "audit", "analysis": "analyze", "testing": "test"
        }

        for chunk in spacy_doc.noun_chunks:
            if len(chunk) > 1:
                head = chunk.root
                head_text = head.text.lower()

                if head_text in control_nouns:
                    verb = verb_mapping.get(head_text, "perform")

                    # Get modifiers
                    modifiers = [t.text for t in chunk if t.i != head.i]

                    if modifiers:
                        phrase = f"{verb} {' '.join(modifiers)}"
                    else:
                        phrase = f"{verb} {head_text}"

                    # Clean phrase
                    phrase_builder = PhraseBuilder(config)
                    clean_phrase = phrase_builder._clean_verb_phrase_enhanced(phrase)

                    where_info = _detect_where_in_prep_phrase_simple(chunk.text, config)

                    action_candidates.append({
                        "verb": verb,
                        "verb_lemma": verb,
                        "full_phrase": clean_phrase,
                        "subject": None,
                        "is_passive": False,
                        "has_future_aux": False,
                        "strength": control_nouns[head_text],
                        "strength_category": "medium",
                        "object_specificity": 0.5,
                        "completeness": 0.7,
                        "score": control_nouns[head_text] * 0.8,
                        "position": chunk.start,
                        "detection_method": "noun_chunk_analysis",
                        "is_core_action": True,
                        "has_where_component": where_info is not None,
                        "where_text": where_info["text"] if where_info else None,
                        "where_type": where_info["type"] if where_info else None
                    })

    except Exception as e:
        print(f"Error in noun chunk extraction: {str(e)}")

    return action_candidates


def _detect_where_in_prep_phrase_simple(text: str, config: WhatDetectionConfig) -> Optional[Dict]:
    """Simple WHERE detection for pattern-based extraction"""
    try:
        where_systems = config._get_config_value("where_systems", [
            "system", "application", "database", "platform"
        ])

        text_lower = text.lower()
        for indicator in where_systems:
            if indicator in text_lower:
                return {
                    "text": text,
                    "type": "system"
                }
        return None
    except Exception:
        return None


def _has_strong_primary_action(candidates: List[Dict]) -> bool:
    """Determine if we have at least one strong action candidate"""
    STRONG_ACTION_THRESHOLD = 0.7
    return any(c["score"] >= STRONG_ACTION_THRESHOLD for c in candidates)


def _contains_purpose_indicators(text: str) -> bool:
    """Check if text contains purpose clause indicators"""
    purpose_indicators = [
        "to manage", "to ensure", "to maintain", "to provide", "to comply",
        "to achieve", "to support", "to enable", "to prevent", "to detect"
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in purpose_indicators)


def _extract_main_verb_from_pattern(match) -> str:
    """Extract the main verb from a regex match"""
    # Simple extraction logic - can be enhanced based on pattern structure
    groups = match.groups()
    for group in groups:
        if group and group not in ["will", "is", "are", "was", "were"]:
            return group.split()[0]  # First word of the group
    return "perform"  # Fallback


def _filter_action_candidates_enhanced(candidates: List[Dict], phrase_builder: PhraseBuilder) -> List[Dict]:
    """Enhanced candidate filtering with purpose clause removal"""
    if not candidates:
        return []

    try:
        # Remove purpose clause candidates
        non_purpose_candidates = []

        for candidate in candidates:
            if not phrase_builder.is_purpose_clause_enhanced(candidate["full_phrase"]):
                non_purpose_candidates.append(candidate)

        # Remove duplicates more intelligently
        seen_phrases = set()
        unique_candidates = []

        for candidate in non_purpose_candidates:
            normalized_phrase = candidate["full_phrase"].lower().strip()
            key = (candidate["verb_lemma"], normalized_phrase)

            if key not in seen_phrases:
                seen_phrases.add(key)
                unique_candidates.append(candidate)

        # Enhanced confidence threshold
        CONFIDENCE_THRESHOLD = 0.3
        filtered_candidates = [c for c in unique_candidates if c["score"] >= CONFIDENCE_THRESHOLD]

        # Keep at least one candidate if we have any
        if not filtered_candidates and unique_candidates:
            best_candidate = max(unique_candidates, key=lambda x: x["score"])
            filtered_candidates.append(best_candidate)

        return filtered_candidates
    except Exception as e:
        print(f"Error in enhanced filtering: {str(e)}")
        return candidates


def _determine_primary_secondary_actions_enhanced(filtered_candidates: List[Dict], phrase_builder: PhraseBuilder) -> Tuple[Optional[Dict], List[Dict]]:
    """Enhanced primary/secondary action determination with main clause prioritization"""
    primary_action = None
    secondary_actions = []

    if filtered_candidates:
        # Sort by main clause boost first, then by score
        filtered_candidates.sort(key=lambda x: (x.get("main_clause_boost", 1.0), x["score"]), reverse=True)

        # Assign primary action (prefer main clause verbs)
        primary_action = filtered_candidates[0]

        # Assign secondary actions (exclude obvious purpose clauses)
        secondary_candidates = filtered_candidates[1:3] if len(filtered_candidates) > 1 else []

        for candidate in secondary_candidates:
            if not phrase_builder.is_purpose_clause_enhanced(candidate["full_phrase"]):
                secondary_actions.append(candidate)

    return primary_action, secondary_actions


def _determine_voice(active_count: int, passive_count: int) -> str:
    """Determine the dominant voice in the control"""
    if active_count > passive_count:
        return "active"
    elif passive_count > active_count:
        return "passive"
    elif active_count > 0 or passive_count > 0:
        return "mixed"
    else:
        return "unknown"


def _evaluate_control_type_alignment(primary_action: Optional[Dict], secondary_actions: List[Dict],
                                  control_type: Optional[str], text: str, config: WhatDetectionConfig) -> Dict[str, Any]:
    """Evaluate alignment between detected actions and declared control type"""
    if not control_type or not primary_action:
        return {
            "is_aligned": True,
            "message": "No control type specified for validation" if not control_type else "No action detected",
            "score": 1.0 if not control_type else 0.0
        }

    control_type_indicators = config._get_config_value("control_type_indicators", {
        "preventive": ["prevent", "block", "restrict", "limit", "authorize", "validate before"],
        "detective": ["detect", "identify", "review", "monitor", "verify", "reconcile", "examine"],
        "corrective": ["correct", "remediate", "fix", "resolve", "address", "adjust"]
    })

    control_type = control_type.lower().strip()
    if control_type not in control_type_indicators:
        return {"is_aligned": False, "message": f"Unknown control type: {control_type}", "score": 0.0}

    indicators = control_type_indicators[control_type]
    primary_verb = primary_action["verb_lemma"]
    primary_phrase = primary_action["full_phrase"].lower()

    # Check alignments
    if any(indicator in primary_verb for indicator in indicators):
        return {"is_aligned": True, "message": f"Primary action aligns with {control_type}", "score": 1.0}
    elif any(indicator in primary_phrase for indicator in indicators):
        return {"is_aligned": True, "message": f"Action phrase aligns with {control_type}", "score": 0.8}
    else:
        return {"is_aligned": False, "message": f"No alignment with {control_type} control type", "score": 0.0}


def _calculate_final_score(primary_action: Optional[Dict], secondary_actions: List[Dict],
                         text: str, control_type_aligned: bool) -> float:
    """Calculate final score for the WHAT element"""
    if not primary_action:
        return 0.0

    # Weight components
    primary_score = primary_action["score"] * 0.5
    secondary_score = 0.0

    if secondary_actions:
        avg_secondary = sum(a["score"] for a in secondary_actions) / len(secondary_actions)
        secondary_score = avg_secondary * 0.2

    # Text structure score
    structure_score = 0.2
    word_count = len(text.split())
    if word_count < 10:
        structure_score *= 0.7
    elif word_count > 100:
        structure_score *= 0.8

    # Control type alignment
    control_type_score = 0.1 if control_type_aligned else 0.0

    return min(1.0, primary_score + secondary_score + structure_score + control_type_score)


def _generate_what_suggestions(candidates: List[Dict], text: str, control_type: Optional[str],
                               control_type_alignment: Dict, config: WhatDetectionConfig) -> List[str]:
    """Generate better improvement suggestions"""
    suggestions = []

    try:
        if not candidates:
            suggestions.append("No clear control action detected. Add a specific action verb (e.g., 'review', 'approve', 'verify').")
            return suggestions

        primary = candidates[0] if candidates else None

        # Better weak verb suggestions
        if primary and primary["strength_category"] in ["low_strength_verbs", "problematic_verbs"]:
            verb = primary["verb"]
            alternatives = _get_specific_alternatives(primary["verb_lemma"], control_type, config)
            suggestions.append(f"Replace weak verb '{verb}' with a stronger control verb: {alternatives}")

        # Voice suggestions
        voice_counts = {"active": 0, "passive": 0}
        for candidate in candidates:
            if candidate["is_passive"]:
                voice_counts["passive"] += 1
            else:
                voice_counts["active"] += 1

        if voice_counts["passive"] > voice_counts["active"]:
            suggestions.append("Consider using active voice to clearly specify who performs the control")

        # WHERE component suggestion
        has_where = any(c.get("has_where_component", False) for c in candidates)
        if not has_where and len(text.split()) > 20:
            suggestions.append("Consider specifying WHERE the control is performed (system, application, location)")

        # Object specificity
        if primary and primary.get("object_specificity", 1.0) < 0.4:
            suggestions.append("Make the object of the action more specific (what exactly is being reviewed/approved?)")

        # Multiple actions
        distinct_actions = set(c["verb_lemma"] for c in candidates)
        if len(distinct_actions) > 3:
            suggestions.append("This describes multiple actions. Consider breaking into separate controls.")

        # Control type alignment
        if control_type and not control_type_alignment.get("is_aligned", True):
            control_type_indicators = config._get_config_value("control_type_indicators", {})
            if control_type.lower() in control_type_indicators:
                indicators = control_type_indicators[control_type.lower()]
                examples = indicators[:3] if len(indicators) > 3 else indicators
                suggestions.append(
                    f"Actions don't align with '{control_type}' control type. "
                    f"Consider using: {', '.join(examples)}"
                )

        return suggestions
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
        return ["Error generating suggestions. Please review manually."]


def _get_specific_alternatives(verb_lemma: str, control_type: Optional[str], config: WhatDetectionConfig) -> str:
    """Better alternative suggestions based on control type"""
    # Control type specific alternatives
    if control_type:
        control_type_lower = control_type.lower()
        if control_type_lower == "preventive":
            return "'restrict', 'block', 'prevent', 'authorize'"
        elif control_type_lower == "detective":
            return "'review', 'monitor', 'verify', 'reconcile'"
        elif control_type_lower == "corrective":
            return "'resolve', 'correct', 'remediate', 'fix'"

    # General alternatives
    alternatives = {
        "perform": "'verify', 'examine', 'evaluate'",
        "do": "'execute', 'conduct', 'complete'",
        "handle": "'process', 'resolve', 'manage'",
        "check": "'verify', 'validate', 'examine'",
        "review": "'examine', 'analyze', 'evaluate'",
        "manage": "'administer', 'oversee', 'supervise'"
    }

    return alternatives.get(verb_lemma, "'verify', 'approve', 'reconcile'")


def _determine_if_process(candidates: List[Dict], text: str) -> bool:
    """Determine if this is describing a process rather than a single control"""
    distinct_verbs = set(c["verb_lemma"] for c in candidates)

    process_indicators = 0
    if len(distinct_verbs) > 3:
        process_indicators += 2

    sequence_markers = ["then", "after", "before", "next", "subsequently", "finally"]
    if any(marker in text.lower() for marker in sequence_markers):
        process_indicators += 1

    if len(text.split()) > 50:
        process_indicators += 1

    return process_indicators >= 3


def mark_possible_standalone_controls(text: str, nlp) -> list:
    """
    Identify potential standalone controls within a description
    (Kept from original for compatibility)
    """
    if not text or len(text) < 20:
        return []

    candidates = []

    # Look for numbered controls
    numbered_pattern = r'\b(\d+[\.\)]\s+[A-Z][^\.;]{10,100})'
    for match in re.finditer(numbered_pattern, text):
        candidates.append({
            "text": match.group(1).strip(),
            "score": 0.9,
            "action": "Split into separate control"
        })

    # Look for explicit control statements
    control_pattern = r'(Control\s+\d+:?\s+[^\.;]{10,100})'
    for match in re.finditer(control_pattern, text, re.IGNORECASE):
        candidates.append({
            "text": match.group(1).strip(),
            "score": 0.95,
            "action": "Split into separate control"
        })

    # Remove duplicates and sort
    unique_candidates = []
    seen_texts = set()

    for candidate in candidates:
        normalized_text = candidate["text"].lower()
        if normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            unique_candidates.append(candidate)

    unique_candidates.sort(key=lambda x: x["score"], reverse=True)
    return unique_candidates


def _create_empty_result(message: str) -> Dict[str, Any]:
    """Create minimal result for empty input"""
    return {
        "primary_action": None,
        "secondary_actions": [],
        "actions": [],
        "score": 0,
        "voice": None,
        "suggestions": [message],
        "is_process": False,
        "control_type_alignment": {"is_aligned": False, "message": message}
    }


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create minimal result for error cases"""
    return {
        "primary_action": None,
        "secondary_actions": [],
        "actions": [],
        "score": 0,
        "voice": None,
        "suggestions": [f"Error analyzing text: {error_msg}"],
        "is_process": False,
        "control_type_alignment": {"is_aligned": False, "message": "Error during analysis"}
    }