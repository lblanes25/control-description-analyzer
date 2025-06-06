"""
WHY Element Detection Module

This module analyzes control descriptions to identify purpose statements using
multiple detection strategies, including pattern matching, semantic analysis,
and alignment with risk descriptions.

The implementation prioritizes reliability and consistent detection across
different control phrasing patterns with configurable thresholds and improves
handling of escalation targets vs. purpose statements.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class DetectionThresholds:
    """Confidence score thresholds for different types of matches."""
    EXPLICIT_LEAD_IN = 0.95
    DIRECT_PURPOSE = 0.9
    INDIRECT_PURPOSE = 0.75
    IMPLIED_PURPOSE = 0.6
    MINIMAL_PURPOSE = 0.4
    VAGUE_PENALTY = 0.7
    IMPLICIT_DISCOUNT = 0.8
    RISK_BOOST_THRESHOLD = 0.7
    RISK_BOOST_MINIMUM = 0.5

    # Text percentage limits
    SHORT_CONTROL_WORDS = 12
    MEDIUM_CONTROL_WORDS = 30
    SHORT_MAX_PERCENTAGE = 1.0
    MEDIUM_MAX_PERCENTAGE = 0.85
    STANDARD_MAX_PERCENTAGE = 0.7


class PatternLibrary:
    """Regex patterns for purpose detection."""

    # Purpose detection patterns
    PURPOSE_PATTERNS = [
        r'(?i)to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)\s+([^\.;,]{3,50})',
        r'(?i)to\s+(?!(?:the|a|an|our|their)\s)(\w+)\s+([^\.;,]{3,50})',
        r'(?i)^to\s+(?!(?:the|a|an|our|their)\s)(\w+)\s+([^\.;,]{3,50})',
        r'(?i)in\s+order\s+to\s+([^\.;,]{3,50})',
        r'(?i)for\s+the\s+purpose\s+of\s+([^\.;,]{3,50})',
        r'(?i)designed\s+to\s+([^\.;,]{3,50})',
        r'(?i)intended\s+to\s+([^\.;,]{3,50})',
        r'(?i)so\s+that\s+([^\.;,]{3,50})',
        r'(?i)with\s+the\s+aim\s+(?:of|to)\s+([^\.;,]{3,50})',
        r'(?i)in\s+an\s+effort\s+to\s+([^\.;,]{3,50})',
        r'(?i)the\s+purpose\s+(?:of\s+this\s+control\s+is|is)\s+to\s+([^\.;,]{3,50})',
        r'(?i)this\s+control\s+(?:is\s+designed|exists|is\s+in\s+place)\s+to\s+([^\.;,]{3,50})',
        r'(?i)the\s+objective\s+(?:of\s+this\s+control\s+is|is)\s+to\s+([^\.;,]{3,50})',
        r'(?i)this\s+control\s+helps\s+(?:to|in)\s+([^\.;,]{3,50})',
        r'(?i)(?:is|are|will\s+be)\s+(?:performed|executed|conducted|carried\s+out)\s+to\s+([^\.;,]{3,50})',
        r'(?i)(?:which|that)\s+(?:helps|serves)\s+to\s+([^\.;,]{3,50})'
    ]

    # Mid-sentence purpose patterns
    MID_SENTENCE_PATTERNS = [
        r'\s+(?:which|that|to)\s+(?:ensures?|prevents?|detects?|mitigates?)\s+([^\.;,]{3,40})',
        r'\s+(?:which|that|to)\s+(?:helps?|serves?)\s+(?:ensure|prevent|detect|mitigate)\s+([^\.;,]{3,40})'
    ]

    # Multi-control splitting patterns
    NUMBERED_CONTROLS = r'Control\s+\d+\s*:([^(Control\s+\d+\s*:)]+)'
    LIST_PATTERNS = [
        r'\d+\.\s*([^\d\.]+?)(?=\d+\.\s*|\Z)',
        r'•\s*([^•]+?)(?=•|\Z)',
        r'-\s*([^-]+?)(?=-|\Z)'
    ]

    # Temporal patterns
    TEMPORAL_PREVENTION = r"(before|prior to)\s+\w+ing"
    TEMPORAL_ACTION = r"(before|prior to)\s+(\w+ing)"

    # Success criteria patterns
    SUCCESS_CRITERIA_PATTERNS = [
        r'(\$\d+[,\d]*|\d+\s*%|\d+\s*percent)',
        r'greater than|less than|at least|at most|minimum|maximum',
        r'threshold of|limit of|tolerance of',
        r'within \d+\s*(day|hour|minute|week|month)',
        r'criteria|criterion|standard|benchmark'
    ]


class IntentCategories(Enum):
    """Intent classification categories."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    COMPLIANCE = "compliance"
    RISK_MITIGATION = "risk_mitigation"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    AUTHORIZATION = "authorization"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PurposeCandidate:
    """Represents a purpose statement candidate."""
    text: str
    method: str
    score: float
    span: Tuple[int, int] = (0, 0)
    context: str = ""
    verb: Optional[str] = None
    implied_purpose: Optional[str] = None
    has_vague_term: bool = False


@dataclass
class SemanticConcept:
    """Represents a semantic concept extracted from text."""
    concept_type: str
    text: str
    lemma: str = ""
    modifiers: List[Dict] = field(default_factory=list)
    objects: List[Dict] = field(default_factory=list)
    target: Optional[str] = None
    negates: Optional[str] = None


@dataclass
class RiskAlignment:
    """Risk alignment analysis result."""
    score: float
    feedback: str
    relationships: List[Dict] = field(default_factory=list)
    term_overlap: float = 0.0
    risk_aspects: List[str] = field(default_factory=list)


@dataclass
class SegmentResult:
    """Result for a single control segment."""
    text: str
    explicit_purposes: List[PurposeCandidate]
    implicit_purposes: List[PurposeCandidate]
    top_match: Optional[PurposeCandidate]
    score: float
    vague_phrases: List[Dict]
    intent_category: Optional[str]
    risk_alignment: Optional[RiskAlignment]


@dataclass
class DetectionConfig:
    """Configuration for WHY detection."""
    weight: int = 11
    keywords: List[str] = field(default_factory=list)
    purpose_patterns: List[str] = field(default_factory=list)
    vague_terms: List[str] = field(default_factory=list)
    intent_verbs: Dict[str, List[str]] = field(default_factory=dict)
    mitigation_verbs: List[str] = field(default_factory=list)
    verb_purpose_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Complete WHY detection result."""
    explicit_why: List[PurposeCandidate]
    implicit_why: List[PurposeCandidate]
    top_match: Optional[PurposeCandidate]
    why_category: Optional[str]
    score: float
    is_inferred: bool
    risk_alignment_score: Optional[float]
    risk_alignment_feedback: Optional[str]
    extracted_keywords: List[str]
    has_success_criteria: bool
    vague_why_phrases: List[Dict]
    is_actual_mitigation: bool
    intent_category: Optional[str]
    improvement_suggestions: List[str]
    is_multi_control: bool


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigurationManager:
    """Manages WHY detection configuration."""

    @staticmethod
    def get_default_config() -> DetectionConfig:
        """Get default WHY configuration."""
        return DetectionConfig(
            weight=11,
            keywords=[
                "to ensure", "in order to", "for the purpose of", "designed to",
                "intended to", "so that", "purpose", "objective", "goal",
                "prevent", "detect", "mitigate", "risk", "error", "fraud",
                "misstatement", "compliance", "regulatory", "requirement",
                "accuracy", "completeness", "validity", "integrity"
            ],
            purpose_patterns=PatternLibrary.PURPOSE_PATTERNS,
            vague_terms=[
                "proper functioning", "appropriate", "adequately", "properly",
                "as needed", "as required", "as appropriate", "correct functioning",
                "effective", "efficient", "functioning", "operational", "successful",
                "appropriate action", "necessary action", "properly functioning"
            ],
            intent_verbs={
                "preventive": ["prevent", "avoid", "stop", "block", "prohibit", "restrict", "limit"],
                "detective": ["detect", "identify", "discover", "find", "monitor", "review", "reconcile"],
                "corrective": ["correct", "resolve", "address", "fix", "remediate", "rectify", "resolve"],
                "compliance": ["comply", "adhere", "conform", "follow", "meet", "satisfy", "fulfill"]
            },
            mitigation_verbs=[
                "resolve", "correct", "address", "remediate", "fix", "prevent",
                "block", "stop", "deny", "restrict", "escalate", "alert",
                "notify", "disable", "lockout", "report"
            ],
            verb_purpose_mapping={
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
                "reconcile": {"default": "to ensure data integrity and accuracy"},
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
            },
            categories={
                "risk_mitigation": ["risk", "prevent", "mitigate", "reduce", "avoid", "minimize"],
                "compliance": ["comply", "compliance", "regulatory", "regulation", "requirement", "policy", "standard"],
                "accuracy": ["accuracy", "accurate", "correct", "error-free", "integrity", "reliable"],
                "completeness": ["complete", "completeness", "all", "comprehensive"],
                "authorization": ["authorize", "approval", "permission", "authorization"]
            },
            confidence_thresholds={
                "explicit_lead_in": DetectionThresholds.EXPLICIT_LEAD_IN,
                "direct_purpose": DetectionThresholds.DIRECT_PURPOSE,
                "indirect_purpose": DetectionThresholds.INDIRECT_PURPOSE,
                "implied_purpose": DetectionThresholds.IMPLIED_PURPOSE,
                "minimal_purpose": DetectionThresholds.MINIMAL_PURPOSE
            }
        )

    @staticmethod
    def merge_config(base_config: DetectionConfig, user_config: Optional[Union[Dict, List]]) -> DetectionConfig:
        """Merge user configuration with defaults."""
        if isinstance(user_config, list):
            base_config.keywords = user_config
            return base_config

        if not user_config:
            return base_config

        why_config = user_config.get("elements", {}).get("WHY", {})

        for key, value in why_config.items():
            if hasattr(base_config, key):
                if isinstance(value, list) and isinstance(getattr(base_config, key), list):
                    append_flag = why_config.get(f"append_{key}", True)
                    if append_flag:
                        current_list = getattr(base_config, key).copy()
                        for item in value:
                            if item not in current_list:
                                current_list.append(item)
                        setattr(base_config, key, current_list)
                    else:
                        setattr(base_config, key, value)
                elif isinstance(value, dict) and isinstance(getattr(base_config, key), dict):
                    current_dict = getattr(base_config, key).copy()
                    current_dict.update(value)
                    setattr(base_config, key, current_dict)
                else:
                    setattr(base_config, key, value)

        if "weight" in user_config.get("elements", {}).get("WHY", {}):
            base_config.weight = user_config["elements"]["WHY"]["weight"]

        return base_config


# =============================================================================
# TEXT ANALYSIS AND SEMANTIC EXTRACTION
# =============================================================================

class TextAnalyzer:
    """Analyzes text for semantic concepts and patterns."""

    @staticmethod
    def split_multi_control_description(text: str) -> List[str]:
        """Split control description into separate segments."""
        # Check for numbered controls
        numbered_controls = re.findall(PatternLibrary.NUMBERED_CONTROLS, text)
        if numbered_controls:
            return [segment.strip() for segment in numbered_controls]

        # Check for list patterns
        for pattern in PatternLibrary.LIST_PATTERNS:
            segments = re.findall(pattern, text)
            if len(segments) > 1:
                return [segment.strip() for segment in segments]

        # Check for sentence-based splitting for longer text
        if len(text) > 100:
            return TextAnalyzer._split_by_semantic_cues(text)

        return [text]

    @staticmethod
    def _split_by_semantic_cues(text: str) -> List[str]:
        """Split text based on semantic cues like timing and performers."""
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 15]

        # Check for different timing patterns
        timing_patterns = ["daily", "weekly", "monthly", "quarterly", "annually"]
        timing_counts = {}

        for pattern in timing_patterns:
            for i, sentence in enumerate(sentences):
                if re.search(r'\b' + pattern + r'\b', sentence, re.IGNORECASE):
                    timing_counts.setdefault(pattern, []).append(i)

        if len(timing_counts) > 1 and any(len(indices) == 1 for indices in timing_counts.values()):
            return sentences

        # Check for different performers
        performers = []
        for sentence in sentences:
            role_candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sentence)
            if role_candidates:
                performers.append(role_candidates)

        if len(performers) > 1 and len(set(tuple(p) for p in performers if p)) > 1:
            return sentences

        return [text]

    @staticmethod
    def extract_semantic_concepts(doc) -> List[SemanticConcept]:
        """Extract semantic concepts from spaCy document."""
        concepts = []

        # Extract action-object pairs
        for token in doc:
            if token.pos_ == "VERB":
                concept = SemanticConcept(
                    concept_type="action",
                    text=token.text,
                    lemma=token.lemma_
                )

                # Find objects and modifiers
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        obj_text = TextAnalyzer._get_noun_phrase(child, doc)
                        concept.objects.append({
                            "text": obj_text,
                            "lemma": child.lemma_,
                            "has_modifiers": any(c.dep_ in ["amod", "compound"] for c in child.children)
                        })
                    elif child.dep_ in ["advmod", "amod", "aux"]:
                        concept.modifiers.append({
                            "text": child.text,
                            "lemma": child.lemma_
                        })

                if concept.objects or concept.modifiers:
                    concepts.append(concept)

        # Extract negations and attributes
        concepts.extend(TextAnalyzer._extract_negations(doc))
        concepts.extend(TextAnalyzer._extract_attributes(doc))

        return concepts

    @staticmethod
    def _get_noun_phrase(token, doc) -> str:
        """Get complete noun phrase for a token."""
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk.text
        return token.text

    @staticmethod
    def _extract_negations(doc) -> List[SemanticConcept]:
        """Extract negation concepts."""
        concepts = []
        for token in doc:
            if token.dep_ == "neg" or token.text.lower() in ["without", "no", "lack", "missing"]:
                head = token.head
                concepts.append(SemanticConcept(
                    concept_type="negation",
                    text=f"{token.text} {head.text}",
                    lemma=head.lemma_,
                    target=head.lemma_,
                    negates="approval" if "approv" in head.lemma_ else head.lemma_
                ))
        return concepts

    @staticmethod
    def _extract_attributes(doc) -> List[SemanticConcept]:
        """Extract attribute concepts."""
        concepts = []
        for token in doc:
            if (token.dep_ == "amod" and
                token.text.lower() in ["appropriate", "proper", "unauthorized", "authorized"]):
                head = token.head
                concepts.append(SemanticConcept(
                    concept_type="attribute",
                    text=f"{token.text} {head.text}",
                    lemma=token.lemma_,
                    target=head.lemma_
                ))
        return concepts


# =============================================================================
# PURPOSE DETECTION ENGINE
# =============================================================================

class PurposeDetector:
    """Detects explicit and implicit purposes in control descriptions."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def detect_explicit_purposes(self, text: str, doc) -> List[PurposeCandidate]:
        """Detect explicit purpose statements."""
        candidates = []
        text_word_count = len(text.split())
        max_percentage = self._calculate_max_percentage(text_word_count)

        # Handle short controls that are entirely purpose statements
        if self._is_short_purpose_statement(text, text_word_count):
            candidate = self._extract_short_purpose(text)
            if candidate:
                return [candidate]

        # Pattern-based detection
        candidates.extend(self._detect_by_patterns(text, max_percentage, text_word_count))

        # Keyword-based detection (fallback)
        if not candidates:
            candidates.extend(self._detect_by_keywords(text, doc, max_percentage, text_word_count))

        # Mid-sentence purpose detection
        candidates.extend(self._detect_mid_sentence_purposes(text))

        return candidates

    def _calculate_max_percentage(self, word_count: int) -> float:
        """Calculate maximum allowed percentage of text for purpose."""
        if word_count <= DetectionThresholds.SHORT_CONTROL_WORDS:
            return DetectionThresholds.SHORT_MAX_PERCENTAGE
        elif word_count <= DetectionThresholds.MEDIUM_CONTROL_WORDS:
            return DetectionThresholds.MEDIUM_MAX_PERCENTAGE
        else:
            return DetectionThresholds.STANDARD_MAX_PERCENTAGE

    def _is_short_purpose_statement(self, text: str, word_count: int) -> bool:
        """Check if text is a short purpose statement."""
        return (re.match(r'(?i)^to\s+\w+\s+', text) and
                word_count <= DetectionThresholds.SHORT_CONTROL_WORDS)

    def _extract_short_purpose(self, text: str) -> Optional[PurposeCandidate]:
        """Extract purpose from short control descriptions."""
        match = re.match(r'(?i)(to\s+\w+\s+[^\.;,]{3,50})', text.strip())
        if not match:
            return None

        matched_phrase = match.group(1).strip()
        verb_match = re.match(r'(?i)to\s+(\w+)', matched_phrase)
        purpose_verb = verb_match.group(1) if verb_match else None

        purpose_verbs = [
            "ensure", "verify", "confirm", "validate", "prevent", "detect",
            "mitigate", "comply", "adhere", "demonstrate", "maintain",
            "support", "achieve", "provide", "identify"
        ]

        if purpose_verb and purpose_verb.lower() in purpose_verbs:
            return PurposeCandidate(
                text=matched_phrase,
                verb=purpose_verb,
                method="direct_purpose_statement",
                score=DetectionThresholds.EXPLICIT_LEAD_IN,
                span=(0, len(matched_phrase)),
                context=matched_phrase
            )
        return None

    def _detect_by_patterns(self, text: str, max_percentage: float, text_word_count: int) -> List[PurposeCandidate]:
        """Detect purposes using regex patterns."""
        candidates = []

        for pattern in self.config.purpose_patterns:
            for match in re.finditer(pattern, text):
                purpose_text = match.group(0)

                # Calculate confidence based on pattern type
                confidence = self._calculate_pattern_confidence(pattern, purpose_text, match.start())

                # Validate match length
                if len(purpose_text.split()) / text_word_count <= max_percentage:
                    verb = self._extract_purpose_verb(purpose_text)
                    method = self._determine_detection_method(pattern, purpose_text)

                    candidates.append(PurposeCandidate(
                        text=purpose_text,
                        verb=verb,
                        method=method,
                        score=confidence,
                        span=(match.start(), match.end()),
                        context=text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                    ))

        return candidates

    def _calculate_pattern_confidence(self, pattern: str, purpose_text: str, start_pos: int) -> float:
        """Calculate confidence score for pattern match."""
        pattern_lower = pattern.lower()
        purpose_lower = purpose_text.lower()

        # Explicit lead-in phrases
        lead_ins = ["purpose of this control", "this control is designed",
                   "this control exists", "objective of this control"]
        if any(lead_in in pattern_lower for lead_in in lead_ins):
            confidence = self.config.confidence_thresholds["explicit_lead_in"]
        elif purpose_lower.startswith("to "):
            confidence = self.config.confidence_thresholds["direct_purpose"]
        elif any(phrase in purpose_lower for phrase in
                ["in order to", "for the purpose of", "designed to", "intended to", "so that"]):
            confidence = self.config.confidence_thresholds["indirect_purpose"]
        else:
            confidence = 0.7

        # Boost for start-of-text statements
        if start_pos == 0:
            confidence = min(DetectionThresholds.EXPLICIT_LEAD_IN, confidence * 1.1)

        return confidence

    def _extract_purpose_verb(self, purpose_text: str) -> Optional[str]:
        """Extract purpose verb from purpose text."""
        if "to " in purpose_text.lower():
            verb_match = re.search(r'to\s+(\w+)', purpose_text.lower())
            return verb_match.group(1) if verb_match else None
        return None

    def _determine_detection_method(self, pattern: str, purpose_text: str) -> str:
        """Determine detection method based on pattern and text."""
        pattern_lower = pattern.lower()
        purpose_lower = purpose_text.lower()

        if any(lead_in in pattern_lower for lead_in in
               ["purpose of this control", "this control is designed"]):
            return "explicit_lead_in"
        elif purpose_lower.startswith("to "):
            return "direct_purpose_statement"
        elif any(phrase in purpose_lower for phrase in
                ["in order to", "for the purpose of"]):
            return "indirect_purpose_reference"
        else:
            return "pattern_match"

    def _detect_by_keywords(self, text: str, doc, max_percentage: float, text_word_count: int) -> List[PurposeCandidate]:
        """Detect purposes using keywords as fallback."""
        candidates = []
        text_lower = text.lower()

        for keyword in self.config.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                pos = text_lower.find(keyword_lower)
                sentence = self._find_containing_sentence(pos, doc)

                if (sentence and
                    len(sentence.text.split()) / text_word_count <= max_percentage and
                    self._is_purpose_context(keyword_lower, sentence)):

                    candidates.append(PurposeCandidate(
                        text=sentence.text,
                        method="keyword_match",
                        score=self.config.confidence_thresholds["minimal_purpose"],
                        span=(sentence.start_char, sentence.end_char),
                        context=text[max(0, sentence.start_char - 20):min(len(text), sentence.end_char + 20)]
                    ))

        return candidates

    def _find_containing_sentence(self, pos: int, doc):
        """Find sentence containing the given position."""
        for sent in doc.sents:
            if pos >= sent.start_char and pos < sent.end_char:
                return sent
        return None

    def _is_purpose_context(self, keyword: str, sentence) -> bool:
        """Check if keyword is used in purpose context."""
        keyword_pos = sentence.text.lower().find(keyword)
        return (keyword_pos == 0 or
                sentence.text[keyword_pos - 1] in " .,;:(" or
                re.search(r'\b(is|are|was|were|be)\s+' + re.escape(keyword),
                         sentence.text.lower()))

    def _detect_mid_sentence_purposes(self, text: str) -> List[PurposeCandidate]:
        """Detect mid-sentence purpose clauses."""
        candidates = []

        for pattern in PatternLibrary.MID_SENTENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(PurposeCandidate(
                    text=match.group(0).strip(),
                    method="mid_sentence_purpose",
                    score=self.config.confidence_thresholds["minimal_purpose"],
                    span=(match.start(), match.end()),
                    context=text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                ))

        return candidates

    def detect_implicit_purposes(self, action_concepts: List[SemanticConcept], text: str) -> List[PurposeCandidate]:
        """Detect implicit purposes from control actions."""
        candidates = []

        for concept in action_concepts:
            if concept.concept_type == "action":
                verb = concept.lemma

                if verb in self.config.verb_purpose_mapping:
                    purpose_key = self._find_purpose_key(concept, verb)
                    purpose = self._get_implied_purpose(verb, purpose_key, text)
                    confidence = self._calculate_implicit_confidence(purpose_key)

                    candidates.append(PurposeCandidate(
                        text=f"{verb} {' '.join([obj['text'] for obj in concept.objects])}",
                        implied_purpose=purpose,
                        score=confidence,
                        method="inferred_from_action",
                        context="actions"
                    ))

        # Add temporal prevention patterns
        candidates.extend(self._detect_temporal_patterns(text))

        return candidates

    def _find_purpose_key(self, concept: SemanticConcept, verb: str) -> str:
        """Find specific purpose key based on verb objects."""
        for obj in concept.objects:
            obj_text = obj["text"].lower()
            for key in self.config.verb_purpose_mapping[verb]:
                if key != "default" and key in obj_text:
                    return key
        return "default"

    def _get_implied_purpose(self, verb: str, purpose_key: str, text: str) -> str:
        """Get implied purpose for verb and context."""
        if verb == "approve" and re.search(PatternLibrary.TEMPORAL_PREVENTION, text, re.IGNORECASE):
            return "to prevent unauthorized actions"
        return self.config.verb_purpose_mapping[verb][purpose_key]

    def _calculate_implicit_confidence(self, purpose_key: str) -> float:
        """Calculate confidence for implicit purpose."""
        base_confidence = self.config.confidence_thresholds["implied_purpose"]
        if purpose_key != "default":
            return min(0.75, base_confidence + 0.05)
        return base_confidence

    def _detect_temporal_patterns(self, text: str) -> List[PurposeCandidate]:
        """Detect temporal prevention patterns."""
        candidates = []

        match = re.search(PatternLibrary.TEMPORAL_PREVENTION, text, re.IGNORECASE)
        if match:
            action_match = re.search(PatternLibrary.TEMPORAL_ACTION, text, re.IGNORECASE)
            action = action_match.group(2) if action_match else "action"

            candidates.append(PurposeCandidate(
                text=match.group(0),
                implied_purpose=f"to prevent unauthorized {action}",
                score=0.7,
                method="temporal_pattern",
                context="temporal_prevention"
            ))

        return candidates


# =============================================================================
# ESCALATION TARGET FILTER
# =============================================================================

class EscalationTargetFilter:
    """Filters out escalation targets mistaken as purposes."""

    ESCALATION_TERMS = [
        "escalate", "escalation", "escalated", "report", "notify", "alert",
        "inform", "communicate", "refer", "forward", "route", "transmit"
    ]

    ROLE_INDICATORS = [
        "manager", "director", "supervisor", "team", "department", "committee",
        "board", "officer", "executive", "cfo", "ceo", "cio", "head", "chief",
        "president", "vp", "vice president", "group", "lead", "leadership"
    ]

    @classmethod
    def filter_candidates(cls, candidates: List[PurposeCandidate], text: str, nlp) -> List[PurposeCandidate]:
        """Filter out escalation targets from purpose candidates."""
        filtered = []

        for candidate in candidates:
            if cls._is_escalation_target(candidate, text, nlp):
                continue
            filtered.append(candidate)

        return filtered

    @classmethod
    def _is_escalation_target(cls, candidate: PurposeCandidate, text: str, nlp) -> bool:
        """Check if candidate is an escalation target."""
        purpose_text = candidate.text.lower()

        if not purpose_text.startswith("to "):
            return False

        words = purpose_text.split()
        if len(words) < 2:
            return False

        # Check "to [someone]" pattern
        second_word = words[2] if len(words) >= 3 and words[1] == "the" else words[1]

        # Check if it's a role
        is_role = (second_word in cls.ROLE_INDICATORS or
                  any(indicator in purpose_text for indicator in cls.ROLE_INDICATORS))

        # Check for escalation context
        span_start = candidate.span[0]
        preceding_text = text[:span_start].lower()
        has_escalation_context = any(term in preceding_text for term in cls.ESCALATION_TERMS)

        if is_role and has_escalation_context:
            return True

        # Validate verb after "to"
        verb_match = re.search(r'\bto\s+(\w+)', purpose_text)
        if verb_match:
            verb = verb_match.group(1)
            verb_doc = nlp(verb)
            if verb_doc[0].pos_ != "VERB" and not is_role:
                return True

        return False


# =============================================================================
# RISK ALIGNMENT ANALYZER
# =============================================================================

class RiskAlignmentAnalyzer:
    """Analyzes alignment between control purposes and risk descriptions."""

    def __init__(self, nlp, config: DetectionConfig):
        self.nlp = nlp
        self.config = config

    def analyze_alignment(self, text: str, risk_description: str,
                         explicit_purposes: List[PurposeCandidate],
                         implicit_purposes: List[PurposeCandidate],
                         control_concepts: List[SemanticConcept],
                         control_id: Optional[str] = None) -> RiskAlignment:
        """Analyze alignment between control and risk."""
        risk_doc = self.nlp(risk_description.lower())
        risk_concepts = TextAnalyzer.extract_semantic_concepts(risk_doc)
        risk_aspects = self._extract_risk_aspects(risk_description)

        # Identify relationships
        relationships = self._identify_relationships(control_concepts, risk_concepts, text, risk_description)

        # Handle special approval-change case
        if self._is_approval_change_case(text, risk_description):
            relationships.append({
                "relationship": "addresses_unauthorized_changes",
                "score": 0.95,
                "description": "Control implements approval process to address unauthorized changes"
            })

            # Add derived purpose
            explicit_purposes.append(PurposeCandidate(
                text=f"to prevent {risk_description.lower()}",
                method="derived_from_risk",
                score=0.85,
                span=(0, len(text)),
                context="Derived from risk description"
            ))

        # Calculate alignment score and generate feedback
        alignment_score, feedback = self._calculate_alignment_score(
            relationships, text, risk_description, risk_aspects, control_id
        )

        return RiskAlignment(
            score=alignment_score,
            feedback=feedback,
            relationships=relationships,
            term_overlap=self._calculate_term_overlap(text, risk_description),
            risk_aspects=risk_aspects
        )

    def _extract_risk_aspects(self, risk_text: str) -> List[str]:
        """Extract risk components for partial matching."""
        if not risk_text:
            return []

        aspects = []

        # Split on "and" for compound risks
        if " and " in risk_text:
            parts = [part.strip() for part in risk_text.split(" and ") if len(part.split()) > 3]
            aspects.extend(parts)

        # Split on impact markers
        impact_markers = ["resulting in", "leading to", "causing", "which may cause", "which could result in"]
        for marker in impact_markers:
            if marker in risk_text.lower():
                parts = risk_text.lower().split(marker)
                if len(parts) > 1:
                    aspects.extend([parts[0].strip(), parts[1].strip()])

        # Split by sentences
        if not aspects:
            sentences = [s.strip() for s in re.split(r'[.;]', risk_text) if len(s.strip()) > 10]
            aspects.extend(sentences)

        return aspects if aspects else [risk_text]

    def _identify_relationships(self, control_concepts: List[SemanticConcept],
                              risk_concepts: List[SemanticConcept],
                              control_text: str, risk_text: str) -> List[Dict]:
        """Identify semantic relationships between control and risk concepts."""
        relationships = []

        # Define relationship patterns
        patterns = [
            {
                "control_type": "action", "risk_type": "negation",
                "match_fn": lambda c, r: (c.lemma == r.target or
                                        any(obj["lemma"] == r.target for obj in c.objects)),
                "score": 0.9, "relationship": "mitigates_negation"
            },
            {
                "control_type": "attribute", "risk_type": "attribute",
                "match_fn": lambda c, r: (hasattr(c, 'modifiers') and hasattr(r, 'target') and
                                        (getattr(c, 'modifiers', None) == getattr(r, 'target', None) or
                                         c.target == r.target)),
                "score": 0.8, "relationship": "implements_attribute"
            },
            {
                "control_type": "action", "risk_type": "action",
                "match_fn": lambda c, r: (c.lemma == r.lemma or
                                        any(c.lemma == obj["lemma"] for obj in r.objects)),
                "score": 0.7, "relationship": "verb_match"
            }
        ]

        # Check relationships
        for c_concept in control_concepts:
            for r_concept in risk_concepts:
                for pattern in patterns:
                    if (c_concept.concept_type == pattern["control_type"] and
                        r_concept.concept_type == pattern["risk_type"] and
                        pattern["match_fn"](c_concept, r_concept)):

                        relationships.append({
                            "control_concept": c_concept,
                            "risk_concept": r_concept,
                            "relationship": pattern["relationship"],
                            "score": pattern["score"],
                            "description": f"Control {pattern['relationship']} in risk"
                        })

        # Check approval-authorization special case
        if self._has_approval_authorization_relationship(control_text, risk_text):
            relationships.append({
                "relationship": "approval_authorization",
                "score": 0.85,
                "description": "Control approval_authorization in risk"
            })

        return relationships

    def _is_approval_change_case(self, control_text: str, risk_text: str) -> bool:
        """Check for approval controls addressing unauthorized changes."""
        approval_terms = ["approv", "authoriz", "review"]
        change_terms = ["chang", "modif", "updat"]

        control_has_approval = any(term in control_text.lower() for term in approval_terms)
        control_has_changes = any(term in control_text.lower() for term in change_terms)
        risk_has_approval = any(term in risk_text.lower() for term in approval_terms)
        risk_has_changes = any(term in risk_text.lower() for term in change_terms)
        risk_has_negation = "without" in risk_text.lower() or "no " in risk_text.lower()

        return (control_has_approval and control_has_changes and
                risk_has_approval and risk_has_changes and risk_has_negation)

    def _has_approval_authorization_relationship(self, control_text: str, risk_text: str) -> bool:
        """Check for approval-authorization relationship."""
        return (("approv" in control_text.lower() and "approv" in risk_text.lower()) or
                ("authoriz" in control_text.lower() and "authoriz" in risk_text.lower()))

    def _calculate_alignment_score(self, relationships: List[Dict], control_text: str,
                                 risk_description: str, risk_aspects: List[str],
                                 control_id: Optional[str]) -> Tuple[float, str]:
        """Calculate alignment score and generate feedback."""
        if not relationships:
            return 0.1, self._generate_weak_alignment_feedback(risk_description, control_text, control_id)

        max_rel_score = max(rel["score"] for rel in relationships)
        term_overlap = self._calculate_term_overlap(control_text, risk_description)
        alignment_score = (0.7 * max_rel_score) + (0.3 * term_overlap)

        if alignment_score >= 0.7:
            top_rel = max(relationships, key=lambda x: x["score"])
            feedback = f"Strong alignment with mapped risk. Control {top_rel.get('description', 'addresses the risk directly')}."
        elif alignment_score >= 0.4:
            feedback = "Moderate alignment with mapped risk."
            if len(risk_aspects) > 1:
                feedback += f" Primarily addresses: '{risk_aspects[0]}'."
        else:
            feedback = self._generate_weak_alignment_feedback(risk_description, control_text, control_id)

        return alignment_score, feedback

    def _calculate_term_overlap(self, control_text: str, risk_description: str) -> float:
        """Calculate term overlap between control and risk."""
        control_terms = set(t.lemma_.lower() for t in self.nlp(control_text)
                           if not t.is_stop and t.pos_ in ["NOUN", "VERB", "ADJ"])
        risk_terms = set(t.lemma_.lower() for t in self.nlp(risk_description)
                        if not t.is_stop and t.pos_ in ["NOUN", "VERB", "ADJ"])

        return len(control_terms.intersection(risk_terms)) / len(risk_terms) if risk_terms else 0

    def _generate_weak_alignment_feedback(self, risk_description: str, control_text: str,
                                        control_id: Optional[str]) -> str:
        """Generate feedback for weak alignment."""
        feedback = f"Weak alignment with mapped risk: '{risk_description}'. Consider explicitly addressing how this control mitigates this specific risk."

        if "approval" in risk_description.lower() and "approval" not in control_text.lower():
            feedback += " Consider explicitly mentioning the approval process."
        elif "change" in risk_description.lower() and "change" not in control_text.lower():
            feedback += " Consider explicitly addressing the change management aspect."

        if control_id:
            feedback += f" (Control {control_id})"

        return feedback


# =============================================================================
# QUALITY ANALYZERS
# =============================================================================

class QualityAnalyzer:
    """Analyzes quality aspects of purpose statements."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def identify_vague_phrases(self, candidates: List[PurposeCandidate]) -> List[Dict]:
        """Identify and penalize vague purpose phrases."""
        vague_phrases = []

        for candidate in candidates:
            candidate_text = candidate.text.lower()

            for vague_term in self.config.vague_terms:
                if vague_term.lower() in candidate_text:
                    vague_phrases.append({
                        "text": candidate_text,
                        "vague_term": vague_term,
                        "suggested_replacement": "specific risk, impact, or compliance requirement"
                    })

                    # Apply penalty
                    candidate.score = candidate.score * DetectionThresholds.VAGUE_PENALTY
                    candidate.has_vague_term = True

        return vague_phrases

    def detect_success_criteria(self, text: str) -> bool:
        """Detect specific success criteria in control."""
        return any(re.search(pattern, text, re.IGNORECASE)
                  for pattern in PatternLibrary.SUCCESS_CRITERIA_PATTERNS)

    def detect_mitigation_verbs(self, text: str) -> bool:
        """Detect actual mitigation verbs."""
        return any(re.search(r"\b" + re.escape(verb.lower()) + r"\b", text.lower())
                  for verb in self.config.mitigation_verbs)


# =============================================================================
# CLASSIFICATION AND CATEGORIZATION
# =============================================================================

class PurposeClassifier:
    """Classifies purpose intent and categorizes purposes."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def classify_intent(self, top_match: Optional[PurposeCandidate], doc,
                       risk_description: Optional[str]) -> Optional[str]:
        """Classify purpose intent category."""
        if not top_match:
            return None

        combined_text = self._get_combined_text(top_match)

        # Check intent verbs in top match
        for intent, verbs in self.config.intent_verbs.items():
            if any(verb in combined_text for verb in verbs):
                return intent.capitalize()

        # Check category keywords
        for category, keywords in self.config.categories.items():
            if any(keyword in combined_text for keyword in keywords):
                return category.replace("_", " ").capitalize()

        # Check document verbs
        doc_verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
        for intent, verbs in self.config.intent_verbs.items():
            if any(verb in doc_verbs for verb in verbs):
                return intent.capitalize()

        return "Risk mitigation" if risk_description else None

    def categorize_purpose(self, top_match: Optional[PurposeCandidate]) -> Optional[str]:
        """Categorize purpose into predefined categories."""
        if not top_match:
            return None

        combined_text = self._get_combined_text(top_match)

        for category, keywords in self.config.categories.items():
            if any(keyword in combined_text for keyword in keywords):
                return category.replace("_", " ").capitalize()

        return None

    def _get_combined_text(self, top_match: PurposeCandidate) -> str:
        """Get combined text from candidate."""
        text = top_match.text.lower()
        implied_purpose = getattr(top_match, 'implied_purpose', '') or ''
        return text + " " + implied_purpose.lower()


# =============================================================================
# SUGGESTION GENERATOR
# =============================================================================

class SuggestionGenerator:
    """Generates improvement suggestions for WHY elements."""

    @staticmethod
    def generate_suggestions(explicit_purposes: List[PurposeCandidate],
                           implicit_purposes: List[PurposeCandidate],
                           vague_phrases: List[Dict],
                           risk_alignment: Optional[RiskAlignment],
                           top_match: Optional[PurposeCandidate],
                           is_multi_control: bool = False) -> List[str]:
        """Generate comprehensive improvement suggestions."""
        suggestions = []

        # Missing purpose suggestions
        suggestions.extend(SuggestionGenerator._missing_purpose_suggestions(top_match))

        # Vague term suggestions
        suggestions.extend(SuggestionGenerator._vague_term_suggestions(vague_phrases))

        # Risk alignment suggestions
        suggestions.extend(SuggestionGenerator._risk_alignment_suggestions(risk_alignment))

        # Implicit purpose suggestions
        suggestions.extend(SuggestionGenerator._implicit_purpose_suggestions(
            top_match, explicit_purposes, implicit_purposes))

        # Multi-control suggestions
        suggestions.extend(SuggestionGenerator._multi_control_suggestions(
            is_multi_control, explicit_purposes))

        return suggestions

    @staticmethod
    def _missing_purpose_suggestions(top_match: Optional[PurposeCandidate]) -> List[str]:
        """Generate suggestions for missing purposes."""
        if not top_match:
            return ["No clear purpose or objective detected. Add an explicit statement of why the control exists."]
        return []

    @staticmethod
    def _vague_term_suggestions(vague_phrases: List[Dict]) -> List[str]:
        """Generate suggestions for vague terms."""
        return [f"Replace vague term '{vague['vague_term']}' with {vague['suggested_replacement']}."
                for vague in vague_phrases]

    @staticmethod
    def _risk_alignment_suggestions(risk_alignment: Optional[RiskAlignment]) -> List[str]:
        """Generate risk alignment suggestions."""
        if risk_alignment and risk_alignment.score < 0.4:
            feedback = risk_alignment.feedback
            if "Consider" in feedback:
                suggestion = feedback.split("Consider")[1].split("(")[0].strip()
                return [f"Consider{suggestion}"]
        return []

    @staticmethod
    def _implicit_purpose_suggestions(top_match: Optional[PurposeCandidate],
                                    explicit_purposes: List[PurposeCandidate],
                                    implicit_purposes: List[PurposeCandidate]) -> List[str]:
        """Generate suggestions for implicit purposes."""
        suggestions = []

        if top_match and top_match.method in ["inferred_from_action", "temporal_pattern"]:
            suggestions.append("Make the control purpose explicit by adding a clear statement of why the control exists.")

        if not explicit_purposes and implicit_purposes:
            top_implicit = implicit_purposes[0]
            if hasattr(top_implicit, 'implied_purpose') and top_implicit.implied_purpose:
                suggestions.append(f"Add explicit purpose statement: '{top_implicit.implied_purpose}'")

        return suggestions

    @staticmethod
    def _multi_control_suggestions(is_multi_control: bool, explicit_purposes: List[PurposeCandidate]) -> List[str]:
        """Generate multi-control suggestions."""
        if not is_multi_control:
            return []

        if len(explicit_purposes) > 1:
            return ["Multiple purpose statements detected. Consider separating into distinct controls or clarifying the primary purpose."]
        elif len(explicit_purposes) <= 1:
            return ["Multi-control description detected, but not all controls have clear purposes. Add a specific purpose for each control."]

        return []


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class WhyDetectionOrchestrator:
    """Main orchestrator for WHY detection process."""

    def __init__(self, nlp, config: Optional[Union[Dict, List]] = None):
        self.nlp = nlp
        base_config = ConfigurationManager.get_default_config()
        self.config = ConfigurationManager.merge_config(base_config, config)

        # Initialize components
        self.purpose_detector = PurposeDetector(self.config)
        self.risk_analyzer = RiskAlignmentAnalyzer(nlp, self.config)
        self.quality_analyzer = QualityAnalyzer(self.config)
        self.classifier = PurposeClassifier(self.config)

    def detect_purposes(self, text: str, risk_description: Optional[str] = None,
                       control_id: Optional[str] = None) -> DetectionResult:
        """Main detection method - orchestrates the entire process."""
        if not text or text.strip() == '':
            return self._create_empty_result()

        # Phase 1: Text analysis and segmentation
        segments = TextAnalyzer.split_multi_control_description(text)
        is_multi_control = len(segments) > 1

        # Phase 2: Process each segment
        segment_results = []
        for segment in segments:
            segment_result = self._process_segment(segment, risk_description, control_id)
            segment_results.append(segment_result)

        # Phase 3: Combine results
        combined_results = self._combine_segment_results(segment_results, is_multi_control)

        # Phase 4: Generate final analysis
        return self._generate_final_result(combined_results, text, is_multi_control)

    def _process_segment(self, segment: str, risk_description: Optional[str],
                        control_id: Optional[str]) -> SegmentResult:
        """Process a single control segment."""
        doc = self.nlp(segment)

        # Extract purposes
        explicit_purposes = self.purpose_detector.detect_explicit_purposes(segment, doc)
        explicit_purposes = EscalationTargetFilter.filter_candidates(explicit_purposes, segment, self.nlp)

        # Extract semantic concepts and implicit purposes
        control_concepts = TextAnalyzer.extract_semantic_concepts(doc)
        implicit_purposes = self.purpose_detector.detect_implicit_purposes(control_concepts, segment)

        # Analyze quality
        vague_phrases = self.quality_analyzer.identify_vague_phrases(explicit_purposes)

        # Risk alignment analysis
        risk_alignment = None
        if risk_description:
            risk_alignment = self.risk_analyzer.analyze_alignment(
                segment, risk_description, explicit_purposes, implicit_purposes,
                control_concepts, control_id
            )

        # Select best purpose and classify
        top_match, score = self._select_best_purpose(explicit_purposes, implicit_purposes, risk_alignment)
        intent_category = self.classifier.classify_intent(top_match, doc, risk_description)

        return SegmentResult(
            text=segment,
            explicit_purposes=explicit_purposes,
            implicit_purposes=implicit_purposes,
            top_match=top_match,
            score=score,
            vague_phrases=vague_phrases,
            intent_category=intent_category,
            risk_alignment=risk_alignment
        )

    def _select_best_purpose(self, explicit_purposes: List[PurposeCandidate],
                           implicit_purposes: List[PurposeCandidate],
                           risk_alignment: Optional[RiskAlignment]) -> Tuple[Optional[PurposeCandidate], float]:
        """Select the best purpose match and calculate confidence."""
        explicit_purposes.sort(key=lambda x: x.score, reverse=True)
        implicit_purposes.sort(key=lambda x: x.score, reverse=True)

        if explicit_purposes:
            top_match = explicit_purposes[0]
            confidence = top_match.score

            # Boost for explicit purpose statements
            if "purpose of this control" in top_match.text.lower():
                confidence = min(1.0, confidence * 1.1)
        elif implicit_purposes:
            top_match = implicit_purposes[0]
            confidence = top_match.score * DetectionThresholds.IMPLICIT_DISCOUNT
        else:
            top_match = None
            confidence = 0

        # Risk alignment boost
        if (risk_alignment and risk_alignment.score > DetectionThresholds.RISK_BOOST_THRESHOLD and
            confidence < DetectionThresholds.RISK_BOOST_MINIMUM):
            confidence = max(confidence, DetectionThresholds.RISK_BOOST_MINIMUM)

        # Add derived purpose from risk if needed
        if (not top_match and risk_alignment and risk_alignment.score > 0):
            risk_text = risk_alignment.feedback.split("with mapped risk:")[-1].split(".")[0].strip()
            if risk_text:
                top_match = PurposeCandidate(
                    text=f"to prevent {risk_text}",
                    implied_purpose=f"to prevent {risk_text}",
                    method="derived_from_risk",
                    score=0.5
                )

        return top_match, confidence

    def _combine_segment_results(self, segment_results: List[SegmentResult],
                               is_multi_control: bool) -> Dict:
        """Combine results from multiple segments."""
        if not segment_results:
            return self._get_empty_combined_results()

        if len(segment_results) == 1:
            result = segment_results[0]
            return {
                "explicit_why": result.explicit_purposes,
                "implicit_why": result.implicit_purposes,
                "top_match": result.top_match,
                "score": result.score,
                "vague_why_phrases": result.vague_phrases,
                "intent_category": result.intent_category,
                "risk_alignment": result.risk_alignment,
                "is_inferred": self._is_inferred_purpose(result.top_match)
            }

        # Combine multiple segments
        all_explicit = []
        all_implicit = []
        all_vague = []

        for result in segment_results:
            all_explicit.extend(result.explicit_purposes)
            all_implicit.extend(result.implicit_purposes)
            all_vague.extend(result.vague_phrases)

        # Find best match across segments
        best_score = 0
        best_match = None
        best_intent = None
        best_risk_alignment = None

        for result in segment_results:
            if result.top_match and result.score > best_score:
                best_score = result.score
                best_match = result.top_match
                best_intent = result.intent_category
                best_risk_alignment = result.risk_alignment

        return {
            "explicit_why": all_explicit,
            "implicit_why": all_implicit,
            "top_match": best_match,
            "score": best_score,
            "vague_why_phrases": all_vague,
            "intent_category": best_intent,
            "risk_alignment": best_risk_alignment,
            "is_inferred": self._is_inferred_purpose(best_match)
        }

    def _is_inferred_purpose(self, top_match: Optional[PurposeCandidate]) -> bool:
        """Check if purpose is inferred."""
        return (top_match and top_match.method in
               ["inferred_from_action", "temporal_pattern", "derived_from_risk"])

    def _get_empty_combined_results(self) -> Dict:
        """Get empty combined results structure."""
        return {
            "explicit_why": [],
            "implicit_why": [],
            "top_match": None,
            "score": 0,
            "vague_why_phrases": [],
            "intent_category": None,
            "risk_alignment": None,
            "is_inferred": False
        }

    def _generate_final_result(self, combined_results: Dict, text: str,
                             is_multi_control: bool) -> DetectionResult:
        """Generate the final detection result."""
        # Extract keywords
        extracted_keywords = []
        if combined_results["explicit_why"]:
            extracted_keywords = [p.text for p in combined_results["explicit_why"]]
        elif (combined_results["top_match"] and
              hasattr(combined_results["top_match"], 'implied_purpose') and
              combined_results["top_match"].implied_purpose):
            extracted_keywords = [combined_results["top_match"].implied_purpose]

        # Categorize purpose
        why_category = self.classifier.categorize_purpose(combined_results["top_match"])

        # Quality checks
        has_success_criteria = self.quality_analyzer.detect_success_criteria(text)
        is_actual_mitigation = self.quality_analyzer.detect_mitigation_verbs(text)

        # Generate suggestions
        improvement_suggestions = SuggestionGenerator.generate_suggestions(
            combined_results["explicit_why"],
            combined_results["implicit_why"],
            combined_results["vague_why_phrases"],
            combined_results.get("risk_alignment"),
            combined_results["top_match"],
            is_multi_control
        )

        # Extract risk alignment details
        risk_alignment = combined_results.get("risk_alignment")
        risk_alignment_score = risk_alignment.score if risk_alignment else None
        risk_alignment_feedback = risk_alignment.feedback if risk_alignment else None

        return DetectionResult(
            explicit_why=combined_results["explicit_why"],
            implicit_why=combined_results["implicit_why"],
            top_match=combined_results["top_match"],
            why_category=why_category,
            score=combined_results["score"],
            is_inferred=combined_results.get("is_inferred", False),
            risk_alignment_score=risk_alignment_score,
            risk_alignment_feedback=risk_alignment_feedback,
            extracted_keywords=extracted_keywords,
            has_success_criteria=has_success_criteria,
            vague_why_phrases=combined_results["vague_why_phrases"],
            is_actual_mitigation=is_actual_mitigation,
            intent_category=combined_results["intent_category"],
            improvement_suggestions=improvement_suggestions,
            is_multi_control=is_multi_control
        )

    def _create_empty_result(self) -> DetectionResult:
        """Create empty result for invalid input."""
        return DetectionResult(
            explicit_why=[],
            implicit_why=[],
            top_match=None,
            why_category=None,
            score=0,
            is_inferred=False,
            risk_alignment_score=None,
            risk_alignment_feedback=None,
            extracted_keywords=[],
            has_success_criteria=False,
            vague_why_phrases=[],
            is_actual_mitigation=False,
            intent_category=None,
            improvement_suggestions=["No clear purpose statement detected in the control description."],
            is_multi_control=False
        )


# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

def enhance_why_detection(text: str, nlp, risk_description: Optional[str] = None,
                          config: Optional[Dict] = None, control_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced WHY detection with improved pattern recognition and contextual understanding.

    This is the main public API function that maintains backward compatibility
    with the original interface while using the refactored internal structure.

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        risk_description: Optional risk description text for alignment analysis
        config: Optional configuration dictionary
        control_id: Optional control identifier for reference

    Returns:
        Dictionary with comprehensive WHY detection results
    """
    # Create orchestrator with configuration
    orchestrator = WhyDetectionOrchestrator(nlp, config)

    # Perform detection
    result = orchestrator.detect_purposes(text, risk_description, control_id)

    # Convert result back to dictionary format for backward compatibility
    return _convert_result_to_dict(result)


def _convert_result_to_dict(result: DetectionResult) -> Dict[str, Any]:
    """Convert DetectionResult to dictionary for backward compatibility."""
    return {
        "explicit_why": [_convert_candidate_to_dict(c) for c in result.explicit_why],
        "implicit_why": [_convert_candidate_to_dict(c) for c in result.implicit_why],
        "top_match": _convert_candidate_to_dict(result.top_match) if result.top_match else None,
        "why_category": result.why_category,
        "score": result.score,
        "is_inferred": result.is_inferred,
        "risk_alignment_score": result.risk_alignment_score,
        "risk_alignment_feedback": result.risk_alignment_feedback,
        "extracted_keywords": result.extracted_keywords,
        "has_success_criteria": result.has_success_criteria,
        "vague_why_phrases": result.vague_why_phrases,
        "is_actual_mitigation": result.is_actual_mitigation,
        "intent_category": result.intent_category,
        "improvement_suggestions": result.improvement_suggestions,
        "is_multi_control": result.is_multi_control
    }


def _convert_candidate_to_dict(candidate: Optional[PurposeCandidate]) -> Optional[Dict[str, Any]]:
    """Convert PurposeCandidate to dictionary."""
    if not candidate:
        return None

    result = {
        "text": candidate.text,
        "method": candidate.method,
        "score": candidate.score,
        "span": candidate.span,
        "context": candidate.context
    }

    # Add optional fields if present
    if candidate.verb:
        result["verb"] = candidate.verb
    if candidate.implied_purpose:
        result["implied_purpose"] = candidate.implied_purpose
    if candidate.has_vague_term:
        result["has_vague_term"] = candidate.has_vague_term

    return result


# =============================================================================
# LEGACY FUNCTION STUBS FOR BACKWARD COMPATIBILITY
# =============================================================================

def get_why_config(config: Optional[Union[Dict, List]]) -> Dict:
    """Legacy function for backward compatibility."""
    base_config = ConfigurationManager.get_default_config()
    merged_config = ConfigurationManager.merge_config(base_config, config)

    # Convert back to dictionary format
    return {
        "weight": merged_config.weight,
        "keywords": merged_config.keywords,
        "purpose_patterns": merged_config.purpose_patterns,
        "vague_terms": merged_config.vague_terms,
        "intent_verbs": merged_config.intent_verbs,
        "mitigation_verbs": merged_config.mitigation_verbs,
        "verb_purpose_mapping": merged_config.verb_purpose_mapping,
        "categories": merged_config.categories,
        "confidence_thresholds": merged_config.confidence_thresholds
    }


def create_empty_result() -> Dict:
    """Legacy function for backward compatibility."""
    orchestrator = WhyDetectionOrchestrator(None)
    result = orchestrator._create_empty_result()
    return _convert_result_to_dict(result)


def split_multi_control_description(text: str) -> List[str]:
    """Legacy function for backward compatibility."""
    return TextAnalyzer.split_multi_control_description(text)


def extract_semantic_concepts(doc) -> List[Dict]:
    """Legacy function for backward compatibility."""
    concepts = TextAnalyzer.extract_semantic_concepts(doc)
    return [_convert_concept_to_dict(concept) for concept in concepts]


def _convert_concept_to_dict(concept: SemanticConcept) -> Dict[str, Any]:
    """Convert SemanticConcept to dictionary."""
    result = {
        "type": concept.concept_type,
        "text": concept.text,
        "lemma": concept.lemma
    }

    if concept.modifiers:
        result["modifiers"] = concept.modifiers
    if concept.objects:
        result["objects"] = concept.objects
    if concept.target:
        result["target"] = concept.target
    if concept.negates:
        result["negates"] = concept.negates

    return result


def detect_explicit_purposes(text: str, doc, config: Dict) -> List[Dict]:
    """Legacy function for backward compatibility."""
    detection_config = DetectionConfig(**config)
    detector = PurposeDetector(detection_config)
    candidates = detector.detect_explicit_purposes(text, doc)
    return [_convert_candidate_to_dict(c) for c in candidates]


def filter_escalation_targets(purpose_candidates: List[Dict], text: str, nlp) -> List[Dict]:
    """Legacy function for backward compatibility."""
    candidates = [_convert_dict_to_candidate(c) for c in purpose_candidates]
    filtered = EscalationTargetFilter.filter_candidates(candidates, text, nlp)
    return [_convert_candidate_to_dict(c) for c in filtered]


def _convert_dict_to_candidate(candidate_dict: Dict) -> PurposeCandidate:
    """Convert dictionary to PurposeCandidate."""
    return PurposeCandidate(
        text=candidate_dict.get("text", ""),
        method=candidate_dict.get("method", ""),
        score=candidate_dict.get("score", 0.0),
        span=tuple(candidate_dict.get("span", (0, 0))),
        context=candidate_dict.get("context", ""),
        verb=candidate_dict.get("verb"),
        implied_purpose=candidate_dict.get("implied_purpose"),
        has_vague_term=candidate_dict.get("has_vague_term", False)
    )


def infer_implicit_purposes(action_concepts: List[Dict], text: str, config: Dict) -> List[Dict]:
    """Legacy function for backward compatibility."""
    semantic_concepts = [_convert_dict_to_concept(c) for c in action_concepts]
    detection_config = DetectionConfig(**config)
    detector = PurposeDetector(detection_config)
    candidates = detector.detect_implicit_purposes(semantic_concepts, text)
    return [_convert_candidate_to_dict(c) for c in candidates]


def _convert_dict_to_concept(concept_dict: Dict) -> SemanticConcept:
    """Convert dictionary to SemanticConcept."""
    return SemanticConcept(
        concept_type=concept_dict.get("type", ""),
        text=concept_dict.get("text", ""),
        lemma=concept_dict.get("lemma", ""),
        modifiers=concept_dict.get("modifiers", []),
        objects=concept_dict.get("objects", []),
        target=concept_dict.get("target"),
        negates=concept_dict.get("negates")
    )


def find_temporal_prevention_patterns(text: str) -> List[Dict]:
    """Legacy function for backward compatibility."""
    detection_config = ConfigurationManager.get_default_config()
    detector = PurposeDetector(detection_config)
    candidates = detector._detect_temporal_patterns(text)
    return [_convert_candidate_to_dict(c) for c in candidates]


def combine_segment_results(segment_results: List[Dict], is_multi_control: bool) -> Dict:
    """Legacy function for backward compatibility."""
    # This function would need more complex conversion logic
    # For now, return a basic structure
    if not segment_results:
        return {
            "explicit_why": [],
            "implicit_why": [],
            "top_match": None,
            "score": 0,
            "vague_why_phrases": [],
            "intent_category": None,
            "risk_alignment": None
        }

    # Simplified combination logic for backward compatibility
    return segment_results[0] if segment_results else {}


def identify_vague_phrases(purpose_candidates: List[Dict], config: Dict) -> List[Dict]:
    """Legacy function for backward compatibility."""
    detection_config = DetectionConfig(**config)
    analyzer = QualityAnalyzer(detection_config)
    candidates = [_convert_dict_to_candidate(c) for c in purpose_candidates]
    return analyzer.identify_vague_phrases(candidates)


def align_with_risk_description(text: str, risk_description: str,
                                explicit_purposes: List[Dict], implicit_purposes: List[Dict],
                                control_concepts: List[Dict], nlp, config: Dict,
                                control_id: Optional[str] = None) -> Dict:
    """Legacy function for backward compatibility."""
    detection_config = DetectionConfig(**config)
    analyzer = RiskAlignmentAnalyzer(nlp, detection_config)

    # Convert inputs
    explicit_candidates = [_convert_dict_to_candidate(c) for c in explicit_purposes]
    implicit_candidates = [_convert_dict_to_candidate(c) for c in implicit_purposes]
    semantic_concepts = [_convert_dict_to_concept(c) for c in control_concepts]

    # Perform analysis
    alignment = analyzer.analyze_alignment(
        text, risk_description, explicit_candidates, implicit_candidates,
        semantic_concepts, control_id
    )

    # Convert back to dictionary
    return {
        "score": alignment.score,
        "feedback": alignment.feedback,
        "relationships": alignment.relationships,
        "term_overlap": alignment.term_overlap,
        "risk_aspects": alignment.risk_aspects
    }


def select_best_purpose(explicit_purposes: List[Dict], implicit_purposes: List[Dict],
                        risk_alignment: Optional[Dict], config: Dict) -> Tuple[Optional[Dict], float]:
    """Legacy function for backward compatibility."""
    orchestrator = WhyDetectionOrchestrator(None, config)

    # Convert inputs
    explicit_candidates = [_convert_dict_to_candidate(c) for c in explicit_purposes]
    implicit_candidates = [_convert_dict_to_candidate(c) for c in implicit_purposes]

    # Convert risk alignment
    risk_align = None
    if risk_alignment:
        risk_align = RiskAlignment(
            score=risk_alignment.get("score", 0),
            feedback=risk_alignment.get("feedback", ""),
            relationships=risk_alignment.get("relationships", []),
            term_overlap=risk_alignment.get("term_overlap", 0),
            risk_aspects=risk_alignment.get("risk_aspects", [])
        )

    # Perform selection
    top_match, confidence = orchestrator._select_best_purpose(
        explicit_candidates, implicit_candidates, risk_align
    )

    return _convert_candidate_to_dict(top_match), confidence


def classify_purpose_intent(top_match: Optional[Dict], doc,
                           risk_description: Optional[str], config: Dict) -> Optional[str]:
    """Legacy function for backward compatibility."""
    detection_config = DetectionConfig(**config)
    classifier = PurposeClassifier(detection_config)

    candidate = _convert_dict_to_candidate(top_match) if top_match else None
    return classifier.classify_intent(candidate, doc, risk_description)


def categorize_purpose(top_match: Optional[Dict], config: Dict) -> Optional[str]:
    """Legacy function for backward compatibility."""
    detection_config = DetectionConfig(**config)
    classifier = PurposeClassifier(detection_config)

    candidate = _convert_dict_to_candidate(top_match) if top_match else None
    return classifier.categorize_purpose(candidate)


def generate_improvement_suggestions(explicit_purposes: List[Dict],
                                     implicit_purposes: List[Dict],
                                     vague_phrases: List[Dict],
                                     risk_alignment: Optional[Dict],
                                     top_match: Optional[Dict],
                                     config: Dict,
                                     is_multi_control: bool = False) -> List[str]:
    """Legacy function for backward compatibility."""
    # Convert inputs
    explicit_candidates = [_convert_dict_to_candidate(c) for c in explicit_purposes]
    implicit_candidates = [_convert_dict_to_candidate(c) for c in implicit_purposes]

    risk_align = None
    if risk_alignment:
        risk_align = RiskAlignment(
            score=risk_alignment.get("score", 0),
            feedback=risk_alignment.get("feedback", ""),
            relationships=risk_alignment.get("relationships", []),
            term_overlap=risk_alignment.get("term_overlap", 0),
            risk_aspects=risk_alignment.get("risk_aspects", [])
        )

    candidate = _convert_dict_to_candidate(top_match) if top_match else None

    return SuggestionGenerator.generate_suggestions(
        explicit_candidates, implicit_candidates, vague_phrases,
        risk_align, candidate, is_multi_control
    )


def detect_success_criteria(text: str) -> bool:
    """Legacy function for backward compatibility."""
    config = ConfigurationManager.get_default_config()
    analyzer = QualityAnalyzer(config)
    return analyzer.detect_success_criteria(text)


def detect_mitigation_verbs(text: str, config: Dict) -> bool:
    """Legacy function for backward compatibility."""
    detection_config = DetectionConfig(**config)
    analyzer = QualityAnalyzer(detection_config)
    return analyzer.detect_mitigation_verbs(text)