"""
Enhanced WHEN detection module with optimized performance and clean architecture.
Identifies and analyzes timing information in control descriptions.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TimingDetectionConfig:
    """Configuration for timing detection parameters."""
    control_type: Optional[str] = None
    frequency_metadata: Optional[str] = None


@dataclass
class TimingCandidate:
    """Represents a detected timing element candidate."""
    text: str
    method: str
    score: float
    span: List[int]
    is_primary: bool = False
    is_vague: bool = False
    is_semi_vague: bool = False
    frequency: Optional[str] = None
    pattern_type: Optional[str] = None
    context: Optional[str] = None


@dataclass
class VagueTermInfo:
    """Represents information about a vague timing term."""
    text: str
    span: List[int]
    suggested_replacement: str
    is_primary: bool = False


@dataclass
class ValidationResult:
    """Represents validation results against metadata."""
    is_valid: bool
    message: str


@dataclass
class TimingDetectionResult:
    """Complete result of timing detection analysis."""
    candidates: List[TimingCandidate]
    top_match: Optional[TimingCandidate]
    score: float
    extracted_keywords: List[str]
    multi_frequency_detected: bool
    frequencies: List[str]
    validation: ValidationResult
    vague_terms: List[VagueTermInfo]
    improvement_suggestions: List[str]
    specific_timing_found: bool = False
    primary_vague_term: bool = False


class TimingPatterns:
    """Pre-compiled regex patterns for timing detection."""

    def __init__(self, config: Dict[str, Any]):
        timing_config = config.get('timing_detection', {})
        patterns = timing_config.get('patterns', {})

        self.DAILY = re.compile(patterns.get('daily', r'\b(daily|each\s+day|every\s+day|on\s+a\s+daily\s+basis|day\b|daily\s+basis)\b'), re.IGNORECASE)
        self.WEEKLY = re.compile(patterns.get('weekly', r'\b(weekly|each\s+week|every\s+week|on\s+a\s+weekly\s+basis|week\b|weekly\s+basis)\b'), re.IGNORECASE)
        self.MONTHLY = re.compile(patterns.get('monthly', r'\b(monthly|each\s+month|every\s+month|on\s+a\s+monthly\s+basis|(?<!(?:at|by)\s)month(?!\s*[-]?\s*end)|monthly\s+basis)\b'), re.IGNORECASE)
        self.QUARTERLY = re.compile(patterns.get('quarterly', r'\b(quarterly|each\s+quarter|every\s+quarter|on\s+a\s+quarterly\s+basis|(?<!(?:at|by)\s)quarter(?!\s*[-]?\s*end)|once\s+per\s+quarter|each\s+fiscal\s+quarter|every\s+three\s+months)\b'), re.IGNORECASE)
        self.ANNUALLY = re.compile(patterns.get('annually', r'\b(annually|yearly|each\s+year|every\s+year|annual\b|on\s+an\s+annual\s+basis|(?<!(?:at|by)\s)year(?!\s*[-]?\s*end))\b'), re.IGNORECASE)
        self.ADHOC = re.compile(patterns.get('adhoc', r'\b(adhoc|ad[\s-]hoc|on\s+an\s+ad[\s-]hoc\s+basis)\b'), re.IGNORECASE)
        self.WEEKDAY = re.compile(patterns.get('weekday', r'\b(every|each|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b'), re.IGNORECASE)
        self.PERIOD_END = re.compile(patterns.get('period_end', r'\b(at|during|before|after|by)\s+(the\s+)?(fiscal|calendar)?\s*(year|quarter|month)[\s-]end(\s+close)?\b'), re.IGNORECASE)
        self.CLOSE_PERIOD = re.compile(patterns.get('close_period', r'\b(at|during)\s+(each|every|the)\s+closing\s+(cycle|period|process)\b'), re.IGNORECASE)
        self.TIMELINE = re.compile(patterns.get('timeline', r'\b(within|after|before|prior\s+to|following|upon|by)\s+'), re.IGNORECASE)
        self.VAGUE_TERMS = re.compile(patterns.get('vague_terms', r'\b(as\s+needed|when\s+needed|if\s+needed|as\s+appropriate|when\s+appropriate|if\s+appropriate|as\s+required|when\s+required|if\s+required|periodically|occasionally|from\s+time\s+to\s+time|regularly|timely|may\s+vary|may\s+change|may\s+differ|may\s+be|may\s+not|on\s+demand|as\s+and\s+when|upon\s+request|when\s+requested|non[\s-]scheduled)\b'), re.IGNORECASE)
        self.PROBLEMATIC_MAY = re.compile(patterns.get('problematic_may', r'\bmay\s+(?:vary|differ|change|be|not|need)\b'), re.IGNORECASE)
        self.BUSINESS_CYCLE = re.compile(patterns.get('business_cycle', r'\b(during|after|before|at|upon|following|as\s+part\s+of)\s+(the\s+)?(each|every)?\s*(audit|review|assessment|reporting|close|closing)\s+(cycle|period|process)\b'), re.IGNORECASE)
        self.PROCEDURE_REFERENCE = re.compile(patterns.get('procedure_reference', r'\b(defined|outlined|described|according|per|as\s+per|based|in\s+accordance)\s+(in|on|with|to)\s+(procedure|policy|document|standard)\b'), re.IGNORECASE)
        self.EVENT_TRIGGER = re.compile(patterns.get('event_trigger', r'\b(upon|after|when|following|immediately|promptly)\s+(receipt|notification|identification|detection|discovery|system|application|platform|database)\b'), re.IGNORECASE)
        self.WITHIN_TIMEFRAME = re.compile(patterns.get('within_timeframe', r'within\s+(\d+)\s+(day|week|month|business day|working day)s?'), re.IGNORECASE)


class TimingDetector:
    """Main timing detection class with extracted methods for each detection phase."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patterns = TimingPatterns(config)
        self.timing_config = config.get('timing_detection', {})
        self.scores = self.timing_config.get('scores', {})
        self.vague_suggestions = self.timing_config.get('vague_term_suggestions', {})
        self.penalties = config.get('penalties', {}).get('timing', {})

    def get_vague_term_suggestion(self, vague_term: str) -> str:
        """Get specific alternative suggestion for vague timing terms."""
        # Check configured suggestions first
        configured_suggestion = self.vague_suggestions.get(vague_term.lower())
        if configured_suggestion:
            return configured_suggestion
        
        # Fallback to default specific suggestions based on the enhancement guide
        default_suggestions = {
            'periodically': 'specific frequency (daily, weekly, monthly, quarterly)',
            'regularly': 'specific frequency (daily, weekly, monthly, quarterly)', 
            'occasionally': 'specific frequency (monthly, quarterly, annually) or event-based timing (upon notification, after review)',
            'as needed': 'specific trigger events (upon request, when threshold exceeded, if issues identified)',
            'when needed': 'specific trigger events (upon request, when threshold exceeded, if issues identified)',
            'if needed': 'specific trigger events (upon request, when threshold exceeded, if issues identified)',
            'as appropriate': 'specific timing criteria (monthly, quarterly, or upon specific events)',
            'when appropriate': 'specific timing criteria (monthly, quarterly, or upon specific events)',
            'if appropriate': 'specific timing criteria (monthly, quarterly, or upon specific events)',
            'as required': 'specific requirements or triggers (weekly, monthly, or upon notification)',
            'when required': 'specific requirements or triggers (weekly, monthly, or upon notification)',
            'if required': 'specific requirements or triggers (weekly, monthly, or upon notification)',
            'from time to time': 'specific frequency (monthly, quarterly, semi-annually)',
            'on demand': 'specific request processes (upon formal request, within X business days)',
            'as and when': 'specific triggering conditions and timing',
            'upon request': 'specific request processing timeframe (within X business days)',
            'when requested': 'specific request processing timeframe (within X business days)'
        }
        
        return default_suggestions.get(vague_term.lower(), "a specific timeframe or frequency")

    def _create_empty_result(self) -> TimingDetectionResult:
        """Create empty result for when no text is provided."""
        return TimingDetectionResult(
            candidates=[],
            top_match=None,
            score=0,
            extracted_keywords=[],
            multi_frequency_detected=False,
            frequencies=[],
            validation=ValidationResult(is_valid=False, message="No text provided"),
            vague_terms=[],
            improvement_suggestions=[]
        )

    def _create_vague_result(self, vague_term: str) -> TimingDetectionResult:
        """Create result for vague terms."""
        vague_score = self.penalties.get('vague_term_score', 0.1)

        candidate = TimingCandidate(
            text=vague_term,
            method="vague_timing",
            score=vague_score,
            span=[0, len(vague_term)],
            is_vague=True,
            is_primary=True
        )

        vague_info = VagueTermInfo(
            text=vague_term,
            span=[0, len(vague_term)],
            suggested_replacement=self.get_vague_term_suggestion(vague_term),
            is_primary=True
        )

        return TimingDetectionResult(
            candidates=[candidate],
            top_match=candidate,
            score=0,
            extracted_keywords=[vague_term],
            multi_frequency_detected=False,
            frequencies=[],
            validation=ValidationResult(is_valid=False, message="Vague timing detected"),
            vague_terms=[vague_info],
            improvement_suggestions=[
                f"Replace vague timing term '{vague_term}' with specific frequency (daily, weekly, monthly)."
            ],
            specific_timing_found=False,
            primary_vague_term=True
        )

    def _check_early_exit_conditions(self, normalized_text: str, has_adhoc: bool) -> Optional[TimingDetectionResult]:
        """Check for early exit conditions - bypass when ad-hoc timing is present."""
        if has_adhoc:
            return None

        # Handle problematic "may" (excluding month of May)
        if " may " in normalized_text:
            if not re.search(r'\b(?:in|of|during|by|for|before|after)\s+may\b', normalized_text):
                if not re.search(r'may\s+(?:\d{1,2}|\d{4})', normalized_text):
                    if self.patterns.PROBLEMATIC_MAY.search(normalized_text):
                        return self._create_vague_result("may")

        # Check if text starts with a vague term
        vague_start_match = self.patterns.VAGUE_TERMS.match(normalized_text)
        if vague_start_match:
            return self._create_vague_result(vague_start_match.group(0))

        # Check for procedure reference without timing
        if self._has_procedure_reference_only(normalized_text):
            return TimingDetectionResult(
                candidates=[],
                top_match=None,
                score=0,
                extracted_keywords=[],
                multi_frequency_detected=False,
                frequencies=[],
                validation=ValidationResult(is_valid=False, message="Only procedure reference without timing"),
                vague_terms=[],
                improvement_suggestions=[
                    "Add specific frequency (daily, weekly, monthly) instead of just referencing a procedure."
                ],
                specific_timing_found=False,
                primary_vague_term=False
            )

        return None

    def _has_procedure_reference_only(self, normalized_text: str) -> bool:
        """Check if text only contains procedure reference without timing."""
        has_procedure = self.patterns.PROCEDURE_REFERENCE.search(normalized_text)
        has_timing = any(
            pattern.search(normalized_text) for pattern in [
                self.patterns.DAILY, self.patterns.WEEKLY, self.patterns.MONTHLY,
                self.patterns.QUARTERLY, self.patterns.ANNUALLY, self.patterns.ADHOC
            ]
        )
        return has_procedure and not has_timing

    def _detect_vague_terms(self, normalized_text: str) -> List[VagueTermInfo]:
        """Find and record all vague terms."""
        vague_terms_found = []
        for match in self.patterns.VAGUE_TERMS.finditer(normalized_text):
            vague_terms_found.append(VagueTermInfo(
                text=match.group(),
                span=[match.start(), match.end()],
                suggested_replacement=self.get_vague_term_suggestion(match.group())
            ))
        return vague_terms_found

    def _detect_adhoc_timing(self, normalized_text: str) -> Optional[TimingCandidate]:
        """Handle ad-hoc timing detection."""
        ad_hoc_match = self.patterns.ADHOC.search(normalized_text)
        if not ad_hoc_match:
            return None

        start, end = ad_hoc_match.span()
        surrounding_text = normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]

        return TimingCandidate(
            text=ad_hoc_match.group(),
            method="adhoc_frequency",
            score=self.scores.get('adhoc_frequency', 0.7),
            span=[start, end],
            frequency="ad-hoc",
            is_primary=True,
            is_vague=False,
            is_semi_vague=True,
            context=surrounding_text
        )

    def _detect_frequency_patterns(self, normalized_text: str, vague_terms_found: List[VagueTermInfo]) -> List[TimingCandidate]:
        """Detect standard frequency patterns."""
        candidates = []
        frequency_checks = [
            (self.patterns.DAILY, "daily"),
            (self.patterns.WEEKLY, "weekly"),
            (self.patterns.MONTHLY, "monthly"),
            (self.patterns.QUARTERLY, "quarterly"),
            (self.patterns.ANNUALLY, "annually"),
        ]

        for pattern, freq_name in frequency_checks:
            for match in pattern.finditer(normalized_text):
                start, end = match.span()

                # Skip if this is part of a vague term
                if self._is_part_of_vague_term(start, end, vague_terms_found):
                    continue

                surrounding_text = normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]
                is_primary = self._is_primary_timing(surrounding_text)

                candidates.append(TimingCandidate(
                    text=match.group(),
                    method="explicit_frequency",
                    score=self.scores.get('explicit_frequency', 0.9),
                    span=[start, end],
                    frequency=freq_name,
                    is_primary=is_primary,
                    is_vague=False,
                    context=surrounding_text
                ))

        return candidates

    def _detect_weekday_patterns(self, normalized_text: str, vague_terms_found: List[VagueTermInfo]) -> List[TimingCandidate]:
        """Check for weekday patterns."""
        candidates = []
        for match in self.patterns.WEEKDAY.finditer(normalized_text):
            start, end = match.span()

            if self._is_part_of_vague_term(start, end, vague_terms_found):
                continue

            candidates.append(TimingCandidate(
                text=match.group(),
                method="weekly_schedule",
                score=self.scores.get('explicit_frequency', 0.9),
                span=[start, end],
                frequency="weekly",
                is_primary=True,
                is_vague=False,
                context=normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]
            ))

        return candidates

    def _detect_period_end_patterns(self, normalized_text: str, vague_terms_found: List[VagueTermInfo]) -> List[TimingCandidate]:
        """Check for period end patterns."""
        candidates = []

        # Period end patterns
        for match in self.patterns.PERIOD_END.finditer(normalized_text):
            start, end = match.span()

            if self._is_part_of_vague_term(start, end, vague_terms_found):
                continue

            period_type = re.search(r'(year|quarter|month)', match.group())

            if period_type and "year" in period_type.group():
                period_freq = "annually"
            elif period_type and "quarter" in period_type.group():
                period_freq = "quarterly"
            else:
                period_freq = "monthly"

            candidates.append(TimingCandidate(
                text=match.group(),
                method="period_end_pattern",
                score=self.scores.get('period_end_pattern', 0.85),
                span=[start, end],
                frequency=period_freq,
                is_primary=True,
                is_vague=False,
                context=normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]
            ))

        # Closing period patterns
        for match in self.patterns.CLOSE_PERIOD.finditer(normalized_text):
            start, end = match.span()

            if self._is_part_of_vague_term(start, end, vague_terms_found):
                continue

            candidates.append(TimingCandidate(
                text=match.group(),
                method="close_period_pattern",
                score=self.scores.get('close_period_pattern', 0.85),
                span=[start, end],
                frequency="monthly",
                is_primary=True,
                is_vague=False,
                context=normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]
            ))

        return candidates

    def _detect_business_cycle_patterns(self, normalized_text: str, vague_terms_found: List[VagueTermInfo]) -> List[TimingCandidate]:
        """Check for business cycle and event patterns."""
        candidates = []

        # Business cycle patterns
        for match in self.patterns.BUSINESS_CYCLE.finditer(normalized_text):
            start, end = match.span()

            if self._is_part_of_vague_term(start, end, vague_terms_found):
                continue

            candidates.append(TimingCandidate(
                text=match.group(),
                method="business_cycle_pattern",
                score=self.scores.get('business_cycle_pattern', 0.75),
                span=[start, end],
                pattern_type="business_cycle",
                is_primary=True,
                is_vague=False,
                context=normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]
            ))
            break

        # Event trigger patterns
        if not candidates:
            for match in self.patterns.EVENT_TRIGGER.finditer(normalized_text):
                start, end = match.span()

                if self._is_part_of_vague_term(start, end, vague_terms_found):
                    continue

                candidates.append(TimingCandidate(
                    text=match.group(),
                    method="event_trigger_pattern",
                    score=self.scores.get('event_trigger_pattern', 0.75),
                    span=[start, end],
                    pattern_type="event_trigger",
                    is_primary=True,
                    is_vague=False,
                    context=normalized_text[max(0, start - 30):min(len(normalized_text), end + 30)]
                ))
                break

        return candidates

    def _detect_implicit_timing(self, text: str, nlp, vague_terms_found: List[VagueTermInfo]) -> List[TimingCandidate]:
        """Use spaCy for implicit timing detection."""
        candidates = []
        doc = nlp(text)

        for token in doc:
            if token.dep_ == "npadvmod" and token.head.pos_ == "VERB":
                if any(time_word in token.text.lower() for time_word in
                       ["time", "moment", "instance", "point", "period"]):

                    token_start = token.idx
                    token_end = token.idx + len(token.text)

                    if self._is_part_of_vague_term(token_start, token_end, vague_terms_found):
                        continue

                    context_window = text.lower()[max(0, token_start - 5):min(len(text), token_end + 15)]
                    if self._is_excluded_context(context_window):
                        continue

                    candidates.append(TimingCandidate(
                        text=token.text,
                        method="implicit_temporal_modifier",
                        score=self.scores.get('implicit_temporal_modifier', 0.6),
                        span=[token.i, token.i + 1],
                        is_primary=False,
                        is_vague=False,
                        context=text[max(0, token.idx - 30):min(len(text), token.idx + len(token.text) + 30)]
                    ))

        return candidates

    def _validate_against_metadata(self, detected_frequencies: List[str], config: TimingDetectionConfig) -> ValidationResult:
        """Validate against metadata frequency if provided."""
        if not config.frequency_metadata:
            return ValidationResult(is_valid=True, message="No frequency metadata provided for validation")

        normalized_metadata = config.frequency_metadata.lower().strip()
        metadata_match = False

        # Handle ad-hoc specially for validation
        if "ad-hoc" in detected_frequencies:
            if any(term in normalized_metadata for term in ["ad-hoc", "ad hoc", "adhoc"]):
                metadata_match = True
        else:
            # Check standard frequencies
            for freq in detected_frequencies:
                if freq == normalized_metadata or self._frequency_pattern_matches(freq, normalized_metadata):
                    metadata_match = True
                    break

        if metadata_match:
            return ValidationResult(
                is_valid=True,
                message=f"Frequency in description matches metadata ({normalized_metadata})"
            )
        else:
            frequencies_str = ', '.join(detected_frequencies) if detected_frequencies else 'none'
            return ValidationResult(
                is_valid=False,
                message=f"Frequency in description ({frequencies_str}) does not match metadata ({normalized_metadata})"
            )

    def _calculate_final_score(self, timing_candidates: List[TimingCandidate],
                              vague_terms_found: List[VagueTermInfo],
                              specific_timing_found: bool,
                              primary_vague_term: bool,
                              config: TimingDetectionConfig) -> float:
        """Calculate final score based on detection results."""
        # Check for primary vague terms (vague terms that are the main timing indicator)
        if not specific_timing_found or primary_vague_term:
            return 0

        # Get the best specific timing score
        specific_scores = [c.score for c in timing_candidates if not c.is_vague]
        if not specific_scores:
            return 0

        final_score = max(specific_scores)

        # Apply penalty for secondary vague terms
        secondary_vague_terms = [
            vague for vague in vague_terms_found
            if not any(c.is_primary and c.is_vague and c.text == vague.text for c in timing_candidates)
        ]

        if secondary_vague_terms:
            penalty_per_term = self.penalties.get('secondary_vague_penalty_per_term', 0.1)
            max_penalty = self.penalties.get('max_secondary_vague_penalty', 0.3)
            penalty = min(max_penalty, len(secondary_vague_terms) * penalty_per_term)
            final_score *= (1 - penalty)

        # Context-aware scoring based on control type
        if config.control_type:
            if config.control_type.lower() == "detective" and final_score > 0:
                # Detective controls need very clear timing
                if not any(c.score >= 0.8 and not c.is_vague for c in timing_candidates):
                    detective_penalty = self.penalties.get('detective_control_penalty', 0.8)
                    final_score *= detective_penalty
            elif config.control_type.lower() == "preventive" and final_score > 0:
                # Preventive controls might have implicit timing
                preventive_boost = self.penalties.get('preventive_control_boost', 1.2)
                final_score = min(1.0, final_score * preventive_boost)

        return max(0, min(1, final_score))

    def _is_part_of_vague_term(self, start: int, end: int, vague_terms_found: List[VagueTermInfo]) -> bool:
        """Check if span is part of a vague term."""
        return any(vague.span[0] <= start and vague.span[1] >= end for vague in vague_terms_found)

    def _is_primary_timing(self, surrounding_text: str) -> bool:
        """Determine if this is the main control timing."""
        return any(
            word in surrounding_text.lower() for word in
            ["review", "verify", "check", "ensure", "validate"]
        )

    def _is_excluded_context(self, context_window: str) -> bool:
        """Check if context should be excluded."""
        excluded_contexts = [
            "real-time", "one time", "first time", "last time",
            "reporting period", "accounting period", "time period"
        ]
        return any(excluded in context_window for excluded in excluded_contexts)

    def _frequency_pattern_matches(self, freq: str, normalized_metadata: str) -> bool:
        """Check if frequency matches using regex patterns."""
        patterns_by_freq = {
            "daily": self.patterns.DAILY,
            "weekly": self.patterns.WEEKLY,
            "monthly": self.patterns.MONTHLY,
            "quarterly": self.patterns.QUARTERLY,
            "annually": self.patterns.ANNUALLY
        }

        if freq in patterns_by_freq:
            return bool(patterns_by_freq[freq].search(normalized_metadata))
        return False

    def _should_check_additional_patterns(self, timing_candidates: List[TimingCandidate], detected_frequencies: List[str]) -> bool:
        """Check if we need to look for additional patterns."""
        return not timing_candidates or (len(timing_candidates) == 1 and "ad-hoc" in detected_frequencies)

    def detect_timing(self, text: str, nlp, config: TimingDetectionConfig) -> TimingDetectionResult:
        """
        Main timing detection method with clean architecture separating detection, validation, and scoring phases.

        DETECTION PHASE: Identifies timing patterns using specialized detection methods
        VALIDATION PHASE: Validates findings against metadata
        SCORING PHASE: Calculates final scores based on detected elements

        Args:
            text: Control description text
            nlp: Loaded spaCy model
            config: TimingDetectionConfig with control_type and frequency_metadata

        Returns:
            TimingDetectionResult with comprehensive analysis results
        """
        if not text or text.strip() == '':
            return self._create_empty_result()

        try:
            # Normalize input text for case-insensitive matching
            normalized_text = text.lower()

            # DETECTION PHASE: Systematic pattern detection organized into logical groups

            # Check for ad-hoc timing first (affects early exit logic)
            has_adhoc = self.patterns.ADHOC.search(normalized_text) is not None

            # Early exit conditions - bypass when ad-hoc timing is present
            early_exit_result = self._check_early_exit_conditions(normalized_text, has_adhoc)
            if early_exit_result:
                return early_exit_result

            # Initialize detection containers
            timing_candidates = []
            detected_frequencies = []
            vague_terms_found = self._detect_vague_terms(normalized_text)
            specific_timing_found = False
            improvement_suggestions = []

            # Ad-hoc timing detection
            if has_adhoc:
                adhoc_candidate = self._detect_adhoc_timing(normalized_text)
                if adhoc_candidate:
                    timing_candidates.append(adhoc_candidate)
                    detected_frequencies.append("ad-hoc")
                    specific_timing_found = True
                    improvement_suggestions.append(
                        "While 'ad-hoc' is an allowed frequency, the control would be stronger if it specified what triggers the ad-hoc review."
                    )

            # Standard frequency pattern detection
            frequency_candidates = self._detect_frequency_patterns(normalized_text, vague_terms_found)
            for candidate in frequency_candidates:
                timing_candidates.append(candidate)
                if candidate.frequency and candidate.frequency not in detected_frequencies:
                    detected_frequencies.append(candidate.frequency)
                specific_timing_found = True

            # Weekday pattern detection (if no strong candidates yet)
            if self._should_check_additional_patterns(timing_candidates, detected_frequencies):
                weekday_candidates = self._detect_weekday_patterns(normalized_text, vague_terms_found)
                for candidate in weekday_candidates:
                    timing_candidates.append(candidate)
                    if "weekly" not in detected_frequencies:
                        detected_frequencies.append("weekly")
                    specific_timing_found = True

            # Period end pattern detection
            if self._should_check_additional_patterns(timing_candidates, detected_frequencies):
                period_candidates = self._detect_period_end_patterns(normalized_text, vague_terms_found)
                for candidate in period_candidates:
                    timing_candidates.append(candidate)
                    if candidate.frequency and candidate.frequency not in detected_frequencies:
                        detected_frequencies.append(candidate.frequency)
                    specific_timing_found = True

            # Timeline pattern detection removed - "within X days" patterns are handled elsewhere

            # Business cycle and event pattern detection
            if self._should_check_additional_patterns(timing_candidates, detected_frequencies):
                business_candidates = self._detect_business_cycle_patterns(normalized_text, vague_terms_found)
                timing_candidates.extend(business_candidates)
                if business_candidates:
                    specific_timing_found = True

            # Implicit timing detection using spaCy (last resort)
            if self._should_check_additional_patterns(timing_candidates, detected_frequencies):
                implicit_candidates = self._detect_implicit_timing(text, nlp, vague_terms_found)
                timing_candidates.extend(implicit_candidates)
                if implicit_candidates:
                    specific_timing_found = True

            # Add vague terms as candidates
            for vague in vague_terms_found:
                # Determine if this is primary or secondary vague term
                if not specific_timing_found:
                    is_primary = True
                else:
                    specific_timing_before = any(
                        not c.is_vague and c.span[0] < vague.span[0] for c in timing_candidates
                    )
                    is_primary = not specific_timing_before

                vague_candidate = TimingCandidate(
                    text=vague.text,
                    method="vague_timing",
                    score=self.penalties.get('vague_term_score', 0.1),
                    span=vague.span,
                    is_primary=is_primary,
                    is_vague=True,
                    context=text[max(0, vague.span[0] - 30):min(len(text), vague.span[1] + 30)]
                )
                timing_candidates.append(vague_candidate)

            # VALIDATION PHASE: Check against metadata and generate suggestions

            # Multi-frequency analysis and improvement suggestions
            is_multi_frequency = len(detected_frequencies) > 1
            if is_multi_frequency:
                improvement_suggestions.append(
                    "Multiple frequencies detected. Consider whether this is describing a process rather than a single control."
                )

            if not specific_timing_found and not has_adhoc:
                improvement_suggestions.append(
                    "No specific timing information detected. Add specific frequency (daily, weekly, monthly) or timing (within X days)."
                )

            for vague_term in vague_terms_found:
                improvement_suggestions.append(
                    f"Replace vague timing term '{vague_term.text}' with {vague_term.suggested_replacement}."
                )

            # Validate against metadata frequency
            validation_result = self._validate_against_metadata(detected_frequencies, config)
            if not validation_result.is_valid and config.frequency_metadata:
                improvement_suggestions.append(
                    f"Align the frequency in the description with the declared frequency ({config.frequency_metadata})"
                )

            # SCORING PHASE: Calculate final score and determine top match

            # Check for primary vague terms
            primary_vague_term = any(c.is_primary and c.is_vague for c in timing_candidates)

            # Calculate final score
            final_score = self._calculate_final_score(
                timing_candidates, vague_terms_found, specific_timing_found, primary_vague_term, config
            )

            # Find the top match for return
            if timing_candidates:
                specific_candidates = [c for c in timing_candidates if not c.is_vague]
                if specific_candidates:
                    top_match = max(specific_candidates, key=lambda x: x.score)
                else:
                    top_match = timing_candidates[0]
            else:
                top_match = None

            # Create final result
            return TimingDetectionResult(
                candidates=timing_candidates,
                top_match=top_match,
                score=final_score,
                extracted_keywords=[c.text for c in timing_candidates],
                multi_frequency_detected=is_multi_frequency,
                frequencies=detected_frequencies,
                validation=validation_result,
                vague_terms=vague_terms_found,
                improvement_suggestions=improvement_suggestions,
                specific_timing_found=specific_timing_found,
                primary_vague_term=primary_vague_term
            )

        except Exception as e:
            print(f"Error in WHEN detection: {str(e)}")
            
            # Provide more specific error feedback based on the type of error
            error_message = f"Error in analysis: {str(e)}"
            improvement_suggestions = []
            
            # Check for common error scenarios and provide targeted feedback
            if "NoneType" in str(e):
                improvement_suggestions.append("Text processing failed. Ensure input text is properly formatted.")
            elif "spacy" in str(e).lower() or "nlp" in str(e).lower():
                improvement_suggestions.append("Language processing error. Check that text contains valid timing information.")
            elif "regex" in str(e).lower() or "pattern" in str(e).lower():
                improvement_suggestions.append("Pattern matching error. Text may contain unusual formatting or characters.")
            else:
                improvement_suggestions.append(f"Analysis failed: {str(e)}. Please verify text format and content.")
            
            return TimingDetectionResult(
                candidates=[],
                top_match=None,
                score=0,
                extracted_keywords=[],
                multi_frequency_detected=False,
                frequencies=[],
                validation=ValidationResult(is_valid=False, message=error_message),
                vague_terms=[],
                improvement_suggestions=improvement_suggestions,
                specific_timing_found=False,
                primary_vague_term=False
            )


# Compatibility wrapper to maintain original function signature
def enhance_when_detection(text: str, nlp, control_type: Optional[str] = None,
                           existing_keywords: Optional[List[str]] = None,  # Kept for compatibility but unused
                           frequency_metadata: Optional[str] = None) -> Dict[str, Any]:
    """
    Compatibility wrapper for the enhanced WHEN detection.

    Args:
        text: Control description text
        nlp: Loaded spaCy model
        control_type: Optional control type for context-aware scoring
        existing_keywords: Unused parameter kept for backward compatibility
        frequency_metadata: Optional declared frequency from metadata for validation

    Returns:
        Dictionary with detection results in the original format
    """
    # Create configuration object
    config = TimingDetectionConfig(
        control_type=control_type,
        frequency_metadata=frequency_metadata
    )

    # Use default configuration for now (in production, this would come from the analyzer)
    timing_config = {
        'timing_detection': {
            'patterns': {},  # Would be populated from YAML
            'scores': {},    # Would be populated from YAML
            'vague_term_suggestions': {}  # Would be populated from YAML
        },
        'penalties': {
            'timing': {}  # Would be populated from YAML
        }
    }

    detector = TimingDetector(timing_config)
    result = detector.detect_timing(text, nlp, config)

    # Convert to original dictionary format for backward compatibility
    return {
        "candidates": [
            {
                "text": c.text,
                "method": c.method,
                "score": c.score,
                "span": c.span,
                "is_primary": c.is_primary,
                "is_vague": c.is_vague,
                "frequency": c.frequency,
                "pattern_type": c.pattern_type,
                "context": c.context
            } for c in result.candidates
        ],
        "top_match": {
            "text": result.top_match.text,
            "method": result.top_match.method,
            "score": result.top_match.score,
            "span": result.top_match.span,
            "is_primary": result.top_match.is_primary,
            "is_vague": result.top_match.is_vague,
            "frequency": result.top_match.frequency,
            "pattern_type": result.top_match.pattern_type,
            "context": result.top_match.context
        } if result.top_match else None,
        "score": result.score,
        "extracted_keywords": result.extracted_keywords,
        "multi_frequency_detected": result.multi_frequency_detected,
        "frequencies": result.frequencies,
        "validation": {
            "is_valid": result.validation.is_valid,
            "message": result.validation.message
        },
        "vague_terms": [
            {
                "text": v.text,
                "span": v.span,
                "suggested_replacement": v.suggested_replacement,
                "is_primary": v.is_primary
            } for v in result.vague_terms
        ],
        "improvement_suggestions": result.improvement_suggestions,
        "specific_timing_found": result.specific_timing_found,
        "primary_vague_term": result.primary_vague_term
    }