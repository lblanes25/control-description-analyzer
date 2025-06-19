"""
Enhanced Configuration Manager with separated data and behavioral logic
"""
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# Constants extracted from configuration
class ConfigConstants:
    """Global configuration constants"""
    # Confidence thresholds
    CONFIDENCE_EXPLICIT = 0.9
    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_MEDIUM_HIGH = 0.8
    CONFIDENCE_MEDIUM = 0.75
    CONFIDENCE_MEDIUM_LOW = 0.7
    CONFIDENCE_LOW = 0.6
    CONFIDENCE_VERY_LOW = 0.4
    
    # Element weights
    WEIGHT_WHO = 32
    WEIGHT_WHAT = 25
    WEIGHT_WHEN = 20
    WEIGHT_WHY = 13
    WEIGHT_ESCALATION = 10
    
    # Category thresholds
    CATEGORY_MEETS_EXPECTATIONS = 75
    CATEGORY_REQUIRES_ATTENTION = 50
    
    # Common document types
    DOCUMENT_TYPES = ["procedure", "policy", "document", "standard", 
                      "guideline", "regulation", "instruction", "manual"]


@dataclass
class ElementConfig:
    """Base configuration for control elements"""
    weight: int
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract base class for element detectors"""
    
    @abstractmethod
    def calculate_confidence(self, text: str, matches: List[str]) -> float:
        """Calculate confidence score based on matches"""
        pass


class PersonRoleDetector(BaseDetector):
    """Detector for human role identification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.executive_roles = config.get('executive', [])
        self.management_roles = config.get('management', [])
        self.staff_roles = config.get('staff', [])
        self.finance_roles = config.get('finance', [])
        self.audit_compliance_roles = config.get('audit_compliance', [])
        self.technology_roles = config.get('technology', [])
        self.operations_roles = config.get('operations', [])
        
    def calculate_confidence(self, text: str, matches: List[str]) -> float:
        """Calculate confidence based on role hierarchy"""
        confidence = ConfigConstants.CONFIDENCE_LOW
        
        for match in matches:
            if match in self.executive_roles:
                confidence = max(confidence, ConfigConstants.CONFIDENCE_EXPLICIT)
            elif match in self.management_roles:
                confidence = max(confidence, ConfigConstants.CONFIDENCE_HIGH)
            elif match in self.finance_roles or match in self.audit_compliance_roles:
                confidence = max(confidence, ConfigConstants.CONFIDENCE_MEDIUM_HIGH)
            else:
                confidence = max(confidence, ConfigConstants.CONFIDENCE_MEDIUM)
                
        return confidence
    
    def get_all_roles(self) -> List[str]:
        """Get all configured roles"""
        all_roles = []
        all_roles.extend(self.executive_roles)
        all_roles.extend(self.management_roles)
        all_roles.extend(self.staff_roles)
        all_roles.extend(self.finance_roles)
        all_roles.extend(self.audit_compliance_roles)
        all_roles.extend(self.technology_roles)
        all_roles.extend(self.operations_roles)
        return all_roles


class SystemDetector(BaseDetector):
    """Detector for automated system identification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.automated_keywords = config.get('automated_keywords', [])
        self.system_verbs = config.get('system_verbs', [])
        self.common_systems = config.get('common_systems', [])
        
    def calculate_confidence(self, text: str, matches: List[str]) -> float:
        """Calculate confidence for system detection"""
        if any(system in text.lower() for system in self.common_systems):
            return ConfigConstants.CONFIDENCE_EXPLICIT
        
        verb_count = sum(1 for verb in self.system_verbs if verb in text.lower())
        keyword_count = sum(1 for kw in self.automated_keywords if kw in text.lower())
        
        if verb_count >= 2 or keyword_count >= 2:
            return ConfigConstants.CONFIDENCE_HIGH
        elif verb_count >= 1 or keyword_count >= 1:
            return ConfigConstants.CONFIDENCE_MEDIUM
        
        return ConfigConstants.CONFIDENCE_LOW


class TimingPatternMatcher(BaseDetector):
    """Matcher for complex timing pattern detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.timing_rules = config
        self.vague_terms = {}
        
    def calculate_confidence(self, text: str, pattern_type: str) -> float:
        """Calculate confidence for timing patterns"""
        if pattern_type in self.timing_rules:
            return self.timing_rules[pattern_type].get('confidence', ConfigConstants.CONFIDENCE_MEDIUM)
        return ConfigConstants.CONFIDENCE_LOW
    
    def validate_vague_timing(self, term: str) -> Tuple[bool, str, float]:
        """Validate vague timing terms"""
        if term in self.vague_terms:
            vague_config = self.vague_terms[term]
            return True, vague_config['suggestion'], vague_config['penalty']
        return False, "", 0.0


class ActionAnalyzer(BaseDetector):
    """Analyzer for verb strength categorization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.strong_verbs = config.get('strong_action', {}).get('verbs', [])
        self.moderate_verbs = config.get('moderate_action', {}).get('verbs', [])
        self.weak_verbs = config.get('weak_action', {}).get('verbs', [])
        self.control_nouns = config.get('control_nouns', [])
        self.low_confidence_verbs = config.get('low_confidence_verbs', [])
        self.confidence_threshold = config.get('confidence_threshold', ConfigConstants.CONFIDENCE_VERY_LOW)
        self.action_multipliers = config.get('action_multipliers', {})
        
    def calculate_confidence(self, text: str, verbs_found: List[str]) -> float:
        """Calculate confidence based on verb strength"""
        strong_count = sum(1 for v in verbs_found if v in self.strong_verbs)
        moderate_count = sum(1 for v in verbs_found if v in self.moderate_verbs)
        weak_count = sum(1 for v in verbs_found if v in self.weak_verbs)
        
        if strong_count >= 2:
            return ConfigConstants.CONFIDENCE_EXPLICIT * self.action_multipliers.get('multiple_strong_actions', 1.2)
        elif strong_count == 1:
            return ConfigConstants.CONFIDENCE_HIGH * self.action_multipliers.get('single_strong_action', 1.0)
        elif moderate_count > 0:
            return ConfigConstants.CONFIDENCE_MEDIUM * self.action_multipliers.get('moderate_action_only', 0.8)
        elif weak_count > 0:
            return ConfigConstants.CONFIDENCE_LOW * self.action_multipliers.get('weak_action_only', 0.6)
        
        return self.confidence_threshold * self.action_multipliers.get('no_clear_action', 0.4)


class PurposeAnalyzer(BaseDetector):
    """Analyzer for intent and purpose detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.intent_categories = config.get('control_intent_categories', {})
        self.purpose_patterns = config.get('purpose_patterns', [])
        self.risk_keywords = config.get('risk_keywords', [])
        
    def calculate_confidence(self, text: str, matches: List[str]) -> float:
        """Calculate confidence for purpose detection"""
        max_confidence = ConfigConstants.CONFIDENCE_LOW
        
        for category, cat_config in self.intent_categories.items():
            keywords = cat_config.get('keywords', [])
            category_confidence = cat_config.get('confidence', ConfigConstants.CONFIDENCE_MEDIUM)
            
            if any(kw in text.lower() for kw in keywords):
                max_confidence = max(max_confidence, category_confidence)
                
        return max_confidence
    
    def calculate_risk_alignment_score(self, text: str) -> float:
        """Calculate risk alignment score"""
        risk_count = sum(1 for risk in self.risk_keywords if risk in text.lower())
        if risk_count >= 3:
            return ConfigConstants.CONFIDENCE_EXPLICIT
        elif risk_count >= 2:
            return ConfigConstants.CONFIDENCE_HIGH
        elif risk_count >= 1:
            return ConfigConstants.CONFIDENCE_MEDIUM
        return ConfigConstants.CONFIDENCE_LOW


class EscalationPathDetector(BaseDetector):
    """Detector for escalation paths and exception handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.escalation_roles = config.get('roles', [])
        self.escalation_actions = config.get('actions', [])
        self.exception_terms = config.get('exception_terms', [])
        self.threshold_terms = config.get('threshold_terms', [])
        self.process_terms = config.get('process_terms', [])
        
    def calculate_confidence(self, text: str, matches: Dict[str, List[str]]) -> float:
        """Calculate confidence for escalation detection"""
        role_matches = len(matches.get('roles', []))
        action_matches = len(matches.get('actions', []))
        exception_matches = len(matches.get('exceptions', []))
        
        if role_matches > 0 and action_matches > 0:
            if exception_matches > 0:
                return ConfigConstants.CONFIDENCE_EXPLICIT
            return ConfigConstants.CONFIDENCE_HIGH
        elif action_matches > 0 or exception_matches > 0:
            return ConfigConstants.CONFIDENCE_MEDIUM
            
        return ConfigConstants.CONFIDENCE_LOW


class ScoringPenalties:
    """Class for managing scoring penalties"""
    
    def __init__(self, config: Dict[str, Any]):
        self.vague_term_penalties = config.get('vague_terms', {})
        self.multi_control_penalties = config.get('multi_control', {})
        self.missing_element_penalties = config.get('missing_elements', {})
        self.quality_issue_penalties = config.get('quality_issues', {})
        
    def calculate_vague_term_penalty(self, term_type: str) -> float:
        """Calculate penalty for vague terms"""
        if term_type == 'severe':
            return self.vague_term_penalties.get('severe_penalty', 0.4)
        elif term_type == 'moderate':
            return self.vague_term_penalties.get('moderate_penalty', 0.25)
        elif term_type == 'light':
            return self.vague_term_penalties.get('light_penalty', 0.15)
        return self.vague_term_penalties.get('base_penalty', 0.3)
    
    def calculate_multi_control_penalty(self, control_count: int) -> float:
        """Calculate penalty for multiple controls"""
        per_control = self.multi_control_penalties.get('per_control_penalty', 0.2)
        max_penalty = self.multi_control_penalties.get('max_penalty', 0.6)
        return min(control_count * per_control, max_penalty)
    
    def calculate_missing_element_penalty(self, element: str) -> float:
        """Calculate penalty for missing elements"""
        if element in ['WHO', 'WHAT']:
            return self.missing_element_penalties.get('critical_element', 0.4)
        elif element in ['WHEN', 'WHY']:
            return self.missing_element_penalties.get('important_element', 0.3)
        else:
            return self.missing_element_penalties.get('optional_element', 0.2)


class EnhancedConfigManager:
    """Enhanced configuration manager with behavioral logic separation"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = {}
        self.detectors = {}
        self.penalties = None
        
        if config_file:
            self.load_config(config_file)
            self._initialize_detectors()
            
    def load_config(self, path: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(path, "r") as f:
                self.config = yaml.safe_load(f)
                print(f"Successfully loaded configuration from {path}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise
            
    def _initialize_detectors(self) -> None:
        """Initialize detector classes with configuration"""
        who_config = self.config.get('who_element', {})
        self.detectors['person_role'] = PersonRoleDetector(who_config.get('person_roles', {}))
        self.detectors['system'] = SystemDetector(who_config.get('system_patterns', {}))
        
        when_config = self.config.get('when_element', {})
        timing_matcher = TimingPatternMatcher(when_config.get('timing_pattern_rules', {}))
        timing_matcher.vague_terms = when_config.get('vague_terms', {})
        self.detectors['timing'] = timing_matcher
        
        what_config = self.config.get('what_element', {})
        self.detectors['action'] = ActionAnalyzer(what_config)
        
        why_config = self.config.get('why_element', {})
        self.detectors['purpose'] = PurposeAnalyzer(why_config)
        
        escalation_config = self.config.get('escalation_element', {})
        self.detectors['escalation'] = EscalationPathDetector(escalation_config.get('escalation_indicators', {}))
        
        penalty_config = self.config.get('penalties', {})
        self.penalties = ScoringPenalties(penalty_config)
        
    def get_detector(self, detector_type: str) -> Optional[BaseDetector]:
        """Get a specific detector instance"""
        return self.detectors.get(detector_type)
    
    def get_element_weight(self, element: str) -> int:
        """Get weight for a specific element"""
        weights = self.config.get('scoring', {}).get('element_weights', {})
        return weights.get(element.upper(), 0)
    
    def get_nlp_config(self) -> Dict[str, str]:
        """Get NLP configuration"""
        return self.config.get('nlp_config', {})
    
    def get_column_mapping(self) -> Dict[str, List[str]]:
        """Get column mapping configuration"""
        return self.config.get('column_mapping', {})
    
    def get_constants(self) -> ConfigConstants:
        """Get configuration constants"""
        return ConfigConstants()
    
    def get_penalties_manager(self) -> ScoringPenalties:
        """Get penalties manager"""
        return self.penalties