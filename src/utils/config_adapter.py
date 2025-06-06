"""
Configuration Adapter to bridge old and new configuration structures
"""
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml

from src.utils.config_manager import ConfigManager
from src.utils.config_manager_updated import EnhancedConfigManager, ConfigConstants

logger = logging.getLogger(__name__)


class ConfigAdapter:
    """
    Adapter class that provides backward compatibility while transitioning
    to the new enhanced configuration structure.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize with either old or new configuration file.
        
        Args:
            config_file: Path to configuration YAML file
        """
        self.config_file = config_file
        self.is_new_format = False
        self.legacy_manager = None
        self.enhanced_manager = None
        self.config = {}
        
        if config_file:
            self._load_and_detect_format(config_file)
            
    def _load_and_detect_format(self, config_file: str) -> None:
        """Load configuration and detect format (old vs new)."""
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
                
            # Detect format based on key structure
            if 'nlp_config' in raw_config and 'constants' in raw_config:
                # New format
                self.is_new_format = True
                self.enhanced_manager = EnhancedConfigManager(config_file)
                self.config = raw_config
                logger.info("Detected new configuration format")
            else:
                # Old format
                self.is_new_format = False
                self.legacy_manager = ConfigManager(config_file)
                self.config = self.legacy_manager.config if self.legacy_manager else {}
                logger.info("Detected legacy configuration format")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Fall back to legacy format
            try:
                self.legacy_manager = ConfigManager(config_file)
                self.config = self.legacy_manager.config if self.legacy_manager else {}
            except Exception as fallback_error:
                logger.error(f"Error loading fallback configuration: {fallback_error}")
                self.legacy_manager = None
                self.config = {}
    
    # Unified interface methods that work with both formats
    
    def get_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self.config
    
    def get_nlp_config(self) -> Dict[str, str]:
        """Get NLP/SpaCy configuration."""
        if self.is_new_format:
            return self.enhanced_manager.get_nlp_config()
        else:
            # Map old 'spacy' to new 'nlp_config' format
            spacy_config = self.config.get('spacy', {})
            return {
                'preferred_model': spacy_config.get('preferred_model', 'en_core_web_md'),
                'fallback_model': spacy_config.get('fallback_model', 'en_core_web_sm')
            }
    
    def get_element_config(self, element_name: str) -> Dict[str, Any]:
        """Get configuration for a specific element."""
        element_name = element_name.upper()
        
        if self.is_new_format:
            # Build element config from new structure
            weight = self.enhanced_manager.get_element_weight(element_name)
            
            # Get keywords based on element
            keywords = []
            if element_name == "WHO":
                detector = self.enhanced_manager.get_detector('person_role')
                if detector:
                    keywords = detector.get_all_roles()
            elif element_name == "WHEN":
                when_config = self.config.get('when_element', {})
                # Flatten all timing patterns
                for pattern_type, pattern_config in when_config.get('timing_pattern_rules', {}).items():
                    keywords.extend(pattern_config.get('patterns', []))
            elif element_name == "WHAT":
                what_config = self.config.get('what_element', {})
                action_verbs = what_config.get('actionable_verbs', {})
                for category in action_verbs.values():
                    keywords.extend(category.get('verbs', []))
            elif element_name == "WHY":
                why_config = self.config.get('why_element', {})
                keywords.extend(why_config.get('purpose_patterns', []))
            elif element_name == "ESCALATION":
                esc_config = self.config.get('escalation_element', {}).get('escalation_indicators', {})
                keywords.extend(esc_config.get('actions', []))
                keywords.extend(esc_config.get('exception_terms', []))
                
            return {
                'weight': weight,
                'keywords': keywords
            }
        else:
            # Use legacy method
            return self.legacy_manager.get_element_config(element_name) if self.legacy_manager else {}
    
    def get_element_weight(self, element: str) -> int:
        """Get weight for a specific element."""
        if self.is_new_format:
            return self.enhanced_manager.get_element_weight(element)
        else:
            element_config = self.config.get('elements', {}).get(element.upper(), {})
            return element_config.get('weight', 10)
    
    def get_vague_terms(self) -> List[str]:
        """Get list of vague terms."""
        if self.is_new_format:
            # Extract vague terms from when_element configuration
            when_config = self.config.get('when_element', {})
            vague_terms_config = when_config.get('vague_terms', {})
            return list(vague_terms_config.keys())
        else:
            return self.legacy_manager.get_vague_terms() if self.legacy_manager else []
    
    def get_column_mapping(self) -> Dict[str, Any]:
        """Get column mapping configuration."""
        if self.is_new_format:
            return self.enhanced_manager.get_column_mapping()
        else:
            # Try multiple possible locations in old format
            return (self.config.get('column_mapping', {}) or 
                    self.config.get('columns', {}) or 
                    {})
    
    def get_column_defaults(self) -> Dict[str, str]:
        """Get column defaults (legacy method name for compatibility)."""
        return self.get_column_mapping()
    
    def get_category_thresholds(self) -> Dict[str, int]:
        """Get scoring category thresholds."""
        if self.is_new_format:
            scoring = self.config.get('scoring', {})
            return scoring.get('category_thresholds', {})
        else:
            return self.config.get('category_thresholds', {})
    
    def get_penalties(self) -> Dict[str, Any]:
        """Get penalty configuration."""
        if self.is_new_format:
            return self.config.get('penalties', {})
        else:
            # Map old penalty structure to new format
            penalties = {
                'vague_terms': {
                    'base_penalty': self.config.get('vague_term_penalty', 2) / 10.0,  # Convert to 0-1 scale
                    'max_penalty': self.config.get('max_vague_penalty', 10) / 10.0
                },
                'multi_control': self.config.get('penalties', {}).get('multi_control', {})
            }
            return penalties
    
    def get_confidence_threshold(self, threshold_type: str) -> float:
        """Get confidence threshold value."""
        if self.is_new_format:
            constants = ConfigConstants()
            threshold_map = {
                'explicit': constants.CONFIDENCE_EXPLICIT,
                'high': constants.CONFIDENCE_HIGH,
                'medium_high': constants.CONFIDENCE_MEDIUM_HIGH,
                'medium': constants.CONFIDENCE_MEDIUM,
                'medium_low': constants.CONFIDENCE_MEDIUM_LOW,
                'low': constants.CONFIDENCE_LOW,
                'very_low': constants.CONFIDENCE_VERY_LOW
            }
            return threshold_map.get(threshold_type, 0.5)
        else:
            # Try to find in WHO detection scores or use defaults
            who_scores = self.config.get('who_detection', {}).get('scores', {})
            score_map = {
                'explicit': who_scores.get('explicit_frequency', 0.9),
                'high': who_scores.get('period_end_pattern', 0.85),
                'medium': who_scores.get('business_cycle_pattern', 0.75),
                'low': who_scores.get('implicit_temporal_modifier', 0.6),
                'very_low': 0.4
            }
            return score_map.get(threshold_type, 0.5)
    
    # Enhanced methods for new functionality
    
    def get_detector(self, detector_type: str):
        """Get a detector instance (only available in new format)."""
        if self.is_new_format:
            return self.enhanced_manager.get_detector(detector_type)
        else:
            logger.warning(f"Detector '{detector_type}' not available in legacy configuration")
            return None
    
    def get_penalties_manager(self):
        """Get penalties manager (only available in new format)."""
        if self.is_new_format:
            return self.enhanced_manager.get_penalties_manager()
        else:
            logger.warning("Penalties manager not available in legacy configuration")
            return None
    
    # Legacy compatibility methods
    
    def get_audit_leader_column(self) -> Optional[str]:
        """Get audit leader column name."""
        if self.is_new_format:
            column_mapping = self.get_column_mapping()
            owner_columns = column_mapping.get('owner_columns', [])
            return owner_columns[0] if owner_columns else None
        else:
            return self.legacy_manager.get_audit_leader_column() if self.legacy_manager else None
    
    def get_audit_entity_column(self) -> Optional[str]:
        """Get audit entity column name."""
        if self.is_new_format:
            column_mapping = self.get_column_mapping()
            process_columns = column_mapping.get('process_columns', [])
            return process_columns[0] if process_columns else None
        else:
            return self.legacy_manager.get_audit_entity_column() if self.legacy_manager else None
    
    def is_enhanced_format(self) -> bool:
        """Check if using enhanced configuration format."""
        return self.is_new_format