#!/usr/bin/env python3
"""
Control Type Classification Module

This module implements the control type classification system as specified in 
ScoringUpdate.md. It determines whether a control should be classified as:
- 'system': Controls where systems actively participate in control execution
- 'location_dependent': Controls that require specific physical locations
- 'other': Controls that don't require WHERE specification

The classification drives conditional WHERE scoring.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import os
import yaml
import spacy
from spacy.tokens import Doc


class ControlTypeClassifier:
    """
    Classifies controls based on automation field and control content analysis.
    
    Uses the logic from ScoringUpdate.md:
    1. Check Control_Automation field
    2. For Manual controls, analyze for system participation
    3. For Hybrid controls, determine system vs location prominence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the control type classifier.
        
        Args:
            config: Configuration dictionary with classification settings
        """
        self.config = config or {}
        self._initialize_classification_patterns()
        
    def load_external_systems(self) -> List[str]:
        """Load systems from external YAML file"""
        systems_file = self.config.get('shared_where_config', {}).get('external_systems_file')
        if systems_file and os.path.exists(systems_file):
            try:
                with open(systems_file, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get('systems', [])
            except Exception as e:
                print(f"Warning: Could not load external systems file {systems_file}: {e}")
                return []
        return []
        
    def _initialize_classification_patterns(self):
        """Initialize patterns for control classification."""
        classification_config = self.config.get('classification', {})
        
        # Control-participating verbs (upgrade manual to hybrid)
        self.control_participating_verbs = set(classification_config.get(
            'control_participating_verbs', [
                'calculates', 'validates', 'approves', 'alerts', 'flags',
                'reconciles', 'generates', 'processes', 'identifies', 'matches',
                'automatically'
            ]
        ))
        
        # Documentation verbs (remain manual)
        self.documentation_verbs = set(classification_config.get(
            'documentation_verbs', [
                'saves', 'stores', 'documents', 'records', 'enters',
                'uploads', 'maintains', 'tracks', 'files'
            ]
        ))
        
        # System context indicators
        self.system_context_patterns = [
            r'\bsystem\s+(validates|calculates|processes|generates|identifies|flags|alerts)\b',
            r'\bautomatically\s+\w+',
            r'\b(generates|calculates|validates|processes)\b',
            r'\binterface\s+(validates|reconciles|matches)\b',
            r'\benters?\s+data\s+into\b',
            r'\brecords?\s+in\s+\w+\s+system\b',
            r'\buploads?\s+to\b'
        ]
        
        # Location context indicators  
        self.location_context_patterns = [
            r'\bat\s+(branch|office|facility|headquarters|site)\b',
            r'\bon-site\b',
            r'\bin\s+person\b',
            r'\bphysical\s+(location|facility|premises|vault|building)\b',
            r'\bphysically\s+(inspects?|reviews?|examines?|checks?)\b',
            r'\b(teller|guard|manager)\s+performs\b',
            r'\b(vault|facility|premises)\b',
            r'\bbranch\s+(location|office|facility)\b',
            r'\boffice\s+(location|facility|building)\b'
        ]
        
        # System names from centralized registry
        system_registry = self.config.get('system_registry', [])
        # Fallback to old config structure if registry doesn't exist
        fallback_systems = classification_config.get('system_names', [
            'sap', 'oracle', 'peoplesoft', 'jde', 'dynamics', 'netsuite',
            'sharepoint', 'teams', 'slack', 'confluence', 'servicenow',
            'tableau', 'power bi', 'excel', 'access', 'application', 'system'
        ])
        
        # Load external systems first
        external_systems = self.load_external_systems()
        processed_systems = set()
        for system in external_systems:
            if system and isinstance(system, str):
                processed_systems.add(system.lower())
        
        # Process system registry to handle different formats and filter None values (fallback)
        if system_registry and not processed_systems:  # Only use registry if no external systems
            if isinstance(system_registry, list):
                # Flat list format
                for system in system_registry:
                    if system and isinstance(system, str):
                        processed_systems.add(system.lower())
            elif isinstance(system_registry, dict):
                # Structured format with categories
                for category, systems in system_registry.items():
                    if isinstance(systems, list):
                        for system in systems:
                            if system and isinstance(system, str):
                                processed_systems.add(system.lower())
        
        self.system_names = processed_systems if processed_systems else set(fallback_systems)
        
        # Weighting factors
        self.system_context_weight = classification_config.get('system_context_weight', 2)
        self.location_context_weight = classification_config.get('location_context_weight', 1)
        
    def classify_control(self, control_description: str, 
                        automation_field: Optional[str] = None,
                        nlp_doc: Optional[Doc] = None) -> Dict[str, Any]:
        """
        Classify a control based on automation field and content analysis.
        
        Args:
            control_description: The control description text
            automation_field: Value from Control_Automation field
            nlp_doc: Optional spaCy Doc object for the text
            
        Returns:
            Dictionary with classification results:
            {
                'final_type': 'system'|'location_dependent'|'other',
                'automation_field': original automation value,
                'upgraded': bool indicating if manual was upgraded,
                'system_score': system context score,
                'location_score': location context score,
                'reasoning': explanation of classification
            }
        """
        if not control_description or not control_description.strip():
            return self._empty_classification()
            
        # Normalize automation field
        automation = self._normalize_automation_field(automation_field)
        
        # Initialize scores
        system_score = 0
        location_score = 0
        upgraded = False
        reasoning = []
        
        # Step 1: Primary classification based on automation field
        if automation == 'automated':
            final_type = 'system'
            # Calculate scores for confidence calculation
            system_score = self._calculate_system_context_score(control_description)
            location_score = self._calculate_location_context_score(control_description)
            reasoning.append("Classified as 'system' due to 'Automated' automation field")
            
        elif automation == 'hybrid':
            # Analyze which aspect is more prominent
            system_score = self._calculate_system_context_score(control_description)
            location_score = self._calculate_location_context_score(control_description)
            
            if system_score > location_score:
                final_type = 'system'
                reasoning.append(f"Hybrid control with stronger system context ({system_score} vs {location_score})")
            elif location_score > 0:
                final_type = 'location_dependent'
                reasoning.append(f"Hybrid control with stronger location context ({location_score} vs {system_score})")
            else:
                final_type = 'other'
                reasoning.append("Hybrid control with no clear system or location context")
                
        elif automation == 'manual':
            # Check for system interaction that would upgrade to hybrid
            if self._has_control_participating_system_interaction(control_description):
                upgraded = True
                automation = 'hybrid'  # Upgrade
                
                # Re-analyze as hybrid
                system_score = self._calculate_system_context_score(control_description)
                location_score = self._calculate_location_context_score(control_description)
                
                if system_score > location_score:
                    final_type = 'system'
                    reasoning.append("Manual control upgraded to 'system' due to control-participating system interaction")
                elif location_score > 0:
                    final_type = 'location_dependent' 
                    reasoning.append("Manual control upgraded to 'location_dependent' with some system interaction")
                else:
                    final_type = 'other'
                    reasoning.append("Manual control upgraded but no clear system/location prominence")
            else:
                # Check if it's location-dependent even without system interaction
                location_score = self._calculate_location_context_score(control_description)
                if location_score > 0:
                    final_type = 'location_dependent'
                    reasoning.append("Manual control classified as 'location_dependent' due to physical location requirements")
                else:
                    final_type = 'other'
                    reasoning.append("Manual control classified as 'other' - no system interaction or location dependency")
        else:
            # Unknown or missing automation field
            system_score = self._calculate_system_context_score(control_description)
            location_score = self._calculate_location_context_score(control_description)
            
            if system_score > location_score and system_score > 0:
                final_type = 'system'
                reasoning.append("Classified as 'system' based on content analysis (no automation field)")
            elif location_score > 0:
                final_type = 'location_dependent'
                reasoning.append("Classified as 'location_dependent' based on content analysis")
            else:
                final_type = 'other'
                reasoning.append("Classified as 'other' - no clear automation field or context")
        
        return {
            'final_type': final_type,
            'automation_field': automation_field,
            'normalized_automation': automation,
            'upgraded': upgraded,
            'system_score': system_score,
            'location_score': location_score,
            'reasoning': reasoning,
            'classification_confidence': self._calculate_confidence(
                final_type, system_score, location_score, automation_field
            )
        }
        
    def _normalize_automation_field(self, automation_field: Optional[str]) -> str:
        """Normalize automation field values."""
        if not automation_field:
            return 'unknown'
            
        automation_lower = automation_field.lower().strip()
        
        # Map variations to standard values
        if automation_lower in ['automated', 'automatic', 'system', 'auto']:
            return 'automated'
        elif automation_lower in ['hybrid', 'semi-automated', 'mixed']:
            return 'hybrid'
        elif automation_lower in ['manual', 'human', 'person']:
            return 'manual'
        else:
            return 'unknown'
            
    def _has_control_participating_system_interaction(self, text: str) -> bool:
        """
        Check if manual control has system interaction that participates in control execution.
        
        Only upgrades if system performs control logic, not just documentation.
        """
        text_lower = text.lower()
        
        # Check for control-participating verbs with system context
        for verb in self.control_participating_verbs:
            if verb in text_lower:
                # Check if it's in a system context (stricter patterns)
                system_verb_patterns = [
                    f'system {verb}',
                    f'{verb} by system', 
                    f'automatically {verb}',
                    f'system-{verb}',
                    f'{verb} automatically'
                ]
                if any(pattern in text_lower for pattern in system_verb_patterns):
                    return True
                    
        # Check for system interaction patterns  
        for pattern in self.system_context_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Special case: "automated [action]" should upgrade
        if re.search(r'\bautomated\s+\w+', text_lower):
            return True
                
        # Check if control has system names + control verbs (not just documentation)
        for system_name in self.system_names:
            if system_name in text_lower:
                # Look for control verbs near system name
                words = text_lower.split()
                try:
                    system_idx = words.index(system_name)
                    # Check 5 words before and after system name for better context
                    context_words = words[max(0, system_idx-5):system_idx+6]
                    context_text = ' '.join(context_words)
                    
                    # Check for control-participating verbs, not documentation verbs
                    has_control_verb = any(verb in context_text for verb in self.control_participating_verbs)
                    has_doc_verb = any(verb in context_text for verb in self.documentation_verbs)
                    
                    # Explicit logic for documentation vs control activities
                    if has_doc_verb and not has_control_verb:
                        # Pure documentation activity - do not upgrade
                        continue  # Skip this system - documentation only
                    elif has_control_verb:
                        # Has control-participating verb - upgrade to hybrid
                        return True
                    # If neither doc nor control verb found near system, continue checking
                except ValueError:
                    continue
                    
        return False
        
    def _calculate_system_context_score(self, text: str) -> int:
        """Calculate system context score for hybrid control prominence."""
        score = 0
        text_lower = text.lower()
        
        # System action verbs (weight 2 each)
        system_verbs = ['generates', 'calculates', 'validates', 'processes', 'identifies', 'flags']
        for verb in system_verbs:
            if verb in text_lower:
                score += self.system_context_weight
                
        # Automation phrases (weight 2 each)
        automation_phrases = ['automatically', 'automated', 'system-generated', 'system performs']
        for phrase in automation_phrases:
            if phrase in text_lower:
                score += self.system_context_weight
                
        # System names (weight 2 each) - but only if associated with control verbs, not documentation
        for system_name in self.system_names:
            if system_name in text_lower:
                # Check context around system name for verb type
                words = text_lower.split()
                try:
                    system_idx = words.index(system_name)
                    # Check 5 words before and after system name
                    context_words = words[max(0, system_idx-5):system_idx+6]
                    context_text = ' '.join(context_words)
                    
                    has_control_verb = any(verb in context_text for verb in self.control_participating_verbs)
                    has_doc_verb = any(verb in context_text for verb in self.documentation_verbs)
                    
                    # Only award points if it's control-participating, not pure documentation
                    if has_control_verb or (not has_doc_verb):
                        score += self.system_context_weight
                    # If pure documentation (has_doc_verb and not has_control_verb), don't add points
                except ValueError:
                    # If we can't find the system name in words (edge case), give benefit of doubt
                    score += self.system_context_weight
                
        # Data interaction patterns (weight 1 each)
        data_patterns = ['enters data into', 'records in', 'uploads to']
        for pattern in data_patterns:
            if pattern in text_lower:
                score += 1
                
        return score
        
    def _calculate_location_context_score(self, text: str) -> int:
        """Calculate location context score for hybrid control prominence."""
        score = 0
        text_lower = text.lower()
        
        # Physical location references (weight 1 each)
        for pattern in self.location_context_patterns:
            if re.search(pattern, text_lower):
                score += self.location_context_weight
                
        return score
        
    def _calculate_confidence(self, final_type: str, system_score: int, 
                            location_score: int, automation_field: Optional[str]) -> float:
        """Calculate confidence in the classification."""
        # Base confidence from automation field
        if automation_field and automation_field.lower() in ['automated', 'manual', 'hybrid']:
            base_confidence = 0.8
        else:
            base_confidence = 0.5
            
        # Adjust based on score strength
        if final_type == 'system':
            if system_score >= 4:
                return min(base_confidence + 0.2, 1.0)
            elif system_score >= 2:
                return base_confidence
            else:
                return max(base_confidence - 0.2, 0.3)
        elif final_type == 'location_dependent':
            if location_score >= 2:
                return min(base_confidence + 0.1, 1.0)
            else:
                return base_confidence
        else:  # 'other'
            if system_score == 0 and location_score == 0:
                return base_confidence
            else:
                return max(base_confidence - 0.1, 0.4)
                
    def _empty_classification(self) -> Dict[str, Any]:
        """Return empty classification for invalid input."""
        return {
            'final_type': 'other',
            'automation_field': None,
            'normalized_automation': 'unknown',
            'upgraded': False,
            'system_score': 0,
            'location_score': 0,
            'reasoning': ['Empty or invalid control description'],
            'classification_confidence': 0.0
        }


def classify_control_type(control_description: str, 
                         automation_field: Optional[str] = None,
                         config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to classify a single control.
    
    Args:
        control_description: The control description text
        automation_field: Value from Control_Automation field
        config: Optional configuration dictionary
        
    Returns:
        Classification results dictionary
    """
    classifier = ControlTypeClassifier(config)
    return classifier.classify_control(control_description, automation_field)