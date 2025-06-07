#!/usr/bin/env python3
"""
WHERE Element Detection Module

This module implements WHERE as a standalone element in control descriptions,
identifying where control activities take place. It uses the shared WHERE 
detection service while applying WHERE-specific scoring and analysis logic.

The WHERE element captures:
- Systems where controls are executed
- Physical or virtual locations
- Organizational units or departments
- Geographic regions or jurisdictions
"""

from typing import Dict, List, Any, Optional
import logging
from .where_service import WhereDetectionService

logger = logging.getLogger(__name__)


def enhance_where_detection(text: str, nlp, existing_keywords: List[str] = None,
                          control_type: str = None, config: Dict = None) -> Dict[str, Any]:
    """
    Enhanced WHERE detection as a standalone element.
    
    This function uses the shared WHERE detection service but applies 
    WHERE-specific analysis and scoring logic to evaluate location 
    information as a distinct control element.
    
    Args:
        text: Control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Pre-defined WHERE keywords (optional)
        control_type: Type of control (e.g., 'IT', 'Manual', 'Automated')
        config: Configuration dictionary
        
    Returns:
        Dictionary containing:
        - score: WHERE element score (0-1)
        - components: Detected WHERE components from shared service
        - primary_location: Most relevant location/system
        - location_types: Categories of locations found
        - specificity_score: How specific the location information is
        - suggestions: Improvement recommendations
        - matched_keywords: Keywords that were matched
        - confidence: Overall confidence in WHERE detection
    """
    if not text or not text.strip():
        return _empty_where_result()
        
    # Initialize shared WHERE service
    where_service = WhereDetectionService(config)
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Get base detection results from shared service
    where_components = where_service.detect_where_components(text, doc)
    
    # Apply WHERE-specific scoring logic
    score = calculate_where_score(where_components, control_type, config)
    
    # Generate WHERE-specific insights
    primary_location = determine_primary_location(where_components)
    location_types = categorize_locations(where_components)
    specificity_score = assess_location_specificity(where_components, doc)
    
    # Build WHERE-specific suggestions
    suggestions = generate_where_suggestions(
        where_components, 
        specificity_score,
        location_types,
        control_type
    )
    
    # Extract matched keywords for reporting
    matched_keywords = extract_matched_keywords(where_components)
    
    # Calculate overall confidence
    confidence = calculate_overall_confidence(where_components, specificity_score)
    
    return {
        'score': score,
        'components': where_components,
        'primary_location': primary_location,
        'location_types': location_types,
        'specificity_score': specificity_score,
        'suggestions': suggestions,
        'matched_keywords': matched_keywords,
        'confidence': confidence,
        'element_type': 'WHERE'
    }


def calculate_where_score(components: Dict[str, Any], control_type: str, 
                         config: Dict) -> float:
    """
    Calculate WHERE element score based on detected components.
    
    Scoring considers:
    - Presence of location information
    - Specificity of locations
    - Relevance to control type
    - Number and variety of location types
    """
    if not components or not components.get('all_components'):
        return 0.0
        
    # Get WHERE element configuration
    where_config = config.get('elements', {}).get('WHERE', {})
    min_threshold = where_config.get('min_score_threshold', 0.3)
    
    # Base score from component detection
    base_score = 0.0
    
    # Score based on component types present
    component_type_scores = {
        'systems': 0.4,      # Systems are highly specific
        'locations': 0.3,    # Physical/virtual locations
        'organizational': 0.3 # Organizational units
    }
    
    for comp_type, type_score in component_type_scores.items():
        if components.get(comp_type):
            # Add score based on number and confidence of components
            type_components = components[comp_type]
            avg_confidence = sum(c['confidence'] for c in type_components) / len(type_components)
            base_score += type_score * avg_confidence
            
    # Apply control type relevance if configured
    if control_type and 'control_type_relevance' in where_config:
        relevance = where_config['control_type_relevance'].get(control_type, {})
        
        # Adjust scores based on control type
        for comp_type in ['systems', 'locations', 'organizational']:
            if components.get(comp_type):
                multiplier = relevance.get(comp_type, 1.0)
                type_contribution = component_type_scores[comp_type] * multiplier
                # Recalculate with relevance
                base_score = base_score * multiplier
                
    # Bonus for multiple location types (comprehensive WHERE)
    location_variety = len([t for t in ['systems', 'locations', 'organizational'] 
                           if components.get(t)])
    if location_variety > 1:
        base_score *= 1.1  # 10% bonus for variety
        
    # Penalty for vague locations
    vague_penalty = calculate_vague_location_penalty(components, where_config)
    base_score *= (1 - vague_penalty)
    
    # Ensure score is within bounds
    final_score = max(min_threshold, min(base_score, 1.0))
    
    return final_score


def determine_primary_location(components: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Determine the most relevant WHERE component.
    
    Uses confidence scores and boost factors from the shared service
    to identify the primary location referenced in the control.
    """
    if not components or not components.get('all_components'):
        return None
        
    # The shared service already determines this
    return components.get('primary_component')


def categorize_locations(components: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Categorize detected locations by type for analysis.
    
    Returns a dictionary mapping location types to lists of location texts.
    """
    categories = {
        'systems': [],
        'physical_locations': [],
        'virtual_locations': [],
        'organizational_units': [],
        'geographic_locations': []
    }
    
    # Categorize systems
    for system in components.get('systems', []):
        categories['systems'].append(system['text'])
        
    # Categorize locations
    for location in components.get('locations', []):
        category = location.get('category', 'unknown')
        if category == 'physical':
            categories['physical_locations'].append(location['text'])
        elif category == 'virtual':
            categories['virtual_locations'].append(location['text'])
        elif category == 'geographic':
            categories['geographic_locations'].append(location['text'])
        else:
            # Default to physical if unknown
            categories['physical_locations'].append(location['text'])
            
    # Categorize organizational units
    for org in components.get('organizational', []):
        categories['organizational_units'].append(org['text'])
        
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def assess_location_specificity(components: Dict[str, Any], doc) -> float:
    """
    Assess how specific the location information is.
    
    Higher scores indicate more specific, actionable location information.
    Lower scores indicate vague or generic location references.
    """
    if not components or not components.get('all_components'):
        return 0.0
        
    specificity_score = 0.0
    component_count = len(components['all_components'])
    
    for component in components['all_components']:
        comp_specificity = 0.5  # Base specificity
        
        # Systems are generally specific
        if component['type'] == 'system':
            comp_specificity = 0.8
            # Even more specific if it's a proper noun or acronym
            if component['text'].isupper() or component['text'][0].isupper():
                comp_specificity = 0.9
                
        # Organizational units with departments are specific
        elif component['type'] == 'organizational':
            if component.get('category') == 'departments':
                comp_specificity = 0.75
            else:
                comp_specificity = 0.6
                
        # Locations vary in specificity
        elif component['type'] == 'location':
            category = component.get('category', 'unknown')
            if category == 'named_entity':
                comp_specificity = 0.85  # Named entities are specific
            elif category == 'physical' and component.get('boost_factor', 1.0) > 1.0:
                comp_specificity = 0.7  # Contextual physical locations
            elif category == 'geographic':
                comp_specificity = 0.65
            else:
                comp_specificity = 0.5
                
        # Weight by confidence
        weighted_specificity = comp_specificity * component['confidence']
        specificity_score += weighted_specificity
        
    # Average specificity
    if component_count > 0:
        specificity_score /= component_count
        
    return specificity_score


def generate_where_suggestions(components: Dict[str, Any], specificity_score: float,
                             location_types: Dict[str, List[str]], 
                             control_type: str) -> List[str]:
    """
    Generate specific suggestions for improving WHERE element.
    
    Suggestions are tailored based on what's missing or vague.
    """
    suggestions = []
    
    # Check if any WHERE information exists
    if not components or not components.get('all_components'):
        suggestions.append("Add location information: specify the system, location, or department where this control operates")
        return suggestions
        
    # Check specificity
    if specificity_score < 0.5:
        suggestions.append("Make location references more specific (e.g., 'SAP FI module' instead of 'system')")
        
    # Control type specific suggestions
    if control_type:
        if control_type.lower() in ['it', 'automated', 'system']:
            if 'systems' not in location_types:
                suggestions.append("Specify the system where this automated control operates")
        elif control_type.lower() in ['manual', 'physical']:
            if 'physical_locations' not in location_types and 'organizational_units' not in location_types:
                suggestions.append("Identify the department or physical location where this manual control is performed")
                
    # Check for vague system references
    systems = components.get('systems', [])
    if any(s['text'].lower() in ['system', 'application', 'platform'] for s in systems):
        suggestions.append("Replace generic terms like 'system' with specific system names (e.g., 'SAP', 'Oracle')")
        
    # Check for missing organizational context
    if not location_types.get('organizational_units') and control_type != 'Automated':
        suggestions.append("Consider adding departmental context (e.g., 'Finance department', 'IT team')")
        
    # If only one type of location is present, suggest adding another
    if len(location_types) == 1:
        if 'systems' in location_types:
            suggestions.append("Consider adding organizational context to complement system information")
        elif 'organizational_units' in location_types:
            suggestions.append("Consider specifying the system or location used by this department")
            
    return suggestions


def extract_matched_keywords(components: Dict[str, Any]) -> List[str]:
    """Extract matched keywords for reporting."""
    keywords = []
    
    for comp_list in [components.get('systems', []), 
                     components.get('locations', []),
                     components.get('organizational', [])]:
        for comp in comp_list:
            keywords.append(comp['text'])
            
    return keywords


def calculate_overall_confidence(components: Dict[str, Any], 
                               specificity_score: float) -> float:
    """Calculate overall confidence in WHERE detection."""
    if not components or not components.get('all_components'):
        return 0.0
        
    # Get confidence scores from components
    confidence_scores = components.get('confidence_scores', {})
    overall_confidence = confidence_scores.get('overall', 0.0)
    
    # Adjust based on specificity
    adjusted_confidence = overall_confidence * (0.7 + 0.3 * specificity_score)
    
    return min(adjusted_confidence, 1.0)


def calculate_vague_location_penalty(components: Dict[str, Any], 
                                   where_config: Dict) -> float:
    """Calculate penalty for vague location references."""
    vague_penalty_factor = where_config.get('vague_location_penalty', 0.5)
    
    vague_terms = ['system', 'application', 'platform', 'location', 
                   'place', 'area', 'department', 'team']
    
    total_components = len(components.get('all_components', []))
    if total_components == 0:
        return 0.0
        
    vague_count = 0
    for comp in components.get('all_components', []):
        if comp['text'].lower() in vague_terms and comp['confidence'] < 0.7:
            vague_count += 1
            
    # Calculate penalty as proportion of vague components
    penalty = (vague_count / total_components) * vague_penalty_factor
    
    return min(penalty, vague_penalty_factor)  # Cap at max penalty


def _empty_where_result() -> Dict[str, Any]:
    """Return an empty WHERE result structure."""
    return {
        'score': 0.0,
        'components': {
            'systems': [],
            'locations': [],
            'organizational': [],
            'all_components': [],
            'primary_component': None,
            'confidence_scores': {}
        },
        'primary_location': None,
        'location_types': {},
        'specificity_score': 0.0,
        'suggestions': ["Add location information: specify the system, location, or department where this control operates"],
        'matched_keywords': [],
        'confidence': 0.0,
        'element_type': 'WHERE'
    }