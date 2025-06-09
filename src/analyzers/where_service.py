#!/usr/bin/env python3
"""
Shared WHERE Detection Service

This module provides a centralized service for detecting WHERE (location) components
in control descriptions. It's used by both the WHAT analyzer (for maintaining context)
and the WHERE analyzer (for dedicated location analysis).

The service identifies:
- Systems (SAP, Oracle, SharePoint, etc.)
- Physical locations (offices, facilities, regions)
- Organizational units (departments, divisions)
- Geographic locations (countries, regions)
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import spacy
from spacy.tokens import Doc, Token, Span


class WhereDetectionService:
    """
    Shared service for WHERE detection used by multiple analyzers.
    
    This service provides consistent location detection across the framework,
    allowing both WHAT and WHERE analyzers to identify and utilize location
    information without duplication of logic.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the WHERE detection service.
        
        Args:
            config: Configuration dictionary containing WHERE patterns and settings
        """
        self.config = config or {}
        self._cache = {}
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize detection patterns from configuration."""
        # Get shared WHERE configuration
        where_config = self.config.get('shared_where_config', {})
        patterns = where_config.get('detection_patterns', {})
        
        # System patterns
        self.system_patterns = patterns.get('systems', {})
        self._compile_system_patterns()
        
        # Location patterns  
        self.location_patterns = patterns.get('locations', {})
        self._compile_location_patterns()
        
        # Organizational patterns
        self.organizational_patterns = patterns.get('organizational', {})
        self._compile_organizational_patterns()
        
        # Confidence factors
        self.confidence_factors = where_config.get('confidence_factors', {
            'explicit_preposition': 1.2,
            'implicit_reference': 0.8,
            'contextual_mention': 0.6,
            'proper_noun': 1.1,
            'acronym': 0.9
        })
        
    def _compile_system_patterns(self):
        """Compile system detection patterns."""
        self.system_keywords = set()
        self.system_categories = {}
        
        # Default system patterns if not in config
        default_systems = {
            'erp_systems': {
                'keywords': ['sap', 'oracle', 'peoplesoft', 'jde', 'dynamics', 
                           'netsuite', 'quickbooks', 'sage'],
                'boost_factor': 1.2
            },
            'collaboration': {
                'keywords': ['sharepoint', 'teams', 'slack', 'confluence', 
                           'jira', 'servicenow', 'salesforce'],
                'boost_factor': 1.1
            },
            'financial_systems': {
                'keywords': ['hyperion', 'essbase', 'cognos', 'tableau', 
                           'power bi', 'qlik', 'alteryx'],
                'boost_factor': 1.15
            },
            'custom_systems': {
                'keywords': ['system', 'application', 'platform', 'software',
                           'database', 'repository', 'portal'],
                'boost_factor': 0.9
            }
        }
        
        # Add systems from centralized registry if available
        system_registry = self.config.get('system_registry', [])
        if system_registry:
            # Use centralized registry with default boost factor
            for system in system_registry:
                self.system_keywords.add(system.lower())
                self.system_categories[system.lower()] = {
                    'category': 'registry_systems',
                    'boost_factor': 1.1  # Default boost factor for registry systems
                }
        
        # Merge with config patterns (this allows for custom boost factors)
        for category, details in {**default_systems, **self.system_patterns}.items():
            keywords = details.get('keywords', [])
            self.system_keywords.update(keywords)
            for keyword in keywords:
                self.system_categories[keyword.lower()] = {
                    'category': category,
                    'boost_factor': details.get('boost_factor', 1.0)
                }
                
    def _compile_location_patterns(self):
        """Compile physical and geographic location patterns."""
        self.location_keywords = set()
        self.location_categories = {}
        
        default_locations = {
            'physical': {
                'keywords': ['office', 'facility', 'branch', 'headquarters', 'hq',
                           'building', 'floor', 'room', 'desk', 'workstation'],
                'context_required': True,
                'boost_factor': 1.0
            },
            'geographic': {
                'keywords': ['region', 'country', 'state', 'city', 'global',
                           'domestic', 'international', 'local', 'remote'],
                'boost_factor': 0.95
            },
            'virtual': {
                'keywords': ['cloud', 'server', 'network', 'vpn', 'remote',
                           'online', 'digital', 'virtual'],
                'boost_factor': 1.05
            }
        }
        
        # Merge with config patterns
        for category, details in {**default_locations, **self.location_patterns}.items():
            keywords = details.get('keywords', [])
            self.location_keywords.update(keywords)
            for keyword in keywords:
                self.location_categories[keyword.lower()] = {
                    'category': category,
                    'boost_factor': details.get('boost_factor', 1.0),
                    'context_required': details.get('context_required', False)
                }
                
    def _compile_organizational_patterns(self):
        """Compile organizational unit patterns."""
        self.organizational_keywords = set()
        self.organizational_categories = {}
        
        default_organizational = {
            'departments': {
                'keywords': ['finance', 'accounting', 'hr', 'human resources',
                           'it', 'information technology', 'operations', 'sales',
                           'marketing', 'legal', 'compliance', 'audit'],
                'case_sensitive': False,
                'boost_factor': 1.1
            },
            'levels': {
                'keywords': ['corporate', 'divisional', 'regional', 'local',
                           'group', 'team', 'unit', 'department', 'function'],
                'hierarchical': True,
                'boost_factor': 0.95
            },
            'entities': {
                'keywords': ['subsidiary', 'affiliate', 'joint venture', 'partnership',
                           'parent company', 'holding company'],
                'boost_factor': 1.05
            }
        }
        
        # Merge with config patterns
        for category, details in {**default_organizational, **self.organizational_categories}.items():
            keywords = details.get('keywords', [])
            self.organizational_keywords.update(keywords)
            for keyword in keywords:
                self.organizational_categories[keyword.lower()] = {
                    'category': category,
                    'boost_factor': details.get('boost_factor', 1.0),
                    'case_sensitive': details.get('case_sensitive', False),
                    'hierarchical': details.get('hierarchical', False)
                }
                
    def detect_where_components(self, text: str, nlp_doc: Doc) -> Dict[str, Any]:
        """
        Detect all WHERE components in text.
        
        Args:
            text: The control description text
            nlp_doc: spaCy Doc object for the text
            
        Returns:
            Dictionary containing detected WHERE components with structure:
            {
                'systems': List of detected systems
                'locations': List of detected locations
                'organizational': List of detected organizational units
                'all_components': Combined list of all components
                'primary_component': Most relevant WHERE component
                'confidence_scores': Component-level confidence scores
            }
        """
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        results = {
            'systems': [],
            'locations': [],
            'organizational': [],
            'all_components': [],
            'primary_component': None,
            'confidence_scores': {}
        }
        
        # Detect each type of WHERE component
        results['systems'] = self._detect_systems(text, nlp_doc)
        results['locations'] = self._detect_locations(text, nlp_doc)
        results['organizational'] = self._detect_organizational(text, nlp_doc)
        
        # Combine all components
        results['all_components'] = (
            results['systems'] + 
            results['locations'] + 
            results['organizational']
        )
        
        # Determine primary component
        if results['all_components']:
            results['primary_component'] = self._determine_primary_component(
                results['all_components']
            )
            
        # Calculate confidence scores
        results['confidence_scores'] = self._calculate_confidence_scores(
            results['all_components'], nlp_doc
        )
        
        # Cache results
        self._cache[cache_key] = results
        return results
        
    def _detect_systems(self, text: str, nlp_doc: Doc) -> List[Dict[str, Any]]:
        """Detect system references in the text."""
        systems = []
        text_lower = text.lower()
        
        # Check for system keywords
        for token in nlp_doc:
            token_lower = token.text.lower()
            
            # Direct keyword match
            if token_lower in self.system_keywords:
                system_info = self._extract_system_info(token, nlp_doc)
                if system_info:
                    systems.append(system_info)
                    
            # Check for system names in noun chunks
            if token.pos_ in ['PROPN', 'NOUN']:
                for chunk in nlp_doc.noun_chunks:
                    if token in chunk:
                        chunk_text = chunk.text.lower()
                        for keyword in self.system_keywords:
                            if keyword in chunk_text:
                                system_info = self._extract_system_info(
                                    chunk.root, nlp_doc, full_text=chunk.text
                                )
                                if system_info:
                                    systems.append(system_info)
                                    break
                                    
        # Deduplicate systems
        seen = set()
        unique_systems = []
        for system in systems:
            key = (system['text'].lower(), system['start_char'])
            if key not in seen:
                seen.add(key)
                unique_systems.append(system)
                
        return unique_systems
        
    def _detect_locations(self, text: str, nlp_doc: Doc) -> List[Dict[str, Any]]:
        """Detect location references in the text."""
        locations = []
        
        # Check for location keywords
        for token in nlp_doc:
            token_lower = token.text.lower()
            
            if token_lower in self.location_keywords:
                location_info = self.location_categories.get(token_lower, {})
                
                # Check if context is required
                if location_info.get('context_required', False):
                    # Look for modifiers or proper nouns nearby
                    if not self._has_location_context(token, nlp_doc):
                        continue
                        
                locations.append({
                    'text': token.text,
                    'type': 'location',
                    'category': location_info.get('category', 'unknown'),
                    'start_char': token.idx,
                    'end_char': token.idx + len(token.text),
                    'confidence': self._calculate_location_confidence(token, nlp_doc),
                    'boost_factor': location_info.get('boost_factor', 1.0)
                })
                
        # Check for named entities that might be locations
        for ent in nlp_doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:
                locations.append({
                    'text': ent.text,
                    'type': 'location',
                    'category': 'named_entity',
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'confidence': 0.9,
                    'boost_factor': 1.1
                })
                
        return locations
        
    def _detect_organizational(self, text: str, nlp_doc: Doc) -> List[Dict[str, Any]]:
        """Detect organizational unit references in the text."""
        organizational = []
        
        # Check for organizational keywords
        for token in nlp_doc:
            token_text = token.text
            token_lower = token_text.lower()
            
            if token_lower in self.organizational_keywords:
                org_info = self.organizational_categories.get(token_lower, {})
                
                # Check case sensitivity requirement
                if org_info.get('case_sensitive', False) and token_text != token_text.title():
                    continue
                    
                organizational.append({
                    'text': token.text,
                    'type': 'organizational',
                    'category': org_info.get('category', 'unknown'),
                    'start_char': token.idx,
                    'end_char': token.idx + len(token.text),
                    'confidence': self._calculate_organizational_confidence(token, nlp_doc),
                    'boost_factor': org_info.get('boost_factor', 1.0),
                    'hierarchical': org_info.get('hierarchical', False)
                })
                
        # Look for department names in noun chunks
        for chunk in nlp_doc.noun_chunks:
            chunk_lower = chunk.text.lower()
            for keyword in self.organizational_keywords:
                if keyword in chunk_lower and keyword not in [t.text.lower() for t in chunk]:
                    # Multi-word department name
                    organizational.append({
                        'text': chunk.text,
                        'type': 'organizational',
                        'category': 'department_phrase',
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'confidence': 0.85,
                        'boost_factor': 1.05
                    })
                    
        return organizational
        
    def _extract_system_info(self, token: Token, nlp_doc: Doc, 
                           full_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Extract detailed system information."""
        system_text = full_text or token.text
        system_lower = system_text.lower()
        
        # Find matching system category
        category_info = None
        for keyword in self.system_keywords:
            if keyword in system_lower:
                category_info = self.system_categories.get(keyword)
                break
                
        if not category_info:
            return None
            
        # Calculate confidence based on context
        confidence = self._calculate_system_confidence(token, nlp_doc)
        
        return {
            'text': system_text,
            'type': 'system',
            'category': category_info['category'],
            'start_char': token.idx,
            'end_char': token.idx + len(token.text),
            'confidence': confidence,
            'boost_factor': category_info['boost_factor']
        }
        
    def _has_location_context(self, token: Token, nlp_doc: Doc) -> bool:
        """Check if a location keyword has appropriate context."""
        # Check for modifiers
        for child in token.children:
            if child.dep_ in ['amod', 'compound', 'nmod']:
                return True
                
        # Check if part of a proper noun phrase
        if token.head.pos_ == 'PROPN':
            return True
            
        # Check surrounding tokens
        idx = token.i
        if idx > 0 and nlp_doc[idx - 1].pos_ == 'PROPN':
            return True
        if idx < len(nlp_doc) - 1 and nlp_doc[idx + 1].pos_ == 'PROPN':
            return True
            
        return False
        
    def _calculate_system_confidence(self, token: Token, nlp_doc: Doc) -> float:
        """Calculate confidence score for system detection."""
        confidence = 0.7  # Base confidence
        
        # Boost for prepositions indicating location
        if any(child.dep_ == 'prep' and child.text.lower() in ['in', 'on', 'within', 'from', 'using']
               for child in token.children):
            confidence *= self.confidence_factors.get('explicit_preposition', 1.2)
            
        # Boost for proper nouns
        if token.pos_ == 'PROPN':
            confidence *= self.confidence_factors.get('proper_noun', 1.1)
            
        # Boost for acronyms
        if token.text.isupper() and len(token.text) > 1:
            confidence *= self.confidence_factors.get('acronym', 0.9)
            
        # Check if it's the subject or object of a verb
        if token.dep_ in ['nsubj', 'dobj', 'pobj']:
            confidence *= 1.05
            
        return min(confidence, 1.0)
        
    def _calculate_location_confidence(self, token: Token, nlp_doc: Doc) -> float:
        """Calculate confidence score for location detection."""
        confidence = 0.65  # Base confidence for locations
        
        # Boost for prepositions
        if token.head.dep_ == 'prep':
            confidence *= self.confidence_factors.get('explicit_preposition', 1.2)
            
        # Boost for proper noun context
        if self._has_location_context(token, nlp_doc):
            confidence *= 1.15
            
        # Penalty for generic usage
        if token.dep_ == 'compound' and token.head.pos_ == 'NOUN':
            confidence *= 0.9
            
        return min(confidence, 1.0)
        
    def _calculate_organizational_confidence(self, token: Token, nlp_doc: Doc) -> float:
        """Calculate confidence score for organizational unit detection."""
        confidence = 0.75  # Base confidence
        
        # Boost for title case
        if token.text[0].isupper():
            confidence *= 1.1
            
        # Boost for possessive forms (e.g., "Finance's approval")
        if any(child.dep_ == 'poss' for child in token.children):
            confidence *= 1.15
            
        # Boost if used as subject
        if token.dep_ == 'nsubj':
            confidence *= 1.1
            
        return min(confidence, 1.0)
        
    def _determine_primary_component(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the most relevant WHERE component."""
        if not components:
            return None
            
        # Sort by confidence * boost_factor
        scored_components = []
        for comp in components:
            score = comp['confidence'] * comp.get('boost_factor', 1.0)
            scored_components.append((score, comp))
            
        scored_components.sort(key=lambda x: x[0], reverse=True)
        return scored_components[0][1]
        
    def _calculate_confidence_scores(self, components: List[Dict[str, Any]], 
                                   nlp_doc: Doc) -> Dict[str, float]:
        """Calculate overall confidence scores for each component type."""
        scores = {
            'systems': 0.0,
            'locations': 0.0,
            'organizational': 0.0,
            'overall': 0.0
        }
        
        # Calculate type-specific scores
        for comp_type in ['systems', 'locations', 'organizational']:
            type_components = [c for c in components if c['type'] == comp_type[:-1]]
            if type_components:
                # Average confidence weighted by boost factors
                total_weight = sum(c.get('boost_factor', 1.0) for c in type_components)
                weighted_conf = sum(
                    c['confidence'] * c.get('boost_factor', 1.0) 
                    for c in type_components
                )
                scores[comp_type] = weighted_conf / total_weight if total_weight > 0 else 0
                
        # Overall score
        if components:
            scores['overall'] = sum(c['confidence'] for c in components) / len(components)
            
        return scores
        
    def clear_cache(self):
        """Clear the detection cache."""
        self._cache.clear()
        
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self._cache)