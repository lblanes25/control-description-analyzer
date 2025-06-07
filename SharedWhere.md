Shared WHERE Detection Model
🎯 Project Overview
Objective: Implement WHERE as a standalone element while maintaining existing WHAT functionality through a shared detection model that allows both elements to recognize and utilize location information.
Duration: 4 weeks (reduced from 6 weeks due to less refactoring)
Approach: Additive enhancement rather than extraction
Key Principle: WHERE complements WHAT rather than competing with it

🏗️ Architecture Design
Shared Detection Model
┌─────────────────────┐
│   Shared WHERE      │
│  Detection Service  │
│ (where_service.py)  │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
┌────▼────┐ ┌───▼────┐
│  WHAT   │ │ WHERE  │
│ Analyzer│ │Analyzer│
└─────────┘ └────────┘
Key Design Principles

Non-Breaking: WHAT continues to function exactly as before
Shared Intelligence: Both elements can access location detection
Single Source of Truth: WHERE patterns defined once, used twice
Contextual Scoring: Each element scores WHERE differently based on its needs


📋 Implementation Timeline
Phase 1: Shared Service Development (Week 1)
Objective: Create the shared WHERE detection service
Tasks:

Create src/analyzers/where_service.py
Migrate WHERE detection logic from WHAT
Enhance detection patterns
Create standardized WHERE data structure
Build caching mechanism for efficiency

Technical Implementation:
# src/analyzers/where_service.py
class WhereDetectionService:
    """Shared service for WHERE detection used by multiple analyzers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._cache = {}
        self._initialize_patterns()
    
    def detect_where_components(self, text: str, nlp_doc) -> Dict[str, Any]:
        """
        Detect all WHERE components in text
        Returns structured data usable by both WHAT and WHERE analyzers
        """
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        results = {
            'systems': self._detect_systems(text, nlp_doc),
            'locations': self._detect_locations(text, nlp_doc),
            'organizational': self._detect_organizational(text, nlp_doc),
            'all_components': [],
            'primary_component': None,
            'confidence_scores': {}
        }
        
        self._cache[cache_key] = results
        return results

Configuration Structure:
# Shared WHERE configuration used by service
shared_where_config:
  detection_patterns:
    systems:
      erp_systems:
        keywords: [sap, oracle, peoplesoft, jde]
        boost_factor: 1.2
      collaboration:
        keywords: [sharepoint, teams, slack, confluence]
        boost_factor: 1.1
        
    locations:
      physical:
        keywords: [office, facility, branch, headquarters]
        context_required: true
      geographic:
        keywords: [region, country, global, domestic]
        boost_factor: 0.9
        
    organizational:
      departments:
        keywords: [finance, hr, it, operations, sales]
        case_sensitive: false
      levels:
        keywords: [corporate, divisional, regional, local]
        hierarchical: true

  # Confidence scoring rules
  confidence_factors:
    explicit_preposition: 1.2  # "in SAP"
    implicit_reference: 0.8    # "SAP report"
    contextual_mention: 0.6    # "the system" (after SAP mentioned)
  
Phase 2: WHERE Element Implementation (Week 2)
Objective: Create standalone WHERE analyzer using the shared service
Tasks:

Create src/analyzers/where.py
Implement WHERE-specific scoring logic
Add WHERE element configuration
Create WHERE-specific suggestions
Integrate with element framework

WHERE Analyzer Implementation:
# src/analyzers/where.py
from .where_service import WhereDetectionService

def enhance_where_detection(text: str, nlp, existing_keywords: List[str] = None,
                          control_type: str = None, config: Dict = None) -> Dict[str, Any]:
    """
    Enhanced WHERE detection as a standalone element
    Uses shared service but applies WHERE-specific analysis
    """
    # Initialize shared service
    where_service = WhereDetectionService(config)
    
    # Get base detection results
    where_components = where_service.detect_where_components(text, nlp(text))
    
    # Apply WHERE-specific scoring logic
    score = calculate_where_score(where_components, control_type)
    
    # Generate WHERE-specific insights
    primary_location = determine_primary_location(where_components)
    location_specificity = assess_location_specificity(where_components)
    
    # Build WHERE-specific suggestions
    suggestions = generate_where_suggestions(
        where_components, 
        location_specificity,
        control_type
    )
    
    return {
        'score': score,
        'components': where_components,
        'primary_location': primary_location,
        'location_types': categorize_locations(where_components),
        'specificity_score': location_specificity,
        'suggestions': suggestions,
        'matched_keywords': extract_matched_keywords(where_components)
    }
Element Configuration:
elements:
  # Adjusted weights to accommodate WHERE
  WHO:
    weight: 28      # Reduced from 32
  WHAT:
    weight: 30      # Reduced from 32
  WHEN:
    weight: 20      # Reduced from 22
  WHERE:
    weight: 10      # New element
  WHY:
    weight: 9       # Reduced from 11
  ESCALATION:
    weight: 3       # Unchanged
    
  # WHERE-specific configuration
  WHERE:
    weight: 10
    min_score_threshold: 0.3
    vague_location_penalty: 0.5
    control_type_relevance:
      IT: 
        systems: 1.5
        locations: 0.7
        organizational: 1.0
      Physical:
        systems: 0.7
        locations: 1.5
        organizational: 1.0
      Financial:
        systems: 1.0
        locations: 0.8
        organizational: 1.3
  
Phase 3: WHAT Integration Update (Week 3)
Objective: Update WHAT to use shared service while maintaining functionality
Tasks:

Modify WHAT to use WhereDetectionService
Maintain existing WHERE boost logic
Update confidence calculations
Ensure backward compatibility
Add configuration for WHERE importance in WHAT

WHAT Analyzer Updates:

# src/analyzers/what.py - Modified sections only

from .where_service import WhereDetectionService

class WhatDetectionConfig:
    def __init__(self, spacy_doc, existing_keywords=None, 
                 control_type=None, config=None):
        # ... existing code ...
        self.where_service = WhereDetectionService(config)
        self.where_importance_factor = config.get('what_element', {}).get(
            'where_importance_factor', 0.1
        )

class PhraseBuilder:
    def build_verb_phrase(self, token, spacy_doc, is_passive=False):
        # ... existing phrase building logic ...
        
        # Use shared service for WHERE detection
        where_components = self.config.where_service.detect_where_components(
            phrase, spacy_doc
        )
        
        # Maintain existing boost logic but with richer data
        if where_components['primary_component']:
            where_info = {
                'text': where_components['primary_component']['text'],
                'type': where_components['primary_component']['type'],
                'confidence': where_components['primary_component']['confidence']
            }
        else:
            where_info = None
            
        return verb_phrase, where_info

class ConfidenceCalculator:
    def calculate_verb_confidence(self, token, verb_strength, is_passive,
                                has_future_aux, has_subject, object_specificity,
                                completeness, verb_category, has_where_component):
        # ... existing confidence logic ...
        
        # WHERE boost remains but can be configured
        if has_where_component:
            # Now we can have variable boost based on WHERE quality
            where_boost = self.WHERE_COMPONENT_BOOST
            if has_where_component.get('confidence', 0) > 0.8:
                where_boost *= 1.1  # Extra boost for high-confidence WHERE
            confidence *= where_boost

Configuration for WHAT:
what_element:
  # Existing WHAT config...
  where_importance_factor: 0.1  # How much WHERE affects WHAT scoring
  where_boost_range:
    min: 1.05  # Minimum boost for any WHERE
    max: 1.15  # Maximum boost for perfect WHERE
  maintain_where_in_phrase: true  # Keep WHERE in action phrases
  
Phase 4: Integration and Testing (Week 4)
Objective: Complete integration and comprehensive testing
Week 4, Days 1-2: Core Integration
Tasks:

Update analyzer.py to initialize WHERE element
Add WHERE to Excel report generation
Update GUI to display WHERE results
Modify multi-control detection to consider WHERE

Core Analyzer Updates:
# src/core/analyzer.py
def _initialize_elements(self) -> Dict[str, ControlElement]:
    """Initialize elements including new WHERE element"""
    elements = {}
    config_elements = self.config.get('elements', {})
    
    # Define default elements including WHERE
    default_elements = {
        "WHO": {"weight": 28, "keywords": []},
        "WHEN": {"weight": 20, "keywords": []},
        "WHAT": {"weight": 30, "keywords": []},
        "WHERE": {"weight": 10, "keywords": []},  # NEW
        "WHY": {"weight": 9, "keywords": []},
        "ESCALATION": {"weight": 3, "keywords": []}
    }
    
    # Initialize WHERE service for shared use
    self.where_service = WhereDetectionService(self.config)
    
    # ... rest of initialization

def analyze_control(self, control_id: str, description: str, **kwargs):
    """Analyze control with WHERE element"""
    # ... existing analysis ...
    
    # WHERE analysis
    if "WHERE" in self.elements:
        where_element = self.elements["WHERE"]
        where_element.analyze(
            description, 
            self.nlp, 
            self.use_enhanced_detection,
            where_service=self.where_service,  # Pass shared service
            **context
        )

Week 4, Days 3-4: Testing Framework
Test Categories:

Backward Compatibility Tests
def test_what_scores_unchanged_without_where():
    """Ensure WHAT scores remain stable when WHERE not configured"""
    analyzer_without_where = EnhancedControlAnalyzer("config_no_where.yaml")
    analyzer_with_where = EnhancedControlAnalyzer("config_with_where.yaml")
    
    # WHAT scores should be proportionally similar
Shared Service Tests
def test_where_service_consistency():
    """Ensure WHERE service gives same results to both WHAT and WHERE"""
    service = WhereDetectionService(config)
    
    # Both analyzers should see same WHERE components
Integration Tests
def test_where_element_integration():
    """Test WHERE element in full analysis pipeline"""
    # Test scoring, suggestions, and report generation

Week 4, Day 5: Documentation and Rollout
Documentation Updates:

Update README with WHERE element description
Create WHERE configuration guide
Add WHERE examples to control_analyzer_architecture.md
Create migration guide for existing users

Migration Guide Example:
## Adding WHERE Element to Existing Configuration

The WHERE element is optional and backward-compatible. To enable:

1. Add WHERE to your elements configuration:
   ```yaml
   elements:
     WHERE:
       weight: 10
   
Adjust other weights proportionally (optional):

Reduce WHO from 32 to 28
Reduce WHAT from 32 to 30
Reduce WHEN from 22 to 20
Reduce WHY from 11 to 9


Configure WHERE detection patterns (optional):

Use defaults or customize for your domain
See where_config_template.yaml for examples

---

## 🎯 Key Advantages of Shared Model

### 1. **No Breaking Changes**
- WHAT continues to work exactly as before
- WHERE detection in WHAT is enhanced, not removed
- Existing scores remain stable

### 2. **Richer Analysis**
- WHERE element provides dedicated location insights
- WHAT maintains location context for action completeness
- Both elements benefit from improved detection

### 3. **Flexible Configuration**
- Users can enable WHERE gradually
- Weight distribution is customizable
- WHERE importance in WHAT is tunable

### 4. **Easier Implementation**
- Less refactoring required
- Lower risk of regression
- Faster delivery (4 weeks vs 6)

### 5. **Future Extensibility**
- Shared service model can be extended
- Other elements could use WHERE information
- Clean separation of concerns

---

## 📊 Success Metrics

### Technical Metrics
- WHERE detection accuracy: >85% for explicit locations
- Performance impact: <10% increase in analysis time
- Memory overhead: <50MB for WHERE service cache
- Zero regression in WHAT scores without WHERE config

### Business Metrics
- WHERE element provides actionable location insights
- Backward compatibility: 100% of existing configs work
- User adoption: WHERE element enabled by 50% of users within 3 months

---

## 🚀 Rollout Strategy

### Phase 1: Soft Launch (Week 4)
- WHERE element available but not in default config
- Early adopters can enable via configuration
- Gather feedback on detection quality

### Phase 2: Progressive Enhancement (Week 5-6)
- Add WHERE to default config template
- Update documentation with best practices
- Create WHERE-specific control examples

### Phase 3: Full Integration (Week 7-8)
- Update all sample configurations
- Add WHERE to GUI by default
- Complete integration with visualization

This shared model approach minimizes risk while maximizing value, allowing WHERE to enhance the framework without disrupting existing functionality.