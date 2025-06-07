Control Description Analyzer Simplification Project Plan
Project Overview
Objective: Implement a simplified scoring methodology that focuses on essential control elements while maintaining alignment with organizational requirements for 5W documentation.

Phase 1: Planning & Design 
Requirements Gathering

 Document feedback on WHERE element relevance by control type
# WHERE Element Classification and Scoring Decisions

## Core Classification Logic

### Primary Method: Use Control_Automation Field
- **Automated** → 'system' control type
- **Hybrid** → Analyze which aspect is more prominent (system vs location)
- **Manual** → Check for system interaction, upgrade to hybrid if found

### Manual Control Upgrade Logic
Manual controls should be upgraded to hybrid only when the system actively participates in control execution, not just documentation.

**Upgrade Criteria - System Must Perform Control Logic:**
Use verb analysis to distinguish true hybrid controls from manual controls with electronic documentation.

**Control-Participating Verbs (Upgrade to Hybrid):**
- `calculates`, `validates`, `approves`, `alerts`, `flags`
- `reconciles`, `generates`, `processes`, `identifies`, `matches`
- `automatically [action]`, `system [control verb]`

**Documentation Verbs (Remain Manual):**
- `saves`, `stores`, `documents`, `records`, `enters`
- `uploads`, `maintains`, `tracks`, `files`

**Examples:**
- "System validates transaction limits" → Upgrade (system performs control)
- "Manager saves findings in SharePoint" → No upgrade (documentation only)
- "System flags exceptions for review" → Upgrade (system participates)
- "Results are entered into Excel" → No upgrade (documentation only)

**System Interaction Patterns to Detect (Control-Participating Only):**
- `system [validates/calculates/processes/generates]`
- `automatically [control action]`
- `[system] identifies/flags/alerts`
- `interface [validates/reconciles/matches]`

## Hybrid Control Classification

For controls marked as hybrid (or upgraded manual), determine prominence:

### System Context Indicators (Weight: 2)
- System action verbs: generates, calculates, validates, processes
- Automation phrases: automatically, system-generated
- Data interaction: enters data into, records in, uploads to
- Direct system name mentions

### Location Context Indicators (Weight: 1)  
- Physical location references: at branch, on-site, in person
- Role-based actions: teller performs, manager reviews
- Physical elements: vault, facility, premises

**Decision Logic:**
- If system_score > location_score → 'system'
- If location_score > 0 → 'location_dependent'
- Else → 'other'

## WHERE Element Detection

Control must have WHERE element present to receive points:

**WHERE Indicators:**
- Preposition patterns: `in/at/within/across/through/via [location]`
- System mentions: system, application, platform, database
- Location mentions: branch, office, facility, department, division

## Scoring Rules

**WHERE Points by Control Type:**
- System controls: 10 points (if WHERE present)
- Location-dependent controls: 5 points (if WHERE present)  
- Other controls: 0 points (regardless of WHERE)

**No WHERE Element = 0 Points** (regardless of control type)

## Implementation Examples

```
"System validates transaction limits and flags exceptions for manager review"
- Automation: manual
- Has control-participating verbs: YES (validates, flags)
- Upgraded to: hybrid
- System context > location context
- Final type: system
- WHERE present: YES (system context)
- Score: 10 points

"Branch manager reviews daily exception report and saves findings in SharePoint"
- Automation: manual
- Has control-participating verbs: NO (saves = documentation only)
- Remains: manual
- Final type: other
- WHERE present: YES (in SharePoint - but documentation location)
- Score: 0 points

"Automated reconciliation identifies breaks and analyst investigates discrepancies"
- Automation: manual  
- Has control-participating verbs: YES (identifies)
- Upgraded to: hybrid
- System context > location context
- Final type: system
- WHERE present: YES (system performs identification)
- Score: 10 points

"Security guard performs physical vault inspection"
- Automation: manual
- Has system interaction: NO
- Remains: manual
- Final type: location_dependent
- WHERE present: YES (vault)
- Score: 5 points

"Senior management reviews quarterly risk reports"
- Automation: manual
- Has system interaction: NO
- Remains: manual
- Final type: other
- WHERE present: NO
- Score: 0 points
```

## Configuration Structure

```yaml
classification:
  system_names: [list of org systems]
  control_participating_verbs: [calculates, validates, approves, alerts, flags, reconciles, generates, processes, identifies, matches]
  documentation_verbs: [saves, stores, documents, records, enters, uploads, maintains, tracks, files]
  system_context_weight: 2
  location_context_weight: 1

scoring:
  conditional_elements:
    WHERE:
      system_controls: 10
      location_dependent: 5
      other: 0
```

Technical Assessment

 Review current codebase and identify modification points
 Document all configuration files requiring updates
 Create technical debt list for cleanup opportunities

Initial Design Documentation

 Draft updated scoring methodology document
 Create visual comparison: current vs. proposed scoring
 Prepare stakeholder communication plan
 Schedule Week 2 design review sessions

Detailed Design & Approval
Algorithm Design

Finalize scoring algorithm specifications:

Core element calculations (WHO/WHAT/WHEN at 30/35/35)
Conditional WHERE scoring logic
Demerit application rules
Category threshold definitions


 Create pseudocode for implementation
 Design control type classification system for WHERE relevance

Wednesday: Stakeholder Review

 Present simplified framework to key stakeholders
 Conduct working session to test scoring on 10 sample controls
 Document feedback and required adjustments
 Obtain preliminary approval to proceed

Thursday-Friday: Implementation Planning

 Create detailed technical implementation plan
 Define test scenarios and acceptance criteria
 Prepare rollback plan
 Finalize Phase 2 timeline

Deliverables:

Approved Scoring Methodology Document v2.0
Technical Implementation Plan
Stakeholder Communication Matrix
Test Plan with 50+ control examples

Phase 2: Technical Implementation (Weeks 3-5)
Week 3: Core Algorithm Updates
Monday-Tuesday: Configuration Updates

# Update config.yaml structure
scoring:
  core_elements:
    WHO: 30
    WHAT: 35  
    WHEN: 35
  
  conditional_elements:
    WHERE:
      system_controls: 10
      location_dependent: 5
      other: 0
      
  demerits:
    vague_terms: -2  # per term, uncapped
    multiple_controls: -10
    missing_accountability: -5
    untestable_timing: -5
    
  categories:
    effective: 75
    adequate: 50
  
Wednesday-Thursday: Core Scoring Engine

 Refactor _calculate_overall_score() method
 Implement conditional WHERE scoring logic
 Remove ESCALATION scoring components
 Update WHY to feedback-only status
 Implement uncapped demerit system

Friday: Control Type Classification

 Develop control type detection logic:
 def classify_control_type(control_description):
    # Returns: 'system', 'location_dependent', or 'other'
 

 Create keyword mappings for classification
 Test classification on sample controls

Week 4: Detection Algorithm Refinement
Monday-Tuesday: Enhanced Element Detection

 Simplify WHO detection for binary scoring:

Specific role/system → Full points
Department/team → Half points
Vague → Zero points


 Update WHAT detection for three-tier scoring
 Refine WHEN detection for testability focus

Demerit Detection

 Expand vague terms dictionary
 Implement multiple control detection algorithm
 Create accountability assessment logic
 Develop timing testability checker

Feedback System Updates

 Convert WHY scoring to quality indicator
 Add ESCALATION soft flag capability
 Update recommendation engine for new framework
 Enhance feedback messages for clarity

Testing & Refinement
Monday-Tuesday: Unit Testing

 Test each element scorer independently
 Verify demerit calculations
 Validate conditional WHERE logic
 Ensure backward compatibility where needed

Integration Testing

 Test complete scoring pipeline
 Validate Excel output formatting
 Verify feedback generation
 Test edge cases and error handling
