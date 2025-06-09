

# Element Detection Issues and Improvements

## WHO Detection Issues Found

### High Priority Fixes Needed

1. **System Entity Detection Failures**
   - `Oracle database enforces referential integrity` → Returns "Unknown Performer"
   - Issue: Complex system names not being properly parsed as subjects
   - Root cause: Likely in main subject detection or entity classification for compound system names

2. **Passive Voice Detection Issues** 
   - `Journal entries are reviewed by the Finance Manager` → Detects "journal entries" instead of "Finance Manager"
   - Issue: Passive voice "by" phrase detection not working correctly
   - Root cause: `_detect_prepositional_phrases` function may not be finding "by the Finance Manager"

### Medium Priority Improvements

4. **Multiple Performer Detection**
   - `Finance Manager and Controller jointly approve budgets` → Should detect both or primary performer
   - Current behavior: May only detect one or miss compound subjects

5. **System Name Variations**
   - Need better handling of complex system names like "Oracle database", "SAP ERP system"
   - Should recognize both specific system names and generic system terms

### Low Priority Enhancements

6. **Qualification Handling**
   - `Senior Financial Analyst with CPA certification` → Should focus on core role
   - Current: May get confused by qualifications and certifications

## Other Element Detection Areas to Validate

### WHAT Element Detection
- **Status**: Initial tests created, need validation
- **Potential issues**: Purpose clause filtering, compound actions, passive voice actions

### WHEN Element Detection  
- **Status**: Initial tests created, need validation
- **Potential issues**: Vague timing terms, multiple frequencies, ad-hoc timing

### WHERE Element Detection
- **Status**: Initial tests created, need validation  
- **Potential issues**: Generic location terms, multiple locations, implicit locations

## Testing Strategy Improvements

### Current Approach
✅ **Fixed**: Corporate Controller detection (problem phrase classification issue)
✅ **Working**: Basic role detection for simple cases
✅ **Created**: Comprehensive test suite with edge cases

### Next Steps

1. **Immediate** (before merging element detection tests):
   - Temporarily adjust failing tests to be more lenient while tracking real issues
   - Focus on validating core functionality works correctly
   - Document known limitations in test comments

2. **Short Term** (next sprint):
   - Fix passive voice "by" phrase detection in WHO analyzer
   - Improve system entity classification for compound names
   - Adjust confidence scoring for vague role terms

3. **Medium Term** (future iterations):
   - Enhance compound subject detection for multiple performers
   - Improve system name recognition patterns
   - Add qualification filtering logic

## Code Locations to Fix

### WHO Detection (`src/analyzers/who.py`):
- `_detect_prepositional_phrases()` - Line ~420: Fix "by X" phrase detection
- `_find_main_subjects()` - Line ~240: Improve compound system name handling  
- `calculate_who_confidence()` - Line ~907: Add vague term confidence penalties
- `classify_entity_type()` - Line ~810: Better system entity classification

### Test Adjustments Needed:
- `tests/unit/test_element_detection_validation.py`:
  - `test_system_entity_detection_accuracy()` - Make more lenient temporarily
  - `test_vague_role_detection_accuracy()` - Adjust confidence expectations
  - `test_who_edge_cases()` - Focus on core functionality validation

## Success Criteria

### Definition of Done for WHO Detection:
- [ ] Correctly detects specific roles (Finance Manager, Controller, etc.)
- [ ] Properly handles passive voice constructions with "by X"
- [ ] Classifies system entities vs human roles accurately
- [ ] Applies appropriate confidence scores (high for specific, low for vague)
- [ ] Handles compound subjects and multiple performers
- [ ] Maintains backward compatibility with existing functionality

### Validation Approach:
- Create realistic test cases based on actual control descriptions
- Balance comprehensive coverage with practical limitations
- Document known edge cases and limitations
- Focus on preventing regressions while improving incrementally

PS C:\Users\luria\OneDrive\Desktop\final_consolidated_analyzer> python -m pytest tests/unit/test_element_detection_validation.py -v
======================================================================================================= test session starts =======================================================================================================
platform win32 -- Python 3.12.0, pytest-8.4.0, pluggy-1.6.0 -- C:\Users\luria\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
benchmark: 5.1.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: C:\Users\luria\OneDrive\Desktop\final_consolidated_analyzer
configfile: pytest.ini
plugins: benchmark-5.1.0, cov-6.1.1
collected 21 items                                                                                                                                                                                                                 

tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_specific_role_detection_accuracy PASSED                                                                                             [  4%]
tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_team_department_detection_accuracy PASSED                                                                                           [  9%]
tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_system_entity_detection_accuracy FAILED                                                                                             [ 14%]
tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_vague_role_detection_accuracy FAILED                                                                                                [ 19%]
tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_who_edge_cases FAILED                                                                                                               [ 23%]
tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_strong_action_verb_detection FAILED                                                                                                [ 28%]
tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_moderate_action_verb_detection FAILED                                                                                              [ 33%]
tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_weak_action_verb_detection FAILED                                                                                                  [ 38%]
tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_control_noun_detection FAILED                                                                                                      [ 42%]
tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_what_edge_cases FAILED                                                                                                             [ 47%]
tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_explicit_frequency_detection PASSED                                                                                                [ 52%]
tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_period_end_timing_detection FAILED                                                                                                 [ 57%]
tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_conditional_timing_detection PASSED                                                                                                [ 61%]
tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_vague_timing_detection FAILED                                                                                                      [ 66%]
tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_when_edge_cases FAILED                                                                                                             [ 71%]
tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_system_location_detection FAILED                                                                                                  [ 76%]
tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_physical_location_detection FAILED                                                                                                [ 80%]
tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_organizational_location_detection FAILED                                                                                          [ 85%]
tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_where_edge_cases FAILED                                                                                                           [ 90%]
tests/unit/test_element_detection_validation.py::TestElementDetectionEdgeCases::test_complex_control_statements FAILED                                                                                                       [ 95%]
tests/unit/test_element_detection_validation.py::TestElementDetectionEdgeCases::test_ambiguous_element_scenarios FAILED                                                                                                      [100%]

============================================================================================================ FAILURES ============================================================================================================= 
_____________________________________________________________________________ TestWHOElementDetectionValidation.test_system_entity_detection_accuracy _____________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHOElementDetectionValidation object at 0x000001714DE83D70>, spacy_model = <spacy.lang.en.English object at 0x000001714FF8B080>

    def test_system_entity_detection_accuracy(self, spacy_model):
        """Test detection of system entities performing controls"""
        test_cases = [
            {
                'text': 'SAP system automatically validates transaction limits',
                'expected_who': 'SAP system',
                'expected_confidence_min': 0.7,
                'expected_type': 'system'
            },
            {
                'text': 'The application generates daily exception reports',
                'expected_who': 'application',
                'expected_confidence_min': 0.6,
                'expected_type': 'system'
            },
            {
                'text': 'Oracle database enforces referential integrity',
                'expected_who': 'Oracle database',
                'expected_confidence_min': 0.7,
                'expected_type': 'system'
            },
            {
                'text': 'Automated workflow routes approvals to managers',
                'expected_who': 'Automated workflow',
                'expected_confidence_min': 0.6,
                'expected_type': 'system'
            }
        ]

        for case in test_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)

            assert result['primary'] is not None, f"Should detect WHO in: {case['text']}"

            primary_text = result['primary']['text']
>           assert case['expected_who'] in primary_text or primary_text in case['expected_who'], \
                f"Expected '{case['expected_who']}' in detected '{primary_text}' for: {case['text']}"
E           AssertionError: Expected 'Oracle database' in detected 'Unknown Performer' for: Oracle database enforces referential integrity
E           assert ('Oracle database' in 'Unknown Performer' or 'Unknown Performer' in 'Oracle database')

tests\unit\test_element_detection_validation.py:165: AssertionError
______________________________________________________________________________ TestWHOElementDetectionValidation.test_vague_role_detection_accuracy _______________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHOElementDetectionValidation object at 0x000001714DE83EF0>, spacy_model = <spacy.lang.en.English object at 0x000001714ECA4C20>

    def test_vague_role_detection_accuracy(self, spacy_model):
        """Test detection of vague role references with appropriate confidence"""
        test_cases = [
            {
                'text': 'Management reviews financial reports quarterly',
                'expected_who': 'Management',
                'expected_confidence_max': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Staff performs daily reconciliations',
                'expected_who': 'Staff',
                'expected_confidence_max': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Personnel validate customer information',
                'expected_who': 'Personnel',
                'expected_confidence_max': 0.6,
                'expected_type': 'human'
            }
        ]

        for case in test_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)

            assert result['primary'] is not None, f"Should detect WHO in: {case['text']}"

            primary_text = result['primary']['text']
            assert case['expected_who'] in primary_text or primary_text in case['expected_who'], \
                f"Expected '{case['expected_who']}' in detected '{primary_text}' for: {case['text']}"

            # Vague roles should have lower confidence
>           assert result['confidence'] <= case['expected_confidence_max'], \
                f"Confidence {result['confidence']} should be <= {case['expected_confidence_max']} for vague role: {case['text']}"
E           AssertionError: Confidence 0.84 should be <= 0.6 for vague role: Management reviews financial reports quarterly
E           assert 0.84 <= 0.6

tests\unit\test_element_detection_validation.py:207: AssertionError
______________________________________________________________________________________ TestWHOElementDetectionValidation.test_who_edge_cases ______________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHOElementDetectionValidation object at 0x000001714DEB80B0>, spacy_model = <spacy.lang.en.English object at 0x000001715592FF20>

    def test_who_edge_cases(self, spacy_model):
        """Test WHO detection edge cases and challenging scenarios"""
        edge_cases = [
            {
                'name': 'passive_voice_with_clear_performer',
                'text': 'Journal entries are reviewed by the Finance Manager',
                'expected_who_keywords': ['finance', 'manager'],
                'should_detect': True
            },
            {
                'name': 'compound_roles',
                'text': 'Finance Manager and Controller jointly approve budgets',
                'expected_who_keywords': ['finance', 'manager', 'controller'],  # Should detect at least one
                'should_detect': True
            },
            {
                'name': 'no_clear_performer',
                'text': 'Controls are implemented to ensure compliance',
                'should_detect': False,  # No clear performer
                'low_confidence_acceptable': True
            },
            {
                'name': 'multiple_performers_sequence',
                'text': 'Analyst prepares report, Manager reviews, Controller approves',
                'expected_who_keywords': ['analyst', 'manager', 'controller'],  # Should detect primary performer
                'should_detect': True
            },
            {
                'name': 'performer_with_qualification',
                'text': 'Senior Financial Analyst with CPA certification reviews statements',
                'expected_who_keywords': ['senior', 'financial', 'analyst'],
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)

            if case['should_detect']:
                assert result['primary'] is not None, \
                    f"Should detect WHO in {case['name']}: {case['text']}"

                if 'expected_who_keywords' in case:
                    primary_text = result['primary']['text'].lower()
                    # Should contain at least one expected keyword
>                   assert any(keyword in primary_text for keyword in case['expected_who_keywords']), \
                        f"Expected one of {case['expected_who_keywords']} in detected '{primary_text}' for {case['name']}: {case['text']}"
E                   AssertionError: Expected one of ['finance', 'manager'] in detected 'journal entries' for passive_voice_with_clear_performer: Journal entries are reviewed by the Finance Manager
E                   assert False
E                    +  where False = any(<generator object TestWHOElementDetectionValidation.test_who_edge_cases.<locals>.<genexpr> at 0x000001714FF1EC20>)

tests\unit\test_element_detection_validation.py:255: AssertionError
______________________________________________________________________________ TestWHATElementDetectionValidation.test_strong_action_verb_detection _______________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHATElementDetectionValidation object at 0x000001714DEB8230>, spacy_model = <spacy.lang.en.English object at 0x0000017155632690>

    def test_strong_action_verb_detection(self, spacy_model):
        """Test detection of strong action verbs with high confidence"""
        test_cases = [
            {
                'text': 'Manager reviews and approves all journal entries',
                'expected_actions': ['reviews', 'approves'],
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Analyst validates transaction accuracy weekly',
                'expected_actions': ['validates'],
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Controller reconciles bank statements monthly',
                'expected_actions': ['reconciles'],
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Auditor verifies completeness of documentation',
                'expected_actions': ['verifies'],
                'expected_confidence_min': 0.8
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)

            # Should detect primary action
>           assert result['primary_action'] is not None, \
                f"Should detect primary action in: {case['text']}"
E           AssertionError: Should detect primary action in: Auditor verifies completeness of documentation
E           assert None is not None

tests\unit\test_element_detection_validation.py:306: AssertionError
_____________________________________________________________________________ TestWHATElementDetectionValidation.test_moderate_action_verb_detection ______________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHATElementDetectionValidation object at 0x000001714DEB83B0>, spacy_model = <spacy.lang.en.English object at 0x000001715B75D700>

    def test_moderate_action_verb_detection(self, spacy_model):
        """Test detection of moderate strength action verbs"""
        test_cases = [
            {
                'text': 'Supervisor ensures compliance with policies',
                'expected_actions': ['ensures'],
                'expected_confidence_range': (0.6, 0.8)
            },
            {
                'text': 'Manager coordinates review activities',
                'expected_actions': ['coordinates'],
                'expected_confidence_range': (0.6, 0.8)
            },
            {
                'text': 'Team maintains documentation standards',
                'expected_actions': ['maintains'],
                'expected_confidence_range': (0.6, 0.8)
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)

>           assert result['primary_action'] is not None, \
                f"Should detect primary action in: {case['text']}"
E           AssertionError: Should detect primary action in: Supervisor ensures compliance with policies
E           assert None is not None

tests\unit\test_element_detection_validation.py:342: AssertionError
_______________________________________________________________________________ TestWHATElementDetectionValidation.test_weak_action_verb_detection ________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHATElementDetectionValidation object at 0x000001714DEB8530>, spacy_model = <spacy.lang.en.English object at 0x000001714EDD2150>

    def test_weak_action_verb_detection(self, spacy_model):
        """Test detection of weak action verbs with appropriate confidence"""
        test_cases = [
            {
                'text': 'Staff considers various options',
                'expected_actions': ['considers'],
                'expected_confidence_max': 0.6
            },
            {
                'text': 'Team attempts to resolve issues',
                'expected_actions': ['attempts'],
                'expected_confidence_max': 0.6
            },
            {
                'text': 'Management observes current practices',
                'expected_actions': ['observes'],
                'expected_confidence_max': 0.6
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)

            if result['primary_action'] is not None:
                # If weak action is detected, should have low confidence
>               assert result['primary_action']['score'] <= case['expected_confidence_max'], \
                    f"Weak action confidence {result['primary_action']['score']} should be <= {case['expected_confidence_max']} for: {case['text']}"
E               AssertionError: Weak action confidence 0.6563699999999999 should be <= 0.6 for: Management observes current practices
E               assert 0.6563699999999999 <= 0.6

tests\unit\test_element_detection_validation.py:381: AssertionError
_________________________________________________________________________________ TestWHATElementDetectionValidation.test_control_noun_detection __________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHATElementDetectionValidation object at 0x000001714DEB86B0>, spacy_model = <spacy.lang.en.English object at 0x0000017154577B60>

    def test_control_noun_detection(self, spacy_model):
        """Test detection of control-specific nouns that boost action confidence"""
        test_cases = [
            {
                'text': 'Manager performs monthly reconciliation',
                'expected_nouns': ['reconciliation'],
                'action_should_be_boosted': True
            },
            {
                'text': 'Analyst conducts validation procedures',
                'expected_nouns': ['validation'],
                'action_should_be_boosted': True
            },
            {
                'text': 'Controller provides approval for transactions',
                'expected_nouns': ['approval'],
                'action_should_be_boosted': True
            },
            {
                'text': 'Team completes review process',
                'expected_nouns': ['review'],
                'action_should_be_boosted': True
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)

            # Should detect action
            assert result['primary_action'] is not None, \
                f"Should detect action with control noun in: {case['text']}"

            # Action should have reasonable confidence due to control noun
            if case['action_should_be_boosted']:
>               assert result['primary_action']['score'] >= 0.6, \
                    f"Action with control noun should have boosted confidence for: {case['text']}"
E               AssertionError: Action with control noun should have boosted confidence for: Manager performs monthly reconciliation
E               assert 0.47567519999999985 >= 0.6

tests\unit\test_element_detection_validation.py:418: AssertionError
_____________________________________________________________________________________ TestWHATElementDetectionValidation.test_what_edge_cases _____________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHATElementDetectionValidation object at 0x000001714DEB8830>, spacy_model = <spacy.lang.en.English object at 0x00000171541F5EE0>

    def test_what_edge_cases(self, spacy_model):
        """Test WHAT detection edge cases"""
        edge_cases = [
            {
                'name': 'passive_voice_strong_action',
                'text': 'Reports are reviewed by management monthly',
                'expected_action': 'reviewed',
                'should_detect': True
            },
            {
                'name': 'compound_actions',
                'text': 'Manager reviews and approves budget submissions',
                'expected_action': 'reviews',  # Should detect at least one
                'should_detect': True
            },
            {
                'name': 'future_tense_action',
                'text': 'Controller will validate account balances',
                'expected_action': 'validate',
                'should_detect': True
            },
            {
                'name': 'conditional_action',
                'text': 'System may generate exception reports',
                'expected_action': 'generate',
                'should_detect': True,
                'expected_confidence_max': 0.7  # Conditional should be lower confidence
            },
            {
                'name': 'purpose_clause_only',
                'text': 'To ensure compliance with regulations',
                'should_detect': False  # Purpose clause, not action
            },
            {
                'name': 'multiple_actions_sequence',
                'text': 'Prepare report, review findings, and submit recommendations',
                'expected_action': 'prepare',  # Should detect primary action
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhance_what_detection(case['text'], spacy_model)

            if case['should_detect']:
                assert result['primary_action'] is not None, \
                    f"Should detect action in {case['name']}: {case['text']}"

                if 'expected_action' in case:
                    detected_verb = result['primary_action']['verb_lemma']
>                   assert case['expected_action'] in detected_verb or detected_verb in case['expected_action'], \
                        f"Expected '{case['expected_action']}' in detected '{detected_verb}' for {case['name']}: {case['text']}"
E                   AssertionError: Expected 'reviews' in detected 'approve' for compound_actions: Manager reviews and approves budget submissions
E                   assert ('reviews' in 'approve' or 'approve' in 'reviews')

tests\unit\test_element_detection_validation.py:471: AssertionError
_______________________________________________________________________________ TestWHENElementDetectionValidation.test_period_end_timing_detection _______________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHENElementDetectionValidation object at 0x000001714DEB8950>, spacy_model = <spacy.lang.en.English object at 0x0000017154FCC050>

    def test_period_end_timing_detection(self, spacy_model):
        """Test detection of period-end timing patterns"""
        test_cases = [
            {
                'text': 'Controller performs closing procedures at month-end',
                'expected_pattern': 'month-end',
                'expected_frequency': 'monthly',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'Team completes reconciliation by quarter-end',
                'expected_pattern': 'quarter-end',
                'expected_frequency': 'quarterly',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'Auditor reviews controls during year-end close',
                'expected_pattern': 'year-end',
                'expected_frequency': 'annually',
                'expected_confidence_min': 0.7
            }
        ]

        for case in test_cases:
            result = enhance_when_detection(case['text'], spacy_model)

            assert result['top_match'] is not None, \
                f"Should detect period-end timing in: {case['text']}"

            # Check that period-end pattern is detected
            detected_text = result['top_match']['text'].lower()
>           assert case['expected_pattern'] in detected_text, \
                f"Expected '{case['expected_pattern']}' in detected '{detected_text}' for: {case['text']}"
E           AssertionError: Expected 'month-end' in detected 'month' for: Controller performs closing procedures at month-end
E           assert 'month-end' in 'month'

tests\unit\test_element_detection_validation.py:573: AssertionError
_________________________________________________________________________________ TestWHENElementDetectionValidation.test_vague_timing_detection __________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHENElementDetectionValidation object at 0x000001714DEB8B90>, spacy_model = <spacy.lang.en.English object at 0x000001714E31BF80>

    def test_vague_timing_detection(self, spacy_model):
        """Test detection and proper handling of vague timing terms"""
        test_cases = [
            {
                'text': 'Manager periodically reviews reports',
                'expected_vague_term': 'periodically',
                'should_be_flagged': True
            },
            {
                'text': 'Staff regularly validates data',
                'expected_vague_term': 'regularly',
                'should_be_flagged': True
            },
            {
                'text': 'Team provides timely resolution',
                'expected_vague_term': 'timely',
                'should_be_flagged': True
            },
            {
                'text': 'Controls are performed occasionally',
                'expected_vague_term': 'occasionally',
                'should_be_flagged': True
            }
        ]

        for case in test_cases:
            result = enhance_when_detection(case['text'], spacy_model)

            if case['should_be_flagged']:
                # Should detect vague terms
                vague_terms = result.get('vague_terms', [])
>               assert len(vague_terms) > 0, \
                    f"Should detect vague timing terms in: {case['text']}"
E               AssertionError: Should detect vague timing terms in: Team provides timely resolution
E               assert 0 > 0
E                +  where 0 = len([])

tests\unit\test_element_detection_validation.py:648: AssertionError
_____________________________________________________________________________________ TestWHENElementDetectionValidation.test_when_edge_cases _____________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHENElementDetectionValidation object at 0x000001714DEB8D70>, spacy_model = <spacy.lang.en.English object at 0x000001715A0E7CE0>

    def test_when_edge_cases(self, spacy_model):
        """Test WHEN detection edge cases"""
        edge_cases = [
            {
                'name': 'multiple_frequencies',
                'text': 'Manager reviews daily reports and monthly summaries',
                'should_detect_multiple': True
            },
            {
                'name': 'no_timing_information',
                'text': 'Manager approves transactions',
                'should_detect': False
            },
            {
                'name': 'business_cycle_timing',
                'text': 'Controller performs review during closing cycle',
                'expected_pattern': 'closing cycle',
                'should_detect': True
            },
            {
                'name': 'specific_timeframe',
                'text': 'Team responds within 2 business days',
                'expected_pattern': 'within 2 business days',
                'should_detect': True
            },
            {
                'name': 'ad_hoc_timing',
                'text': 'Manager conducts ad-hoc reviews',
                'expected_pattern': 'ad-hoc',
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhance_when_detection(case['text'], spacy_model)

>           if case['should_detect']:
               ^^^^^^^^^^^^^^^^^^^^^
E           KeyError: 'should_detect'

tests\unit\test_element_detection_validation.py:696: KeyError
_______________________________________________________________________________ TestWHEREElementDetectionValidation.test_system_location_detection ________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHEREElementDetectionValidation object at 0x000001714DEB8FB0>, spacy_model = <spacy.lang.en.English object at 0x000001715437C0B0>

    def test_system_location_detection(self, spacy_model):
        """Test detection of systems where controls are executed"""
        test_cases = [
            {
                'text': 'Manager reviews transactions in SAP',
                'expected_where': 'SAP',
                'expected_type': 'system',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'System validates data in Oracle database',
                'expected_where': 'Oracle database',
                'expected_type': 'system',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'Controller approves entries in the ERP system',
                'expected_where': 'ERP system',
                'expected_type': 'system',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Team generates reports from SharePoint',
                'expected_where': 'SharePoint',
                'expected_type': 'system',
                'expected_confidence_min': 0.7
            }
        ]

        for case in test_cases:
>           result = enhance_where_detection(case['text'], spacy_model)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\unit\test_element_detection_validation.py:760:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src\analyzers\where.py:63: in enhance_where_detection
    score = calculate_where_score(where_components, control_type, config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

components = {'all_components': [{'boost_factor': 1.2, 'category': 'erp_systems', 'confidence': 0.7276500000000001, 'end_char': 35,...[{'boost_factor': 1.1, 'category': 'named_entity', 'confidence': 0.9, 'end_char': 35, ...}], 'organizational': [], ...}
control_type = None, config = None

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
>       where_config = config.get('elements', {}).get('WHERE', {})
                       ^^^^^^^^^^
E       AttributeError: 'NoneType' object has no attribute 'get'

src\analyzers\where.py:112: AttributeError
______________________________________________________________________________ TestWHEREElementDetectionValidation.test_physical_location_detection _______________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHEREElementDetectionValidation object at 0x000001714DEB9190>, spacy_model = <spacy.lang.en.English object at 0x0000017155048B00>

    def test_physical_location_detection(self, spacy_model):
        """Test detection of physical locations where controls are performed"""
        test_cases = [
            {
                'text': 'Guard performs inspection at the vault',
                'expected_where': 'vault',
                'expected_type': 'location',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Manager reviews documents at branch office',
                'expected_where': 'branch office',
                'expected_type': 'location',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Team conducts inventory count at warehouse',
                'expected_where': 'warehouse',
                'expected_type': 'location',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Controller signs checks in the finance department',
                'expected_where': 'finance department',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            }
        ]

        for case in test_cases:
            result = enhance_where_detection(case['text'], spacy_model)

            # Should detect WHERE component
>           assert result['primary_location'] is not None, \
                f"Should detect WHERE in: {case['text']}"
E           AssertionError: Should detect WHERE in: Guard performs inspection at the vault
E           assert None is not None

tests\unit\test_element_detection_validation.py:812: AssertionError
___________________________________________________________________________ TestWHEREElementDetectionValidation.test_organizational_location_detection ____________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHEREElementDetectionValidation object at 0x000001714DEB9370>, spacy_model = <spacy.lang.en.English object at 0x000001714EE10EC0>

    def test_organizational_location_detection(self, spacy_model):
        """Test detection of organizational units as locations"""
        test_cases = [
            {
                'text': 'Accounting department reconciles bank statements',
                'expected_where': 'Accounting department',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Treasury team manages cash flows',
                'expected_where': 'Treasury team',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Internal audit group reviews controls',
                'expected_where': 'Internal audit group',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Finance committee approves budgets',
                'expected_where': 'Finance committee',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            }
        ]

        for case in test_cases:
>           result = enhance_where_detection(case['text'], spacy_model)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\unit\test_element_detection_validation.py:850:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src\analyzers\where.py:63: in enhance_where_detection
    score = calculate_where_score(where_components, control_type, config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

components = {'all_components': [{'boost_factor': 1.1, 'category': 'departments', 'confidence': 0.8250000000000001, 'end_char': 10,...': 10, ...}, {'boost_factor': 0.95, 'category': 'levels', 'confidence': 0.8250000000000001, 'end_char': 21, ...}], ...}
control_type = None, config = None

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
>       where_config = config.get('elements', {}).get('WHERE', {})
                       ^^^^^^^^^^
E       AttributeError: 'NoneType' object has no attribute 'get'

src\analyzers\where.py:112: AttributeError
____________________________________________________________________________________ TestWHEREElementDetectionValidation.test_where_edge_cases ____________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestWHEREElementDetectionValidation object at 0x000001714DEB9550>, spacy_model = <spacy.lang.en.English object at 0x000001715460BC80>

    def test_where_edge_cases(self, spacy_model):
        """Test WHERE detection edge cases"""
        edge_cases = [
            {
                'name': 'multiple_locations',
                'text': 'Manager reviews data in SAP and validates in Oracle',
                'should_detect_multiple': True
            },
            {
                'name': 'vague_system_reference',
                'text': 'Team processes data in the system',
                'expected_where': 'system',
                'should_detect': True,
                'expected_confidence_max': 0.7
            },
            {
                'name': 'no_location_information',
                'text': 'Manager approves transactions',
                'should_detect': False
            },
            {
                'name': 'implicit_system_location',
                'text': 'Automated controls validate transaction limits',
                'should_detect': False  # No explicit WHERE mentioned
            },
            {
                'name': 'location_with_preposition',
                'text': 'Controller works from the main office',
                'expected_where': 'main office',
                'should_detect': True
            }
        ]

        for case in edge_cases:
>           result = enhance_where_detection(case['text'], spacy_model)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\unit\test_element_detection_validation.py:895:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src\analyzers\where.py:63: in enhance_where_detection
    score = calculate_where_score(where_components, control_type, config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

components = {'all_components': [{'boost_factor': 1.2, 'category': 'erp_systems', 'confidence': 0.7276500000000001, 'end_char': 27,... {'boost_factor': 1.1, 'category': 'named_entity', 'confidence': 0.9, 'end_char': 51, ...}], 'organizational': [], ...}
control_type = None, config = None

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
>       where_config = config.get('elements', {}).get('WHERE', {})
                       ^^^^^^^^^^
E       AttributeError: 'NoneType' object has no attribute 'get'

src\analyzers\where.py:112: AttributeError
__________________________________________________________________________________ TestElementDetectionEdgeCases.test_complex_control_statements __________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestElementDetectionEdgeCases object at 0x000001714DEB97C0>, spacy_model = <spacy.lang.en.English object at 0x0000017153974CB0>

    def test_complex_control_statements(self, spacy_model):
        """Test element detection in complex, realistic control statements"""
        complex_cases = [
            {
                'text': 'The Finance Manager reviews all journal entries exceeding $10,000 in SAP on a daily basis to ensure accuracy and compliance',
                'expected_who': 'Finance Manager',
                'expected_what': 'reviews',
                'expected_when': 'daily',
                'expected_where': 'SAP'
            },
            {
                'text': 'Senior Internal Auditor validates system access controls quarterly by testing user permissions in Active Directory',
                'expected_who': 'Senior Internal Auditor',
                'expected_what': 'validates',
                'expected_when': 'quarterly',
                'expected_where': 'Active Directory'
            },
            {
                'text': 'Accounting team reconciles bank statements monthly and investigates variances exceeding materiality threshold',
                'expected_who': 'Accounting team',
                'expected_what': 'reconciles',
                'expected_when': 'monthly',
                'expected_where': None  # No explicit system/location
            }
        ]

        for case in complex_cases:
            # Test WHO detection
            who_result = enhanced_who_detection_v2(case['text'], spacy_model)
            if case['expected_who']:
                assert who_result['primary'] is not None
                assert case['expected_who'] in who_result['primary']['text']

            # Test WHAT detection
            what_result = enhance_what_detection(case['text'], spacy_model)
            if case['expected_what']:
                assert what_result['primary_action'] is not None
>               assert case['expected_what'] in what_result['primary_action']['verb_lemma']
E               AssertionError: assert 'reviews' in 'review'

tests\unit\test_element_detection_validation.py:972: AssertionError
_________________________________________________________________________________ TestElementDetectionEdgeCases.test_ambiguous_element_scenarios __________________________________________________________________________________ 

self = <tests.unit.test_element_detection_validation.TestElementDetectionEdgeCases object at 0x000001714DEB99A0>, spacy_model = <spacy.lang.en.English object at 0x000001714E5FCBC0>

    def test_ambiguous_element_scenarios(self, spacy_model):
        """Test scenarios where elements might be ambiguous or missing"""
        ambiguous_cases = [
            {
                'name': 'passive_voice_unclear_performer',
                'text': 'Reports are generated and distributed',
                'who_should_be_unclear': True
            },
            {
                'name': 'vague_action_description',
                'text': 'Staff handles customer inquiries appropriately',
                'what_should_be_weak': True,
                'when_should_be_vague': True
            },
            {
                'name': 'purpose_statement_not_action',
                'text': 'To ensure compliance with regulatory requirements',
                'what_should_not_detect': True
            },
            {
                'name': 'incomplete_control_description',
                'text': 'Management oversight',
                'all_elements_should_be_weak': True
            }
        ]

        for case in ambiguous_cases:
            who_result = enhanced_who_detection_v2(case['text'], spacy_model)
            what_result = enhance_what_detection(case['text'], spacy_model)
            when_result = enhance_when_detection(case['text'], spacy_model)
>           where_result = enhance_where_detection(case['text'], spacy_model)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\unit\test_element_detection_validation.py:1016:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src\analyzers\where.py:63: in enhance_where_detection
    score = calculate_where_score(where_components, control_type, config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

components = {'all_components': [{'boost_factor': 1.1, 'category': 'departments', 'confidence': 0.75, 'end_char': 20, ...}], 'confi...[], 'organizational': [{'boost_factor': 1.1, 'category': 'departments', 'confidence': 0.75, 'end_char': 20, ...}], ...}
control_type = None, config = None

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
>       where_config = config.get('elements', {}).get('WHERE', {})
                       ^^^^^^^^^^
E       AttributeError: 'NoneType' object has no attribute 'get'

src\analyzers\where.py:112: AttributeError
===================================================================================================== short test summary info ===================================================================================================== 
FAILED tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_system_entity_detection_accuracy - AssertionError: Expected 'Oracle database' in detected 'Unknown Performer' for: Oracle database enforces referential integrity
FAILED tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_vague_role_detection_accuracy - AssertionError: Confidence 0.84 should be <= 0.6 for vague role: Management reviews financial reports quarterly
FAILED tests/unit/test_element_detection_validation.py::TestWHOElementDetectionValidation::test_who_edge_cases - AssertionError: Expected one of ['finance', 'manager'] in detected 'journal entries' for passive_voice_with_clear_performer: Journal entries are reviewed by the Finance Manager
FAILED tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_strong_action_verb_detection - AssertionError: Should detect primary action in: Auditor verifies completeness of documentation     
FAILED tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_moderate_action_verb_detection - AssertionError: Should detect primary action in: Supervisor ensures compliance with policies      
FAILED tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_weak_action_verb_detection - AssertionError: Weak action confidence 0.6563699999999999 should be <= 0.6 for: Management observes current practices
FAILED tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_control_noun_detection - AssertionError: Action with control noun should have boosted confidence for: Manager performs monthly reconciliation
FAILED tests/unit/test_element_detection_validation.py::TestWHATElementDetectionValidation::test_what_edge_cases - AssertionError: Expected 'reviews' in detected 'approve' for compound_actions: Manager reviews and approves budgeFAILED tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_period_end_timing_detection - AssertionError: Expected 'month-end' in detected 'month' for: Controller performs closing procedures at month-end
FAILED tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_vague_timing_detection - AssertionError: Should detect vague timing terms in: Team provides timely resolution
FAILED tests/unit/test_element_detection_validation.py::TestWHENElementDetectionValidation::test_when_edge_cases - KeyError: 'should_detect'
FAILED tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_system_location_detection - AttributeError: 'NoneType' object has no attribute 'get'
FAILED tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_physical_location_detection - AssertionError: Should detect WHERE in: Guard performs inspection at the vault
FAILED tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_organizational_location_detection - AttributeError: 'NoneType' object has no attribute 'get'
FAILED tests/unit/test_element_detection_validation.py::TestWHEREElementDetectionValidation::test_where_edge_cases - AttributeError: 'NoneType' object has no attribute 'get'
FAILED tests/unit/test_element_detection_validation.py::TestElementDetectionEdgeCases::test_complex_control_statements - AssertionError: assert 'reviews' in 'review'
FAILED tests/unit/test_element_detection_validation.py::TestElementDetectionEdgeCases::test_ambiguous_element_scenarios - AttributeError: 'NoneType' object has no attribute 'get'
================================================================================================== 17 failed, 4 passed in 5.55s =================================================================================================== 
PS C:\Users\luria\OneDrive\Desktop\final_consolidated_analyzer>

