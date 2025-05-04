#!/usr/bin/env python3.12
# Control Description Analyzer
# Integrates specialized detection modules for each control element

import argparse
import os
import sys
import re
import pandas as pd
import numpy as np
import spacy
import pickle
import gc
import time
from datetime import datetime
from spacy.matcher import PhraseMatcher
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from config_manager import ConfigManager

# Import enhanced detection modules
from enhanced_who import enhanced_who_detection_v2
from enhanced_what import enhance_what_detection, mark_possible_standalone_controls
from enhanced_when import enhance_when_detection
from enhanced_why import enhance_why_detection
from enhanced_escalation import enhance_escalation_detection


class ControlElement:
    """Class representing a single control element with keywords and scoring logic"""

    def __init__(self, name, weight, keywords=None):
        self.name = name
        self.weight = weight
        self.keywords = keywords or []
        self.score = 0
        self.normalized_score = 0
        self.matched_keywords = []
        self.phrases = []  # For spaCy PhraseMatcher
        self.context_relevance = 0.0  # Measure of how relevant the matches are in context
        self.enhanced_results = {}  # Store results from enhanced detection
        self.matcher = None

    def setup_matchers(self, nlp):
        """Set up phrase matchers for this element's keywords"""
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        # Add single-word and multi-word phrases
        self.phrases = [nlp(keyword) for keyword in self.keywords]
        if self.phrases:
            self.matcher.add(f"{self.name}_patterns", self.phrases)

    def analyze(self, text, nlp, enhanced_mode=True, **context):
        """
        Analyze text for presence of this element's keywords

        Args:
            text: The text to analyze
            nlp: spaCy NLP model
            enhanced_mode: Whether to use enhanced detection modules
            **context: Additional context like control_type, frequency, etc.
        """
        if not text or text.strip() == '':
            return 0

        # Use enhanced detection if available
        if enhanced_mode:
            if self.name == "WHO":
                control_type = context.get("control_type")
                frequency = context.get("frequency")
                self.enhanced_results = enhanced_who_detection_v2(text, nlp, control_type, frequency, self.keywords)
                self.score = self.enhanced_results.get("confidence", 0) if self.enhanced_results else 0
                self.normalized_score = self.score * 100
                self.matched_keywords = [
                    self.enhanced_results.get("primary", {}).get("text",
                                                                 "")] if self.enhanced_results and self.enhanced_results.get(
                    "primary") else []

            elif self.name == "WHAT":
                self.enhanced_results = enhance_what_detection(text, nlp, self.keywords)
                self.score = self.enhanced_results.get("score", 0) if self.enhanced_results else 0
                self.normalized_score = self.score * 100
                if self.enhanced_results and self.enhanced_results.get("primary_action"):
                    self.matched_keywords = [self.enhanced_results["primary_action"]["full_phrase"]]
                else:
                    self.matched_keywords = [a["full_phrase"] for a in self.enhanced_results.get("actions", [])[
                                                                       :3]] if self.enhanced_results else []

            elif self.name == "WHEN":
                self.enhanced_results = enhance_when_detection(
                    text,
                    nlp,
                    context.get("control_type"),
                    self.keywords,
                    context.get("frequency")
                )
                self.score = self.enhanced_results.get("score", 0) if self.enhanced_results else 0
                self.normalized_score = self.score * 100
                self.matched_keywords = self.enhanced_results.get("extracted_keywords",
                                                                  []) if self.enhanced_results else []

            elif self.name == "WHY":
                risk_description = context.get("risk_description")
                self.enhanced_results = enhance_why_detection(text, nlp, risk_description, self.keywords)
                self.score = self.enhanced_results.get("score", 0) if self.enhanced_results else 0
                self.normalized_score = self.score * 100
                if self.enhanced_results:
                    top = self.enhanced_results.get("top_match", {})
                    if isinstance(top, dict):
                        if "implied_purpose" in top:
                            self.matched_keywords = [top["implied_purpose"]]
                        elif "text" in top:
                            self.matched_keywords = [top["text"]]
                        else:
                            self.matched_keywords = self.enhanced_results.get("extracted_keywords", [])
                    else:
                        self.matched_keywords = self.enhanced_results.get("extracted_keywords", [])
                else:
                    self.matched_keywords = []

            elif self.name == "ESCALATION":
                self.enhanced_results = enhance_escalation_detection(text, nlp, self.keywords)
                self.score = self.enhanced_results.get("score", 0) if self.enhanced_results else 0
                self.normalized_score = self.score * 100
                self.matched_keywords = [p["text"] for p in
                                         self.enhanced_results.get("phrases", [])] if self.enhanced_results else []

            else:
                # Fall back to base implementation for other elements
                return self._base_analyze(text, nlp)

            # Return weighted score
            return self.score * self.weight
        else:
            # Use base implementation
            return self._base_analyze(text, nlp)
    def _base_analyze(self, text, nlp):
        """Original analysis method as fallback"""
        # Create spaCy doc
        doc = nlp(text.lower())

        # Find matches using PhraseMatcher
        matches = self.matcher(doc)

        # Extract matched keywords
        matched_spans = []
        for match_id, start, end in matches:
            span = doc[start:end]
            matched_spans.append((span, span.text))

        # Get unique matched keywords
        self.matched_keywords = list(set(match[1] for match in matched_spans))

        # Calculate context relevance
        if matched_spans:
            # Assess the syntactic importance of matched words
            relevance_scores = []
            for span, _ in matched_spans:
                # Check if span is part of a subject or verb phrase
                for token in span:
                    if token.dep_ in ("nsubj", "dobj", "ROOT", "VERB"):
                        relevance_scores.append(1.0)  # High relevance
                    else:
                        relevance_scores.append(0.7)  # Medium relevance

            self.context_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        else:
            self.context_relevance = 0.0

        # Calculate raw score (0 to 1, based on matches and context)
        self.score = min(1.0, len(self.matched_keywords) * 0.25) if self.matched_keywords else 0

        # Apply context relevance
        self.score = self.score * max(0.7, self.context_relevance)

        # Return weighted score
        self.normalized_score = self.score * 100
        return self.score * self.weight


    def get_enhancement_feedback(self):
        """Get feedback from enhanced detection if available"""
        if not self.enhanced_results:
            return None

        if self.name == "WHO":
            return self.enhanced_results.get("message")
        elif self.name == "WHAT":
            return self.enhanced_results.get("suggestions", [])
        elif self.name == "WHEN":
            return self.enhanced_results.get("improvement_suggestions", [])
        elif self.name == "WHY":
            return self.enhanced_results.get("feedback")
        elif self.name == "ESCALATION":
            return self.enhanced_results.get("suggestions", [])
        else:
            return None


def enhanced_semantic_similarity(text, element_keywords, nlp):
    """
    Enhanced semantic similarity detection that better captures contextual meaning
    """
    if not text or text.strip() == '':
        return [], 0.0

    # Process the text
    doc = nlp(text.lower())

    # Create a list to store similar terms found
    similar_terms = []
    similarity_scores = []

    # Analyze at chunk level for better context
    for chunk in doc.noun_chunks:
        # Skip very short chunks
        if len(chunk) < 2:
            continue

        # Get vector for this chunk
        if not chunk.has_vector:
            continue

        # Check similarity against all keywords and keyword phrases
        for keyword in element_keywords:
            # Skip if exact match (this is already handled by exact matcher)
            if keyword.lower() in chunk.text.lower():
                continue

            # Get keyword vector - handle phrases properly
            keyword_doc = nlp(keyword.lower())

            # Skip if no vector
            if not keyword_doc.has_vector:
                continue

            # Calculate similarity
            similarity = chunk.similarity(keyword_doc)

            # Only consider moderately high similarities to avoid false positives
            if similarity > 0.65:
                similar_terms.append((chunk.text, keyword, similarity))
                similarity_scores.append(similarity)

    # Also analyze at sentence level for more context
    for sent in doc.sents:
        if not sent.has_vector:
            continue

        # Compare each sentence with concatenated keywords to detect thematic similarity
        keyword_text = " ".join(element_keywords[:10])  # Use first 10 keywords to avoid too long text
        keyword_doc = nlp(keyword_text)

        if keyword_doc.has_vector:
            sent_similarity = sent.similarity(keyword_doc)

            if sent_similarity > 0.7:  # Higher threshold for sentence-level to avoid false positives
                similar_terms.append((sent.text, "element theme", sent_similarity))
                similarity_scores.append(sent_similarity * 0.8)  # Slight discount for full sentences

    # Sort by similarity score
    similar_terms.sort(key=lambda x: x[2], reverse=True)

    # Calculate average similarity score
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

    return similar_terms[:5], avg_similarity  # Limit to top 5 to avoid noise


def boost_term_by_context(term, doc):
    """Boost similarity scores based on the term's role in the sentence"""
    boost = 1.0

    # Find the term in the document
    term_tokens = []
    for token in doc:
        if token.text.lower() in term.lower():
            term_tokens.append(token)

    if not term_tokens:
        return boost

    # Apply boosts based on dependency relations
    for token in term_tokens:
        # Subject terms are more important
        if token.dep_ in ("nsubj", "nsubjpass"):
            boost *= 1.3

        # Direct objects of main verbs are important
        elif token.dep_ == "dobj" and token.head.dep_ == "ROOT":
            boost *= 1.2

        # Terms that are direct children of root verbs
        elif token.head.dep_ == "ROOT":
            boost *= 1.15

        # Terms in prepositional phrases are less important
        elif token.dep_ == "pobj":
            boost *= 0.9

    return min(1.5, boost)  # Cap the boost


def disambiguate_control_term(term, context, nlp):
    """Disambiguate whether a term like 'check' or 'record' is being used in a control-relevant sense"""
    doc = nlp(context)
    term = term.lower()

    # Only support known ambiguous terms
    if term not in {"check", "record"}:
        return True

    # Define disallowed objects that suggest non-control use
    non_control_objects = {
        "check": {"payment", "mark", "number"},
        "record": {"time", "number", "volume"}
    }

    # Define allowed control-related objects (optional, for boosting)
    control_objects = {
        "check": {"accuracy", "completeness", "compliance", "validity"},
        "record": {"results", "evidence", "findings", "control"}
    }

    for token in doc:
        if token.lemma_ == term and token.pos_ in {"VERB", "NOUN"}:
            for child in token.children:
                if child.dep_ in {"dobj", "pobj"}:
                    obj = child.lemma_.lower()

                    if obj in non_control_objects.get(term, set()):
                        return False  # Likely non-control usage
                    if obj in control_objects.get(term, set()):
                        return True  # Strong control-related usage

    # Fall back to string-based disambiguation if needed
    context_lower = context.lower()
    for phrase in control_objects.get(term, []):
        if phrase in context_lower:
            return True
    for phrase in non_control_objects.get(term, []):
        if phrase in context_lower:
            return False

    return True  # Default to assuming control-relevance


class EnhancedControlAnalyzer:
    """Enhanced analyzer with specialized detection modules for each element"""

    def __init__(self, config_file=None):
        # Load config manager
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config if self.config_manager else {}

        # Initialize spaCy with model specified in config
        self.nlp = self._initialize_spacy_model()

        # Initialize elements with their weights from config
        self.elements = self._initialize_elements()

        # Set up matchers for each element
        for element in self.elements.values():
            element.setup_matchers(self.nlp)

        # Vague terms that should be avoided - loaded from config
        self.vague_terms = self.config.get('vague_terms', [])
        self.vague_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        vague_phrases = [self.nlp(term) for term in self.vague_terms]
        if vague_phrases:
            self.vague_matcher.add("vague_patterns", vague_phrases)

        # Configure enhanced detection - can be overridden via config
        self.use_enhanced_detection = self.config.get('use_enhanced_detection', True)

        # Get penalty configuration from config
        self.vague_term_penalty = self.config.get('vague_term_penalty', 2)
        self.max_vague_penalty = self.config.get('max_vague_penalty', 10)

        # Get multi-control penalty settings from config
        self.multi_control_penalty = self.config.get('penalties', {}).get('multi_control', {})
        self.points_per_control = self.multi_control_penalty.get('points_per_control', 5)
        self.max_multi_control_penalty = self.multi_control_penalty.get('max_penalty', 10)

        # Get domain clusters from config
        self.domain_clusters = self.config.get('domain_clusters', {})

        # Get thresholds from config
        self.excellent_threshold = self.config.get('category_thresholds', {}).get('excellent', 75)
        self.good_threshold = self.config.get('category_thresholds', {}).get('good', 50)

    def _initialize_spacy_model(self):
        """Initialize spaCy model based on configuration"""
        spacy_config = self.config.get('spacy', {})
        preferred_model = spacy_config.get('preferred_model', 'en_core_web_md')
        fallback_model = spacy_config.get('fallback_model', 'en_core_web_sm')

        try:
            # Try loading the preferred model first
            return spacy.load(preferred_model)
        except OSError:
            try:
                # Fall back to smaller model if preferred isn't available
                nlp = spacy.load(fallback_model)
                print(
                    f"Using {fallback_model} model. For better results, install {preferred_model} with: python -m spacy download {preferred_model}")
                return nlp
            except OSError:
                print(f"Downloading spaCy model {fallback_model}...")
                spacy.cli.download(fallback_model)
                return spacy.load(fallback_model)

    def _initialize_elements(self):
        """Initialize elements with weights and keywords from configuration"""
        elements = {}
        config_elements = self.config.get('elements', {})

        # Default element configuration if not specified in YAML
        default_elements = {
            "WHO": {"weight": 30, "keywords": []},
            "WHEN": {"weight": 20, "keywords": []},
            "WHAT": {"weight": 30, "keywords": []},
            "WHY": {"weight": 10, "keywords": []},
            "ESCALATION": {"weight": 3, "keywords": []}
        }

        # Merge defaults with configuration
        for name, defaults in default_elements.items():
            element_config = config_elements.get(name, {})
            weight = element_config.get('weight', defaults['weight'])
            keywords = element_config.get('keywords', defaults['keywords'])

            elements[name] = ControlElement(name, weight, keywords)

        return elements

    def analyze_control(self, control_id, description, frequency=None, control_type=None, risk_description=None):
        """
        Analyze a single control description and return detailed results
        Enhanced with specialized detection modules for each element

        Args:
            control_id: Unique identifier for the control
            description: The control description text
            frequency: Optional declared frequency for validation
            control_type: Optional control type for context
            risk_description: Optional risk description for WHY alignment
        """
        if not description or pd.isna(description):
            return {
                "control_id": control_id,
                "description": "",
                "total_score": 0,
                "category": "Needs Improvement",
                "missing_elements": list(self.elements.keys()),
                "vague_terms_found": [],
                "weighted_scores": {element: 0 for element in self.elements},
                "matched_keywords": {element: [] for element in self.elements},
                "enhancement_feedback": {element: None for element in self.elements},
                "validation_results": {
                    "frequency_valid": False,
                    "frequency_message": "No description provided",
                    "control_type_valid": False,
                    "control_type_message": "No description provided"
                },
                "multi_control_indicators": {"detected": False, "count": 0, "candidates": []}
            }

        # Reset all elements for new analysis
        for element in self.elements.values():
            element.score = 0
            element.matched_keywords = []
            element.enhanced_results = {}

        # Check for multi-control description
        multi_control_indicators = self.detect_multi_control(description)

        # Create context for element analysis
        context = {
            "control_type": control_type,
            "frequency": frequency,
            "risk_description": risk_description
        }

        # Analyze for each element with specialized detection
        normalized_scores = {}
        weighted_scores = {}
        matched_keywords = {}
        enhancement_feedback = {}

        for name, element in self.elements.items():
            element.analyze(
                description,
                self.nlp,
                self.use_enhanced_detection,
                **context
            )
            normalized_scores[name] = element.normalized_score  # For reporting
            weight = element.weight
            weighted_scores[name] = (element.normalized_score * weight / 100)  # Contribution to total
            matched_keywords[name] = element.matched_keywords
            enhancement_feedback[name] = element.get_enhancement_feedback()

        # Check for vague terms with PhraseMatcher
        doc = self.nlp(description)
        vague_matches = self.vague_matcher(doc)
        vague_terms_found = []

        for match_id, start, end in vague_matches:
            span = doc[start:end]
            vague_terms_found.append(span.text)

        # Remove duplicates
        vague_terms_found = list(set(vague_terms_found))

        # Calculate vague terms penalty based on config
        vague_penalty = min(len(vague_terms_found) * self.vague_term_penalty, self.max_vague_penalty)

        # Calculate total score
        total_score = sum(weighted_scores.values()) - vague_penalty
        total_score = max(0, total_score)  # Ensure score is not negative

        # Apply multi-control penalty if detected, using config values
        if multi_control_indicators["detected"]:
            multi_control_penalty = min(
                self.max_multi_control_penalty,
                multi_control_indicators["count"] * self.points_per_control
            )
            total_score = max(0, total_score - multi_control_penalty)

        # Determine category based on config thresholds
        if total_score >= self.excellent_threshold:
            category = "Excellent"
        elif total_score >= self.good_threshold:
            category = "Good"
        else:
            category = "Needs Improvement"

        # Define threshold by element type
        element_thresholds = {
            "WHO": 5.0,
            "WHEN": 5.0,
            "WHAT": 5.0,
            "WHY": 2.0,  # Lower threshold for WHY since it typically scores lower
            "ESCALATION": 1.0  # Lowest threshold as these are often bonus elements
        }

        # Identify missing elements using element-specific thresholds
        missing_elements = [name for name, score in weighted_scores.items()
                            if score < element_thresholds.get(name, 5.0)]

        # Add validation results
        validation_results = {}

        # Use enhanced WHEN validation results if available
        when_element = self.elements["WHEN"]
        if self.use_enhanced_detection and "validation" in when_element.enhanced_results:
            validation_results["frequency_valid"] = when_element.enhanced_results["validation"]["is_valid"]
            validation_results["frequency_message"] = when_element.enhanced_results["validation"]["message"]
        else:
            # Fallback to basic validation
            if frequency:
                frequency_valid, frequency_message = self.validate_frequency(description, frequency)
                validation_results["frequency_valid"] = frequency_valid
                validation_results["frequency_message"] = frequency_message
            else:
                validation_results["frequency_valid"] = False
                validation_results["frequency_message"] = "No frequency value provided"

        # Control type validation
        if control_type:
            control_type_valid, control_type_message = self.validate_control_type(description, control_type)
            validation_results["control_type_valid"] = control_type_valid
            validation_results["control_type_message"] = control_type_message
        else:
            validation_results["control_type_valid"] = False
            validation_results["control_type_message"] = "No control type value provided"

        return {
            "control_id": control_id,
            "description": description,
            "total_score": total_score,
            "category": category,
            "missing_elements": missing_elements,
            "vague_terms_found": vague_terms_found,
            "weighted_scores": weighted_scores,
            "normalized_scores": normalized_scores,
            "matched_keywords": matched_keywords,
            "enhancement_feedback": enhancement_feedback,
            "validation_results": validation_results,
            "multi_control_indicators": multi_control_indicators
        }

    def detect_multi_control(self, description):
        """Detect if a description contains multiple controls"""
        # Use enhanced WHAT detection to identify potential standalone controls
        potential_controls = mark_possible_standalone_controls(description, self.nlp)

        result = {
            "detected": len(potential_controls) > 1,
            "count": len(potential_controls),
            "candidates": potential_controls
        }

        return result

    def validate_frequency(self, description, frequency_value):
        """
        Validate that the frequency mentioned in the description matches the frequency field value
        """
        if not description or not frequency_value:
            return False, "Missing description or frequency value"

        # Get frequency terms from config
        frequency_terms = self.config.get('frequency_terms', {})

        # Default frequencies if not in config
        default_frequency_terms = {
            "daily": ["daily", "each day", "every day"],
            "weekly": ["weekly", "each week", "every week"],
            "monthly": ["monthly", "each month", "every month"],
            "quarterly": ["quarterly", "each quarter", "every quarter"],
            "annually": ["annually", "yearly", "each year", "every year"]
        }

        # Merge with defaults if config doesn't have these frequencies
        for freq, terms in default_frequency_terms.items():
            if freq not in frequency_terms:
                frequency_terms[freq] = terms

        # Normalize frequency value to lowercase
        frequency_value = frequency_value.lower().strip()

        # Get standard terms for this frequency if available
        expected_terms = frequency_terms.get(frequency_value, [frequency_value])

        # Check if any expected terms are in the description
        description_lower = description.lower()
        matches = []

        for term in expected_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, description_lower):
                matches.append(term)

        # If matches found, frequency is consistent
        if matches:
            return True, f"Frequency in description ({', '.join(matches)}) matches expected value ({frequency_value})"
        else:
            return False, f"Frequency in description does not match expected value ({frequency_value})"

    def validate_control_type(self, description, control_type):
        """
        Validate that the control type mentioned in the description matches the control type field value
        """
        if not description or not control_type:
            return False, "Missing description or control type value"

        # Get control type keywords from config
        control_type_keywords = self.config.get('control_type_keywords', {})

        # Default control type keywords if not in config
        default_control_type_keywords = {
            "preventive": ["prevent", "preventive", "preventative", "avoid", "before", "prior to", "stops"],
            "detective": ["detect", "detective", "identify", "discover", "find", "after", "following", "review",
                          "monitor", "reconcile"],
            "corrective": ["correct", "corrective", "remediate", "fix", "resolve", "address", "adjust"],
            "automated": ["system", "automated", "automatic", "software", "program", "script",
                          "job", "batch", "workflow", "algorithm", "systematic"]
        }

        # Merge with defaults if config doesn't have these types
        for type_name, keywords in default_control_type_keywords.items():
            if type_name not in control_type_keywords:
                control_type_keywords[type_name] = keywords

        # Normalize control type
        control_type = control_type.lower().strip()

        # Get relevant keywords for this control type
        expected_keywords = control_type_keywords.get(control_type, [])

        if not expected_keywords:
            return False, f"Unknown control type: {control_type}"

        # Check if any expected keywords are in the description
        description_lower = description.lower()
        matches = []

        for keyword in expected_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, description_lower):
                matches.append(keyword)

        # If matches found, control type is consistent
        if matches:
            return True, f"Control type in description ({', '.join(matches)}) is consistent with {control_type}"
        else:
            return False, f"Control type in description does not indicate {control_type}"

    def analyze_file(self, file_path, id_column, desc_column, freq_column=None, type_column=None, risk_column=None,
                     output_file=None):
        """
        Analyze controls from an Excel file and generate a detailed report
        Enhanced with specialized detection modules and risk alignment

        Args:
            file_path: Path to Excel file containing controls
            id_column: Column containing control IDs
            desc_column: Column containing control descriptions
            freq_column: Optional column containing frequency values
            type_column: Optional column containing control type values
            risk_column: Optional column containing risk descriptions
            output_file: Optional path for output Excel report
        """
        print(f"Reading file: {file_path}")

        try:
            # Read the Excel file
            df = pd.read_excel(file_path, engine='openpyxl')

            # Ensure required columns exist
            if id_column not in df.columns:
                raise ValueError(f"ID column '{id_column}' not found in file")

            if desc_column not in df.columns:
                raise ValueError(f"Description column '{desc_column}' not found in file")

            # Check optional columns
            if freq_column and freq_column not in df.columns:
                print(
                    f"Warning: Frequency column '{freq_column}' not found in file. Frequency validation will be skipped.")
                freq_column = None

            if type_column and type_column not in df.columns:
                print(
                    f"Warning: Control type column '{type_column}' not found in file. Control type validation will be skipped.")
                type_column = None

            if risk_column and risk_column not in df.columns:
                print(
                    f"Warning: Risk description column '{risk_column}' not found in file. Risk alignment will be skipped.")
                risk_column = None

            # Analyze each control
            results = []
            total_controls = len(df)

            print(f"Analyzing {total_controls} controls...")

            for i, (idx, row) in enumerate(df.iterrows()):
                # Show progress
                if i % 100 == 0 or i == total_controls - 1:
                    progress = (i + 1) / total_controls * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{total_controls})")

                control_id = row[id_column]
                description = row[desc_column]

                # Optional metadata
                frequency = row[freq_column] if freq_column and freq_column in row else None
                control_type = row[type_column] if type_column and type_column in row else None
                risk_description = row[risk_column] if risk_column and risk_column in row else None

                # Analyze control
                result = self.analyze_control(control_id, description, frequency, control_type, risk_description)
                results.append(result)

            # Create output file if specified
            if output_file:
                self._generate_enhanced_report(
                    results,
                    output_file,
                    freq_column is not None,
                    type_column is not None,
                    risk_column is not None
                )
                print(f"Analysis complete. Results saved to {output_file}")

            return results

        except Exception as e:
            print(f"Error analyzing file: {e}")
            raise

    def _generate_enhanced_report(self, results, output_file, include_frequency=False,
                                  include_control_type=False, include_risk_alignment=False):
        """
        Generate a detailed Excel report with the enhanced analysis results
        """
        # Create results DataFrame with basic elements
        basic_results = []
        for r in results:
            result_dict = {
                "Control ID": r["control_id"],
                "Description": r["description"],
                "Total Score": r["total_score"],
                "Category": r["category"],
                "Missing Elements": ", ".join(r["missing_elements"]) if r["missing_elements"] else "None",
                "Vague Terms": ", ".join(r["vague_terms_found"]) if r["vague_terms_found"] else "None",
                "WHO Score": r["normalized_scores"]["WHO"],
                "WHEN Score": r["normalized_scores"]["WHEN"],
                "WHAT Score": r["normalized_scores"]["WHAT"],
                "WHY Score": r["normalized_scores"]["WHY"],
                "ESCALATION Score": r["normalized_scores"]["ESCALATION"],
            }

            # Add multi-control indicators
            if r["multi_control_indicators"]["detected"]:
                result_dict["Multiple Controls"] = f"Yes ({r['multi_control_indicators']['count']})"
            else:
                result_dict["Multiple Controls"] = "No"

            # Add validation results if applicable
            if include_frequency:
                result_dict["Frequency Valid"] = "Yes" if r["validation_results"]["frequency_valid"] else "No"
                result_dict["Frequency Message"] = r["validation_results"]["frequency_message"]

            if include_control_type:
                result_dict["Control Type Valid"] = "Yes" if r["validation_results"]["control_type_valid"] else "No"
                result_dict["Control Type Message"] = r["validation_results"]["control_type_message"]

            # Add risk alignment if available
            if include_risk_alignment and "WHY" in r["enhancement_feedback"] and r["enhancement_feedback"]["WHY"]:
                result_dict["Risk Alignment Feedback"] = r["enhancement_feedback"]["WHY"]

            basic_results.append(result_dict)

        df_results = pd.DataFrame(basic_results)

        # Create keyword match DataFrame
        keyword_results = []
        for r in results:
            result_dict = {
                "Control ID": r["control_id"],
                "WHO Keywords": ", ".join(r["matched_keywords"]["WHO"]) if r["matched_keywords"]["WHO"] else "None",
                "WHEN Keywords": ", ".join(r["matched_keywords"]["WHEN"]) if r["matched_keywords"]["WHEN"] else "None",
                "WHAT Keywords": ", ".join(r["matched_keywords"]["WHAT"]) if r["matched_keywords"]["WHAT"] else "None",
                "WHY Keywords": ", ".join(r["matched_keywords"]["WHY"]) if r["matched_keywords"]["WHY"] else "None",
                "ESCALATION Keywords": ", ".join(r["matched_keywords"]["ESCALATION"]) if r["matched_keywords"][
                    "ESCALATION"] else "None"
            }
            keyword_results.append(result_dict)

        df_keywords = pd.DataFrame(keyword_results)

        # Create enhancement feedback DataFrame
        feedback_results = []
        for r in results:
            result_dict = {"Control ID": r["control_id"]}

            # Format each element's feedback
            for element in ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]:
                feedback = r["enhancement_feedback"].get(element)

                if isinstance(feedback, list) and feedback:
                    result_dict[f"{element} Feedback"] = "; ".join(feedback)
                elif isinstance(feedback, str) and feedback:
                    result_dict[f"{element} Feedback"] = feedback
                else:
                    result_dict[f"{element} Feedback"] = "None"

            feedback_results.append(result_dict)

        df_feedback = pd.DataFrame(feedback_results)

        # Calculate summary statistics
        total_controls = len(results)
        excellent_count = sum(1 for r in results if r["category"] == "Excellent")
        good_count = sum(1 for r in results if r["category"] == "Good")
        needs_improvement_count = sum(1 for r in results if r["category"] == "Needs Improvement")

        avg_score = np.mean([r["total_score"] for r in results])

        # Missing elements counts
        missing_elements_counts = {element: 0 for element in self.elements}
        for r in results:
            for element in r["missing_elements"]:
                missing_elements_counts[element] += 1

        # Vague terms frequency
        vague_terms_freq = {}
        for r in results:
            for term in r["vague_terms_found"]:
                vague_terms_freq[term] = vague_terms_freq.get(term, 0) + 1

        # Multi-control statistics
        multi_control_count = sum(1 for r in results if r["multi_control_indicators"]["detected"])

        # Validation statistics if applicable
        freq_validation_stats = None
        control_type_validation_stats = None

        if include_frequency:
            valid_freq_count = sum(1 for r in results if r["validation_results"]["frequency_valid"])
            freq_validation_stats = {
                "Valid": valid_freq_count,
                "Invalid": total_controls - valid_freq_count,
                "Percent Valid": (valid_freq_count / total_controls * 100) if total_controls > 0 else 0
            }

        if include_control_type:
            valid_type_count = sum(1 for r in results if r["validation_results"]["control_type_valid"])
            control_type_validation_stats = {
                "Valid": valid_type_count,
                "Invalid": total_controls - valid_type_count,
                "Percent Valid": (valid_type_count / total_controls * 100) if total_controls > 0 else 0
            }

        # Create workbook
        wb = Workbook()

        # Create Analysis Results sheet
        ws_results = wb.active
        ws_results.title = "Analysis Results"

        # Add data to Results sheet
        for r_idx, row in enumerate(dataframe_to_rows(df_results, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                ws_results.cell(row=r_idx, column=c_idx, value=value)

        # Create Keyword Matches sheet
        ws_keywords = wb.create_sheet(title="Keyword Matches")

        # Add data to Keywords sheet
        for r_idx, row in enumerate(dataframe_to_rows(df_keywords, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                ws_keywords.cell(row=r_idx, column=c_idx, value=value)

        # Create Enhancement Feedback sheet
        ws_feedback = wb.create_sheet(title="Enhancement Feedback")

        # Add data to Feedback sheet
        for r_idx, row in enumerate(dataframe_to_rows(df_feedback, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                ws_feedback.cell(row=r_idx, column=c_idx, value=value)

        # Create Multi-Control Candidates sheet if any detected
        if multi_control_count > 0:
            ws_multi = wb.create_sheet(title="Multi-Control Candidates")

            # Write headers
            headers = ["Control ID", "Potential Control", "Score", "Action"]
            for c_idx, header in enumerate(headers, 1):
                ws_multi.cell(row=1, column=c_idx, value=header)
                ws_multi.cell(row=1, column=c_idx).font = Font(bold=True)
                ws_multi.cell(row=1, column=c_idx).fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7",
                                                                      fill_type="solid")

            # Write data
            r_idx = 2
            for result in results:
                if result["multi_control_indicators"]["detected"]:
                    control_id = result["control_id"]

                    for i, candidate in enumerate(result["multi_control_indicators"]["candidates"]):
                        ws_multi.cell(row=r_idx, column=1, value=f"{control_id}-{i + 1}")
                        ws_multi.cell(row=r_idx, column=2, value=candidate["text"])
                        ws_multi.cell(row=r_idx, column=3, value=f"{candidate['score']:.2f}")
                        ws_multi.cell(row=r_idx, column=4, value=candidate.get("action", "Unknown"))
                        r_idx += 1

            # Set column widths
            ws_multi.column_dimensions['A'].width = 15
            ws_multi.column_dimensions['B'].width = 60
            ws_multi.column_dimensions['C'].width = 10
            ws_multi.column_dimensions['D'].width = 30

        # Create Executive Summary sheet
        ws_summary = wb.create_sheet(title="Executive Summary")

        # Add summary statistics
        summary_data = [
            ["Control Description Analysis - Executive Summary", ""],
            ["", ""],
            ["Total Controls Analyzed", total_controls],
            ["Average Score", f"{avg_score:.1f}"],
            ["", ""],
            ["Category Breakdown", ""],
            [f"Excellent ({self.excellent_threshold}-100)", excellent_count],
            [f"Good ({self.good_threshold}-{self.excellent_threshold - 1})", good_count],
            [f"Needs Improvement (0-{self.good_threshold - 1})", needs_improvement_count],
            ["", ""],
            ["Multi-Control Descriptions", multi_control_count],
            ["", ""],
            ["Missing Elements", ""],
        ]

        # Add missing elements statistics
        for element, count in missing_elements_counts.items():
            percentage = (count / total_controls) * 100 if total_controls > 0 else 0
            summary_data.append([f"Missing {element}", f"{count} ({percentage:.1f}%)"])

        summary_data.append(["", ""])
        summary_data.append(["Top Vague Terms", ""])

        # Add top vague terms
        sorted_vague_terms = sorted(vague_terms_freq.items(), key=lambda x: x[1], reverse=True)
        for term, count in sorted_vague_terms[:10]:  # Top 10 vague terms
            percentage = (count / total_controls) * 100 if total_controls > 0 else 0
            summary_data.append([term, f"{count} ({percentage:.1f}%)"])

        # Add validation statistics if applicable
        if freq_validation_stats:
            summary_data.append(["", ""])
            summary_data.append(["Frequency Validation", ""])
            summary_data.append(["Valid Frequency",
                                 f"{freq_validation_stats['Valid']} ({freq_validation_stats['Percent Valid']:.1f}%)"])
            summary_data.append(["Invalid Frequency",
                                 f"{freq_validation_stats['Invalid']} ({100 - freq_validation_stats['Percent Valid']:.1f}%)"])

        if control_type_validation_stats:
            summary_data.append(["", ""])
            summary_data.append(["Control Type Validation", ""])
            summary_data.append(["Valid Control Type",
                                 f"{control_type_validation_stats['Valid']} ({control_type_validation_stats['Percent Valid']:.1f}%)"])
            summary_data.append(["Invalid Control Type",
                                 f"{control_type_validation_stats['Invalid']} ({100 - control_type_validation_stats['Percent Valid']:.1f}%)"])

        # Write summary data
        for r_idx, row in enumerate(summary_data, 1):
            for c_idx, value in enumerate(row, 1):
                ws_summary.cell(row=r_idx, column=c_idx, value=value)

        # Create Methodology sheet
        ws_method = wb.create_sheet(title="Methodology")

        methodology_text = [
            ["Enhanced Control Description Analysis Methodology", ""],
            ["", ""],
            ["Overview", ""],
            [
                "This analysis evaluates control descriptions based on seven key elements that should be present in a well-written control description:",
                ""],
            ["", ""],
            ["1. WHO performs the control", ""],
            ["2. WHEN the control is performed", ""],
            ["3. WHAT activities are performed", ""],
            ["4. WHY the control exists", ""],
            ["5. ESCALATION procedures", ""],
            ["", ""],
            ["Enhanced NLP Analysis", ""],
            ["The analysis uses advanced Natural Language Processing (NLP) techniques to:", ""],
            ["", ""],
            ["- Detect exact keyword matches and identify semantically similar terms", ""],
            ["- Analyze the context, relevance, and strength of control actions", ""],
            ["- Identify primary versus secondary performers of the control", ""],
            ["- Detect vague timing terms and suggest specific alternatives", ""],
            ["- Evaluate alignment between control purpose and mapped risks", ""],
            ["- Detect when a description contains multiple controls", ""],
            ["", ""],
            ["Scoring Method", ""],
            ["Each element is weighted based on its importance:", ""],
            ["", ""],
            [f"WHO: {self.elements['WHO'].weight}%", ""],
            [f"WHEN: {self.elements['WHEN'].weight}%", ""],
            [f"WHAT: {self.elements['WHAT'].weight}%", ""],
            [f"WHY: {self.elements['WHY'].weight}%", ""],
            [f"ESCALATION: {self.elements['ESCALATION'].weight}%", ""],
            ["", ""],
            ["Penalties are applied for:", ""],
            [f"- Vague terms (up to {self.max_vague_penalty} points)", ""],
            [f"- Multiple controls in one description (up to {self.max_multi_control_penalty} points)", ""],
            ["", ""],
            ["Categories", ""],
            ["Controls are categorized based on their total score:", ""],
            ["", ""],
            [f"Excellent: {self.excellent_threshold}-100", ""],
            [f"Good: {self.good_threshold}-{self.excellent_threshold - 1}", ""],
            [f"Needs Improvement: 0-{self.good_threshold - 1}", ""]
        ]

        # Add validation methodology if applicable
        if include_frequency or include_control_type or include_risk_alignment:
            methodology_text.append(["", ""])
            methodology_text.append(["Enhanced Validation Checks", ""])

        if include_frequency:
            methodology_text.append(["", ""])
            methodology_text.append(["Frequency Validation:", ""])
            methodology_text.append(
                ["Advanced detection of timing patterns with specific and vague term identification.", ""])
            methodology_text.append(["Validation against frequency metadata with context-aware scoring.", ""])

        if include_control_type:
            methodology_text.append(["", ""])
            methodology_text.append(["Control Type Validation:", ""])
            methodology_text.append(["Validates consistency between control actions and declared control type.", ""])
            methodology_text.append(
                ["Provides context-specific feedback for preventive, detective, corrective, and automated controls.",
                 ""])

        if include_risk_alignment:
            methodology_text.append(["", ""])
            methodology_text.append(["Risk Alignment:", ""])
            methodology_text.append(["Evaluates how well the control's purpose aligns with the mapped risk.", ""])
            methodology_text.append(["Identifies both explicit and implicit WHY statements and their strength.", ""])

        # Write methodology data
        for r_idx, row in enumerate(methodology_text, 1):
            for c_idx, value in enumerate(row, 1):
                ws_method.cell(row=r_idx, column=c_idx, value=value)

        # Create Example Controls sheet
        ws_examples = wb.create_sheet(title="Example Controls")

        examples_text = [
            ["Example Controls", ""],
            ["", ""],
            ["Excellent Control Example", ""],
            [
                "The Accounting Manager reviews the monthly reconciliation between the subledger and general ledger by the 5th business day of the following month. The reviewer examines supporting documentation, verifies that all reconciling items have been properly identified and resolved, and ensures compliance with accounting policies. The review is evidenced by electronic sign-off in the financial system. Any discrepancies exceeding $10,000 are escalated to the Controller and documented in the issue tracking system. The reconciliation and review documentation are stored in the Finance SharePoint site and retained according to the document retention policy.",
                ""],
            ["", ""],
            ["Good Control Example", ""],
            [
                "The Accounting Supervisor reviews the monthly journal entries prior to posting to ensure accuracy and completeness. The reviewer checks supporting documentation and approves entries by signing the journal entry form. Any errors are returned to the preparer for correction.",
                ""],
            ["", ""],
            ["Needs Improvement Control Example", ""],
            ["Management reviews financial statements periodically and addresses any issues as appropriate.", ""],
            ["", ""],
            ["Key Improvement Recommendations", ""],
            ["1. Specify WHO is responsible (exact role, not just 'management')", ""],
            ["2. Define WHEN the control occurs (specific timing, not just 'periodically')", ""],
            ["3. Detail WHAT actions are taken (specific verifications, not just 'reviews')", ""],
            ["4. Explain WHY the control exists (what risk it addresses)", ""],
            ["5. Include ESCALATION procedures for exceptions", ""],
            ["6. Avoid vague terms like 'appropriate', 'timely', 'periodically'", ""],
            ["7. Separate multiple controls into individual control descriptions", ""]
        ]

        # Write examples data
        for r_idx, row in enumerate(examples_text, 1):
            for c_idx, value in enumerate(row, 1):
                ws_examples.cell(row=r_idx, column=c_idx, value=value)

        # Apply formatting to all sheets
        for ws in [ws_results, ws_keywords, ws_feedback, ws_summary, ws_method, ws_examples]:
            # Format headers
            if ws in [ws_results, ws_keywords, ws_feedback]:
                header_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
                header_font = Font(bold=True)

                for col in range(1, ws.max_column + 1):
                    cell = ws.cell(row=1, column=col)
                    cell.fill = header_fill
                    cell.font = header_font

            # Set column widths
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    if cell.value:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except (TypeError, ValueError):
                            pass

                adjusted_width = (max_length + 2) * 1.1
                ws.column_dimensions[column_letter].width = min(adjusted_width, 50)

        # Additional formatting for summary sheet
        ws_summary.column_dimensions['A'].width = 35
        ws_summary.column_dimensions['B'].width = 20

        title_font = Font(bold=True, size=14)
        section_font = Font(bold=True)

        ws_summary.cell(row=1, column=1).font = title_font

        # Format section headers in summary
        section_rows = [6, 11, 13]
        section_rows.append(section_rows[-1] + len(missing_elements_counts.items()) + 2)  # Vague terms section

        if freq_validation_stats:
            section_rows.append(section_rows[-1] + len(sorted_vague_terms[:10]) + 2)  # Frequency validation section

        if control_type_validation_stats and freq_validation_stats:
            section_rows.append(section_rows[-1] + 3)  # Control type validation section

        for row in section_rows:
            if row <= ws_summary.max_row:
                ws_summary.cell(row=row, column=1).font = section_font

        # Format methodology sheet
        ws_method.column_dimensions['A'].width = 60

        ws_method.cell(row=1, column=1).font = title_font

        method_section_rows = [3, 14, 24, 39]

        if include_frequency or include_control_type or include_risk_alignment:
            method_section_rows.append(method_section_rows[-1] + 3)  # Validation section

        for row in method_section_rows:
            if row <= ws_method.max_row:
                ws_method.cell(row=row, column=1).font = section_font

        # Format examples sheet
        ws_examples.column_dimensions['A'].width = 70

        ws_examples.cell(row=1, column=1).font = title_font

        for row in [3, 6, 9, 12]:  # Section headers in examples
            ws_examples.cell(row=row, column=1).font = section_font

        # Save workbook
        wb.save(output_file)


def analyze_file_with_batches(self, file_path, id_column, desc_column, freq_column=None,
                              type_column=None, risk_column=None, output_file=None,
                              batch_size=500, temp_dir=None):
    """
    Analyze controls from an Excel file in batches and generate a detailed report

    Args:
        file_path: Path to Excel file containing controls
        id_column: Column containing control IDs
        desc_column: Column containing control descriptions
        freq_column: Optional column containing frequency values
        type_column: Optional column containing control type values
        risk_column: Optional column containing risk descriptions
        output_file: Optional path for output Excel report
        batch_size: Number of controls to process in each batch
        temp_dir: Directory to store temporary batch results
    """
    print(f"Reading file: {file_path}")

    # Create temp directory if specified
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = "temp_batch_results"
        os.makedirs(temp_dir, exist_ok=True)

    try:
        # Read the Excel file
        df = pd.read_excel(file_path, engine='openpyxl')

        # Ensure required columns exist
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in file")

        if desc_column not in df.columns:
            raise ValueError(f"Description column '{desc_column}' not found in file")

        # Check optional columns
        if freq_column and freq_column not in df.columns:
            print(f"Warning: Frequency column '{freq_column}' not found in file. Frequency validation will be skipped.")
            freq_column = None

        if type_column and type_column not in df.columns:
            print(
                f"Warning: Control type column '{type_column}' not found in file. Control type validation will be skipped.")
            type_column = None

        if risk_column and risk_column not in df.columns:
            print(
                f"Warning: Risk description column '{risk_column}' not found in file. Risk alignment will be skipped.")
            risk_column = None

        # Initialize results and tracking variables
        all_results = []
        total_controls = len(df)
        start_time = time.time()

        print(f"Analyzing {total_controls} controls in batches of {batch_size}...")

        # Process in batches
        for batch_start in range(0, total_controls, batch_size):
            batch_end = min(batch_start + batch_size, total_controls)
            batch_num = batch_start // batch_size + 1
            total_batches = (total_controls + batch_size - 1) // batch_size

            batch_start_time = time.time()
            print(f"\nProcessing batch {batch_num}/{total_batches}: controls {batch_start + 1}-{batch_end}")

            # Get the slice of DataFrame for this batch
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_results = []

            # Process each control in the batch
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                if i % 25 == 0:
                    # Show more granular progress within batch
                    progress = (i + 1) / len(batch_df) * 100
                    print(f"  Batch progress: {progress:.1f}% ({i + 1}/{len(batch_df)})")

                control_id = row[id_column]
                description = row[desc_column]

                # Optional metadata
                frequency = row[freq_column] if freq_column and freq_column in row else None
                control_type = row[type_column] if type_column and type_column in row else None
                risk_description = row[risk_column] if risk_column and risk_column in row else None

                try:
                    # Analyze control
                    result = self.analyze_control(control_id, description, frequency, control_type, risk_description)
                    batch_results.append(result)
                except Exception as e:
                    # Log error but continue processing
                    error_msg = f"Error analyzing control {control_id}: {str(e)}"
                    print(error_msg)
                    # Add minimal error entry to maintain indexing
                    batch_results.append({
                        "control_id": control_id,
                        "description": description,
                        "total_score": 0,
                        "category": "Error",
                        "missing_elements": [],
                        "error_message": str(e)
                    })

            # Save batch results to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_filename = os.path.join(temp_dir, f"batch_{batch_num:04d}_{timestamp}.pkl")
            try:
                with open(batch_filename, 'wb') as f:
                    pickle.dump(batch_results, f, protocol=4)  # Use protocol 4 for compatibility
            except Exception as e:
                print(f"Warning: Could not save batch file ({e}). Continuing analysis...")

            # Add to overall results
            all_results.extend(batch_results)

            # Calculate batch statistics
            batch_time = time.time() - batch_start_time
            controls_per_second = len(batch_df) / batch_time if batch_time > 0 else 0

            # Estimate remaining time
            completed = batch_end
            remaining = total_controls - completed
            estimated_time = remaining / controls_per_second if controls_per_second > 0 else 0

            # Report batch completion
            print(f"  Batch {batch_num}/{total_batches} completed in {batch_time:.1f} seconds")
            print(f"  Processing speed: {controls_per_second:.2f} controls/second")
            print(f"  Progress: {completed}/{total_controls} controls ({completed / total_controls * 100:.1f}%)")

            hours, remainder = divmod(estimated_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")

            # Save checkpoint of all results so far
            checkpoint_file = os.path.join(temp_dir, "all_results_checkpoint.pkl")
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(all_results, f, protocol=4)
            except Exception as e:
                print(f"Warning: Could not save checkpoint ({e}). Continuing analysis...")

            # Clear memory
            gc.collect()

            # Optional: Clear the NLP model's cache to free memory
            try:
                # Get all pipelines and clear their caches
                for pipe_name in self.nlp.pipe_names:
                    pipe = self.nlp.get_pipe(pipe_name)
                    if hasattr(pipe, "vocab") and hasattr(pipe.vocab, "strings"):
                        pipe.vocab.strings.clean_up()
            except Exception:
                # If cleanup fails, just continue
                pass

        # Create output file if specified
        if output_file:
            try:
                self._generate_enhanced_report(
                    all_results,
                    output_file,
                    freq_column is not None,
                    type_column is not None,
                    risk_column is not None
                )
                print(f"\nAnalysis complete. Results saved to {output_file}")
            except Exception as e:
                print(f"Error generating final report: {e}")
                # Try to save results in pickle format
                emergency_output = os.path.splitext(output_file)[0] + "_emergency.pkl"
                with open(emergency_output, 'wb') as f:
                    pickle.dump(all_results, f, protocol=4)
                print(f"Emergency results saved to {emergency_output}")

        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Average processing speed: {total_controls / total_time:.2f} controls/second")

        return all_results

    except Exception as e:
        print(f"Error analyzing file: {e}")
        raise
    finally:
        # Save any results we have even if an error occurred
        if 'all_results' in locals() and len(all_results) > 0:
            emergency_file = os.path.join(temp_dir, f"emergency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            try:
                with open(emergency_file, 'wb') as f:
                    pickle.dump(all_results, f, protocol=4)
                print(f"Emergency results saved to {emergency_file}")
            except Exception as e:
                print(f"Failed to save emergency results: {e}")


def resume_from_checkpoint(checkpoint_file, analyzer, args):
    """Resume analysis from a saved checkpoint"""
    import pickle
    import pandas as pd
    import io
    import os

    print(f"Resuming from checkpoint: {checkpoint_file}")

    try:
        # Load checkpoint data
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except (TypeError, io.UnsupportedOperation) as e:
            print(f"Error opening checkpoint file: {e}")
            # Try with absolute path
            abs_path = os.path.abspath(checkpoint_file)
            print(f"Trying with absolute path: {abs_path}")
            with open(abs_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

        # Extract completed results
        if isinstance(checkpoint_data, list):
            # Simple checkpoint with just results
            completed_results = checkpoint_data
            last_control_id = completed_results[-1]["control_id"] if completed_results else None
        elif isinstance(checkpoint_data, dict):
            # Advanced checkpoint with metadata
            completed_results = checkpoint_data.get("results", [])
            last_control_id = checkpoint_data.get("last_control_id")

        print(f"Loaded {len(completed_results)} completed controls from checkpoint")

        # Read the Excel file to get remaining controls
        df = pd.read_excel(args.file, engine='openpyxl')

        # Find the index of the last processed control
        last_index = -1
        if last_control_id:
            for i, row in df.iterrows():
                if str(row[args.id_column]) == str(last_control_id):
                    last_index = i
                    break

        if last_index == -1:
            print("Warning: Could not find last processed control. Starting from beginning.")
            last_index = -1

        # Create DataFrame with remaining controls
        remaining_df = df.iloc[last_index + 1:].copy()

        if len(remaining_df) == 0:
            print("No remaining controls to process. Analysis was already complete.")

            # Generate the final report
            if args.output_file:
                analyzer._generate_enhanced_report(
                    completed_results,
                    args.output_file,
                    args.freq_column is not None,
                    args.type_column is not None,
                    args.risk_column is not None
                )
                print(f"Final report generated at {args.output_file}")

            return completed_results, 0

        print(f"Resuming analysis with {len(remaining_df)} remaining controls")

        # Create a temporary file with the remaining controls
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name

        remaining_df.to_excel(temp_path, engine='openpyxl', index=False)

        # Process the remaining controls
        temp_output = os.path.splitext(args.output_file)[0] + "_remaining.xlsx"

        results = analyzer.analyze_file_with_batches(
            temp_path,
            args.id_column,
            args.desc_column,
            args.freq_column,
            args.type_column,
            args.risk_column,
            temp_output,
            args.batch_size,
            args.temp_dir
        )

        # Combine results
        all_results = completed_results + results

        # Generate the final report
        if args.output_file:
            analyzer._generate_enhanced_report(
                all_results,
                args.output_file,
                args.freq_column is not None,
                args.type_column is not None,
                args.risk_column is not None
            )
            print(f"Final report generated at {args.output_file}")

        # Delete the temporary file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")

        return all_results, 0

    except Exception as e:
        print(f"Error resuming from checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, 1

def main():
    """Command-line interface for the Enhanced Control Description Analyzer"""
    parser = argparse.ArgumentParser(description='Analyze control descriptions with enhanced NLP capabilities.')
    parser.add_argument('file', help='Excel file with control descriptions')
    parser.add_argument('--id-column', default='Control_ID', help='Column containing control IDs (name)')
    parser.add_argument('--desc-column', default='Control_Description',
                        help='Column containing control descriptions (name)')
    parser.add_argument('--freq-column', help='Column containing frequency values for validation (name)')
    parser.add_argument('--type-column', help='Column containing control type values for validation (name)')
    parser.add_argument('--risk-column', help='Column containing risk descriptions for WHY alignment (name)')
    parser.add_argument('--output-file', help='Output Excel file path')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--disable-enhanced-detection', action='store_true', help='Disable enhanced detection modules')

    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.file)[0]
        args.output_file = f"{base_name}_enhanced_analysis.xlsx"

    # Create enhanced analyzer with config file if specified
    analyzer = EnhancedControlAnalyzer(args.config)

    # Toggle enhanced detection if requested
    if args.disable_enhanced_detection:
        analyzer.use_enhanced_detection = False
        print("Enhanced detection modules disabled. Using base analysis only.")

    try:
        analyzer.analyze_file(
            args.file,
            args.id_column,
            args.desc_column,
            args.freq_column,
            args.type_column,
            args.risk_column,
            args.output_file
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())