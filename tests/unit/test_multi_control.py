# test_enhanced_multi_control.py

import unittest
from enhanced_multi_control import (
    detect_multi_control,
    mark_possible_standalone_controls,
    categorize_actions,
    find_timing_for_action,
    find_performer_for_action
)
import spacy


class TestMultiControlDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a small spaCy model for testing
        cls.nlp = spacy.load("en_core_web_sm")

    def test_different_frequencies(self):
        # Create mock data for a control with different frequencies
        who_data = {
            "primary": {"text": "Finance Team"},
            "secondary": []
        }

        what_data = {
            "primary_action": {"full_phrase": "reconcile accounts"},
            "secondary_actions": [{"full_phrase": "review transactions"}]
        }

        when_data = {
            "frequencies": ["monthly", "weekly"],
            "multi_frequency_detected": True,
            "candidates": [
                {"text": "monthly", "frequency": "monthly", "method": "explicit_frequency"},
                {"text": "weekly", "frequency": "weekly", "method": "explicit_frequency"}
            ]
        }

        escalation_data = {"detected": False, "phrases": []}

        text = "The Finance Team reconciles accounts monthly and reviews transactions weekly."

        result = detect_multi_control(text, who_data, what_data, when_data, escalation_data)

        self.assertTrue(result["detected"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["confidence"], "high")

    def test_escalation_vs_control(self):
        # Create mock data for a control with escalation
        who_data = {
            "primary": {"text": "Finance Team"},
            "secondary": [{"text": "Controller"}]
        }

        what_data = {
            "primary_action": {"full_phrase": "reconcile accounts"},
            "secondary_actions": [{"full_phrase": "escalate discrepancies"}]
        }

        when_data = {
            "frequencies": ["monthly"],
            "multi_frequency_detected": False,
            "candidates": [
                {"text": "monthly", "frequency": "monthly", "method": "explicit_frequency"}
            ]
        }

        escalation_data = {"detected": True, "phrases": [{"text": "escalate discrepancies"}]}

        text = "The Finance Team reconciles accounts monthly. If discrepancies are found, they are escalated to the Controller."

        result = detect_multi_control(text, who_data, what_data, when_data, escalation_data)

        # Should NOT detect multiple controls
        self.assertFalse(result["detected"])
        self.assertEqual(result["count"], 1)

    def test_adhoc_component(self):
        # Create mock data for a control with regular and ad-hoc components
        who_data = {
            "primary": {"text": "Finance Team"},
            "secondary": []
        }

        what_data = {
            "primary_action": {"full_phrase": "reconcile accounts"},
            "secondary_actions": [{"full_phrase": "review transactions"}]
        }

        when_data = {
            "frequencies": ["monthly"],
            "multi_frequency_detected": False,
            "candidates": [
                {"text": "monthly", "frequency": "monthly", "method": "explicit_frequency"}
            ]
        }

        escalation_data = {"detected": False, "phrases": []}

        text = "The Finance Team reconciles accounts monthly. They also review transactions as needed when system changes occur."

        result = detect_multi_control(text, who_data, what_data, when_data, escalation_data)

        self.assertTrue(result["detected"])
        self.assertTrue(result["has_adhoc_component"])

    def test_mark_standalone_controls(self):
        text = "The Finance Manager reviews the reconciliation monthly. The Controller approves large transactions weekly."

        result = mark_possible_standalone_controls(text, self.nlp)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(c["score"] > 0.5 for c in result))

    def test_single_control_with_details(self):
        # Create mock data for a single control with detailed steps
        who_data = {
            "primary": {"text": "Finance Team"},
            "secondary": []
        }

        what_data = {
            "primary_action": {"full_phrase": "reconcile accounts"},
            "secondary_actions": [
                {"full_phrase": "compare balances"},
                {"full_phrase": "investigate differences"}
            ]
        }

        when_data = {
            "frequencies": ["monthly"],
            "multi_frequency_detected": False,
            "candidates": [
                {"text": "monthly", "frequency": "monthly", "method": "explicit_frequency"}
            ]
        }

        escalation_data = {"detected": False, "phrases": []}

        text = "The Finance Team reconciles accounts monthly by comparing balances and investigating differences."

        result = detect_multi_control(text, who_data, what_data, when_data, escalation_data)

        # Should NOT detect multiple controls
        self.assertFalse(result["detected"])


if __name__ == '__main__':
    unittest.main()