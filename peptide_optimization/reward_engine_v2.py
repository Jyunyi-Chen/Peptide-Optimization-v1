"""Reward engine compatible with design_rules_v2_1.py.

Main changes from prior reward engine:
- consumes selectivity_proxy_score and aggregation_control_score from soft rules
- normalizes raw selectivity_index before using it
- exposes pH-aware diagnostic features from design rules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import peptide_optimization.design_rules_v2_1 as dr

@dataclass
class RewardConfig:
    hard_fail_reward: float = -5.0
    use_hard_filter_gate: bool = True

    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "length_score": 0.30,
        "net_charge_score": 0.50,
        "hydrophobicity_score": 0.45,
        "basic_fraction_score": 0.30,
        "arg_fraction_score": 0.15,
        "trp_count_score": 0.15,
        "aggregation_control_score": 0.25,
        "selectivity_proxy_score": 0.30,
    })

    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "amp_activity_score": 2.0,
        "hemolysis_score": -1.5,
        "serum_stability_score": 0.8,
        "protease_stability_score": 0.8,
        "amphipathicity_score": 0.6,
        "novelty_score": 0.4,
        "aggregation_risk": -0.7,
        "synthesis_penalty": -0.5,
        "selectivity_index": 1.0,
    })

    handcrafted_weights: Dict[str, float] = field(default_factory=lambda: {
        "consecutive_hydrophobic_penalty": -0.25,
        "identical_run_penalty": -0.20,
        "proline_in_helical_mode_penalty": -0.20,
        "excess_trp_penalty": -0.25,
        "amidation_bonus": 0.10,
    })

    mode: str = "helical"
    pH: float = 7.4

class AMPRewardEngineV2:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = RewardConfig() if config is None else config

    def evaluate(
        self,
        seq: str,
        c_terminal: str = "COOH",
        model_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        dr.validate_sequence(seq)
        model_scores = dict(model_scores or {})

        passed, hard_details = dr.hard_filter_pass(
            seq, c_terminal=c_terminal, pH=self.config.pH
        )
        soft_features = dr.soft_rule_features(
            seq, c_terminal=c_terminal, pH=self.config.pH
        )
        handcrafted = self._handcrafted_terms(seq, c_terminal, hard_details)

        feature_reward = self._score_feature_terms(soft_features)
        model_reward = self._score_model_terms(model_scores)
        handcrafted_reward = sum(handcrafted.values())
        total_reward = feature_reward + model_reward + handcrafted_reward

        if self.config.use_hard_filter_gate and not passed:
            total_reward += self.config.hard_fail_reward

        return {
            "sequence": seq,
            "c_terminal": c_terminal,
            "pH": self.config.pH,
            "hard_filter_pass": passed,
            "hard_details": hard_details,
            "soft_features": soft_features,
            "feature_reward": round(feature_reward, 6),
            "model_scores": model_scores,
            "model_reward": round(model_reward, 6),
            "handcrafted_terms": {k: round(v, 6) for k, v in handcrafted.items()},
            "handcrafted_reward": round(handcrafted_reward, 6),
            "reward": round(total_reward, 6),
        }

    def _score_feature_terms(self, soft_features: Dict[str, float]) -> float:
        total = 0.0
        for key, weight in self.config.feature_weights.items():
            total += weight * soft_features.get(key, 0.0)
        return total

    def _score_model_terms(self, model_scores: Dict[str, float]) -> float:
        total = 0.0
        for key, weight in self.config.model_weights.items():
            if key == "selectivity_index":
                raw_si = float(model_scores.get("selectivity_index", 0.0))
                total += weight * dr.normalize_selectivity_index(raw_si)
            else:
                total += weight * float(model_scores.get(key, 0.0))
        return total

    def _handcrafted_terms(
        self,
        seq: str,
        c_terminal: str,
        hard_details: Dict[str, object],
    ) -> Dict[str, float]:
        terms: Dict[str, float] = {}

        consecutive_hydro = float(hard_details["max_consecutive_hydrophobic"])
        max_identical = float(hard_details["max_identical_residue_run"])

        terms["consecutive_hydrophobic_penalty"] = (
            self.config.handcrafted_weights["consecutive_hydrophobic_penalty"]
            * max(0.0, consecutive_hydro - 3.0)
        )
        terms["identical_run_penalty"] = (
            self.config.handcrafted_weights["identical_run_penalty"]
            * max(0.0, max_identical - 2.0)
        )

        if self.config.mode == "helical":
            terms["proline_in_helical_mode_penalty"] = (
                self.config.handcrafted_weights["proline_in_helical_mode_penalty"] * seq.count("P")
            )
        else:
            terms["proline_in_helical_mode_penalty"] = 0.0

        excess_trp = max(0, seq.count("W") - dr.SOFT_TARGETS["preferred_trp_count_max"])
        terms["excess_trp_penalty"] = (
            self.config.handcrafted_weights["excess_trp_penalty"] * excess_trp
        )
        terms["amidation_bonus"] = (
            self.config.handcrafted_weights["amidation_bonus"] if c_terminal == "CONH2" else 0.0
        )

        return terms

if __name__ == "__main__":

    engine = AMPRewardEngineV2()
    
    result = engine.evaluate(
        seq="KWLKLLKKWLKLLKK",
        c_terminal="CONH2",
        model_scores={
            "amp_activity_score": 0.82,
            "hemolysis_score": 0.18,
            "serum_stability_score": 0.60,
            "protease_stability_score": 0.55,
            "amphipathicity_score": 0.70,
            "novelty_score": 0.65,
            "aggregation_risk": 0.20,
            "synthesis_penalty": 0.10,
            "selectivity_index": 12.0,
        },
    )
    print(result)
