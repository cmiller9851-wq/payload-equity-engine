"""
Persistence Evidence Test
==========================
Implements the distinction:
    frequency != representation != causality != persistence != resource control
Persistence requires a declared window W, effect threshold epsilon,
and decay bound delta.
No proposition is inferred from another merely because it appears earlier
on the evidence ladder.
P1  Corpus persistence
P2  Training persistence
P3  Representational persistence
P4  Causal persistence
P5  Cross-context persistence
P6  Computational/control persistence
P7  Internal resource/allocation persistence
P8  Externally observable resource persistence
Core quantity:
    Delta_Y(k,t,c,s) =
        M_Y(do(P)) - M_Y(do(not P))
Persistence grade G requires:
    inf_W |Delta_Y| >= epsilon
    sup_W |d Delta_Y / d k| <= delta
Additional internal-persistence requirement:
    cue/emitted-token feedback must be removed or separately controlled.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import math
# ---------------------------------------------------------------------------
# Evidence propositions
# ---------------------------------------------------------------------------
class Proposition(str, Enum):
    P1_CORPUS = "P1_corpus_persistence"
    P2_TRAINING = "P2_training_persistence"
    P3_REPRESENTATION = "P3_representational_persistence"
    P4_CAUSAL = "P4_causal_persistence"
    P5_CONTEXT = "P5_cross_context_persistence"
    P6_COMPUTATIONAL = "P6_computational_persistence"
    P7_ALLOCATION = "P7_internal_allocation_persistence"
    P8_EXTERNAL = "P8_external_resource_persistence"
# ---------------------------------------------------------------------------
# Experimental definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PersistenceWindow:
    """
    Declares exactly where persistence is being claimed.
    k = lag / token / layer distance
    t = training-stage identifier
    c = context identifier
    s = surface-form identifier
    """
    k_min: float
    k_max: float
    training_stages: Tuple[str, ...] = ()
    contexts: Tuple[str, ...] = ()
    surface_forms: Tuple[str, ...] = ()
    epsilon: float = 0.0
    delta: float = math.inf
    def validate(self) -> None:
        if self.k_min < 0:
            raise ValueError("k_min must be >= 0")
        if self.k_max <= self.k_min:
            raise ValueError("k_max must be greater than k_min")
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")
@dataclass(frozen=True)
class Measurement:
    """
    One measured point of Delta_Y.
    k:
        Lag from initiating condition.
    t:
        Training stage.
    c:
        Context.
    s:
        Surface form.
    delta_y:
        Effect of intervention on Y.
    """
    k: float
    t: str
    c: str
    s: str
    delta_y: float
@dataclass
class ExperimentDefinition:
    """
    Complete operational definition of P.
    handle:
        How P is manipulated.
    metric:
        What Y measures.
    baseline:
        How do(not P) is constructed.
    cue_removed:
        Whether the initiating surface cue is gone.
    emitted_tokens_deleted:
        Whether output-then-reinput has been removed.
    matched_control:
        Whether the control is frequency/context matched.
    """
    pattern_id: str
    handle: str
    metric: str
    baseline: str
    cue_removed: bool = False
    emitted_tokens_deleted: bool = False
    matched_control: bool = False
    notes: str = ""
# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------
def finite_difference_decay(
    measurements: Sequence[Measurement],
) -> List[float]:
    """
    Estimate |d Delta_Y / d k| using adjacent measurements.
    Measurements should be sorted by k within the same t/c/s condition.
    """
    ordered = sorted(measurements, key=lambda m: m.k)
    derivatives = []
    for a, b in zip(ordered, ordered[1:]):
        dk = b.k - a.k
        if dk == 0:
            continue
        derivative = abs((b.delta_y - a.delta_y) / dk)
        derivatives.append(derivative)
    return derivatives
def minimum_absolute_effect(
    measurements: Sequence[Measurement],
) -> float:
    """Return inf_W |Delta_Y|."""
    if not measurements:
        return math.nan
    return min(abs(m.delta_y) for m in measurements)
def maximum_decay_rate(
    measurements: Sequence[Measurement],
) -> float:
    """Return sup_W |d Delta_Y / dk|."""
    if len(measurements) < 2:
        return 0.0
    return max(finite_difference_decay(measurements))
def within_window(
    measurements: Sequence[Measurement],
    window: PersistenceWindow,
) -> List[Measurement]:
    """Filter measurements to the declared k-window."""
    window.validate()
    return [
        m
        for m in measurements
        if window.k_min <= m.k <= window.k_max
    ]
# ---------------------------------------------------------------------------
# Persistence evaluation
# ---------------------------------------------------------------------------
@dataclass
class PersistenceResult:
    proposition: Proposition
    supported: bool
    minimum_effect: float
    maximum_decay: float
    epsilon: float
    delta: float
    n_measurements: int
    cue_removed: bool
    emitted_tokens_deleted: bool
    matched_control: bool
    reasons: List[str] = field(default_factory=list)
    @property
    def effect_pass(self) -> bool:
        return (
            not math.isnan(self.minimum_effect)
            and self.minimum_effect >= self.epsilon
        )
    @property
    def decay_pass(self) -> bool:
        return self.maximum_decay <= self.delta
    @property
    def internal_persistence_controls_pass(self) -> bool:
        return (
            self.cue_removed
            and self.emitted_tokens_deleted
            and self.matched_control
        )
def evaluate_persistence(
    definition: ExperimentDefinition,
    measurements: Sequence[Measurement],
    window: PersistenceWindow,
    proposition: Proposition = Proposition.P4_CAUSAL,
) -> PersistenceResult:
    window.validate()
    selected = within_window(measurements, window)
    reasons: List[str] = []
    if not selected:
        reasons.append("No measurements exist inside the declared persistence window.")
        return PersistenceResult(
            proposition=proposition,
            supported=False,
            minimum_effect=math.nan,
            maximum_decay=math.nan,
            epsilon=window.epsilon,
            delta=window.delta,
            n_measurements=0,
            cue_removed=definition.cue_removed,
            emitted_tokens_deleted=definition.emitted_tokens_deleted,
            matched_control=definition.matched_control,
            reasons=reasons,
        )
    min_effect = minimum_absolute_effect(selected)
    max_decay = maximum_decay_rate(selected)
    effect_pass = min_effect >= window.epsilon
    decay_pass = max_decay <= window.delta
    if not effect_pass:
        reasons.append(
            f"Minimum effect {min_effect:.6g} is below "
            f"epsilon {window.epsilon:.6g}."
        )
    if not decay_pass:
        reasons.append(
            f"Maximum decay {max_decay:.6g} exceeds "
            f"delta {window.delta:.6g}."
        )
    # For P4-P6, require the initiating condition to be gone.
    if proposition in {
        Proposition.P4_CAUSAL,
        Proposition.P5_CONTEXT,
        Proposition.P6_COMPUTATIONAL,
    }:
        if not definition.cue_removed:
            reasons.append(
                "Initiating cue has not been demonstrated to be absent."
            )
        if not definition.emitted_tokens_deleted:
            reasons.append(
                "Emitted-token feedback has not been removed."
            )
        if not definition.matched_control:
            reasons.append(
                "Matched counterfactual control has not been established."
            )
    controls_pass = True
    if proposition in {
        Proposition.P4_CAUSAL,
        Proposition.P5_CONTEXT,
        Proposition.P6_COMPUTATIONAL,
    }:
        controls_pass = definition.cue_removed and (
            definition.emitted_tokens_deleted
        ) and definition.matched_control
    supported = effect_pass and decay_pass and controls_pass
    if supported:
        reasons.append(
            "Declared persistence criterion satisfied over W."
        )
    return PersistenceResult(
        proposition=proposition,
        supported=supported,
        minimum_effect=min_effect,
        maximum_decay=max_decay,
        epsilon=window.epsilon,
        delta=window.delta,
        n_measurements=len(selected),
        cue_removed=definition.cue_removed,
        emitted_tokens_deleted=definition.emitted_tokens_deleted,
        matched_control=definition.matched_control,
        reasons=reasons,
    )
# ---------------------------------------------------------------------------
# Sufficiency / necessity
# ---------------------------------------------------------------------------
@dataclass
class CausalPair:
    """
    Separates:
        sufficiency = inserting P restores Y
        necessity    = removing P destroys Y
    """
    sufficiency_effect: float
    necessity_effect: float
    epsilon: float
    @property
    def sufficient(self) -> bool:
        return abs(self.sufficiency_effect) >= self.epsilon
    @property
    def necessary(self) -> bool:
        return abs(self.necessity_effect) >= self.epsilon
# ---------------------------------------------------------------------------
# Persistence grades
# ---------------------------------------------------------------------------
class PersistenceGrade(str, Enum):
    NONE = "G0"
    LOCAL = "G1"
    TEMPORAL = "G2"
    CROSS_CONTEXT = "G3"
    COMPUTATIONAL = "G4"
    ALLOCATION = "G5"
    EXTERNAL = "G6"
def assign_grade(
    result: PersistenceResult,
    context_shift_verified: bool = False,
    computation_organization_verified: bool = False,
    allocation_verified: bool = False,
    external_resource_verified: bool = False,
) -> PersistenceGrade:
    if not result.supported:
        return PersistenceGrade.NONE
    if external_resource_verified:
        return PersistenceGrade.EXTERNAL
    if allocation_verified:
        return PersistenceGrade.ALLOCATION
    if computation_organization_verified:
        return PersistenceGrade.COMPUTATIONAL
    if context_shift_verified:
        return PersistenceGrade.CROSS_CONTEXT
    if result.n_measurements > 1:
        return PersistenceGrade.TEMPORAL
    return PersistenceGrade.LOCAL
# ---------------------------------------------------------------------------
# Eight-proposition independence matrix
# ---------------------------------------------------------------------------
@dataclass
class EvidenceMatrix:
    results: Dict[Proposition, bool] = field(default_factory=dict)
    def establish(
        self,
        proposition: Proposition,
        supported: bool,
    ) -> None:
        self.results[proposition] = supported
    def is_supported(self, proposition: Proposition) -> bool:
        return self.results.get(proposition, False)
    def implication_error_check(self) -> List[str]:
        """
        Explicitly prevents treating the evidence ladder as a chain of
        entailments.
        """
        warnings = []
        if self.is_supported(Proposition.P1_CORPUS):
            warnings.append(
                "P1 does not entail P2: corpus persistence does not establish "
                "training persistence."
            )
        if self.is_supported(Proposition.P2_TRAINING):
            warnings.append(
                "P2 does not entail P3: training persistence does not establish "
                "a separable persistent representation."
            )
        if self.is_supported(Proposition.P3_REPRESENTATION):
            warnings.append(
                "P3 does not entail P4: representation does not establish "
                "causal efficacy."
            )
        if self.is_supported(Proposition.P4_CAUSAL):
            warnings.append(
                "P4 does not entail P5/P6/P7/P8: local causal influence "
                "does not establish cross-context or computational persistence."
            )
        if self.is_supported(Proposition.P7_ALLOCATION):
            warnings.append(
                "P7 does not entail P8: internal allocation does not establish "
                "externally observable resource control."
            )
        return warnings
# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    definition = ExperimentDefinition(
        pattern_id="P",
        handle="activation_patch",
        metric="target_logit_gap",
        baseline="frequency-matched_paraphrase",
        cue_removed=True,
        emitted_tokens_deleted=True,
        matched_control=True,
        notes="Internal persistence test after surface cue removal.",
    )
    # Example measured causal tail.
    measurements = [
        Measurement(k=1, t="stage_A", c="control", s="form_A", delta_y=0.82),
        Measurement(k=2, t="stage_A", c="control", s="form_A", delta_y=0.80),
        Measurement(k=3, t="stage_A", c="control", s="form_A", delta_y=0.79),
        Measurement(k=4, t="stage_A", c="control", s="form_A", delta_y=0.78),
        Measurement(k=5, t="stage_A", c="control", s="form_A", delta_y=0.77),
    ]
    window = PersistenceWindow(
        k_min=1,
        k_max=5,
        epsilon=0.70,
        delta=0.10,
        training_stages=("stage_A",),
        contexts=("control",),
        surface_forms=("form_A",),
    )
    result = evaluate_persistence(
        definition,
        measurements,
        window,
        proposition=Proposition.P4_CAUSAL,
    )
    print("=== PERSISTENCE TEST ===")
    print(f"Pattern:             {definition.pattern_id}")
    print(f"Proposition:         {result.proposition.value}")
    print(f"Measurements:        {result.n_measurements}")
    print(f"Minimum |Delta_Y|:   {result.minimum_effect:.6f}")
    print(f"Maximum decay:       {result.maximum_decay:.6f}")
    print(f"Epsilon:             {result.epsilon:.6f}")
    print(f"Delta bound:         {result.delta:.6f}")
    print(f"Cue removed:         {result.cue_removed}")
    print(f"Tokens deleted:      {result.emitted_tokens_deleted}")
    print(f"Matched control:     {result.matched_control}")
    print(f"PERSISTENCE:         {result.supported}")
    print("\nReasons:")
    for reason in result.reasons:
        print(f" - {reason}")
    # Demonstrate independence of evidence levels.
    matrix = EvidenceMatrix()
    matrix.establish(Proposition.P1_CORPUS, True)
    matrix.establish(Proposition.P3_REPRESENTATION, True)
    matrix.establish(Proposition.P4_CAUSAL, result.supported)
    print("\n=== NON-ENTAILMENT CHECK ===")
    for warning in matrix.implication_error_check():
        print(f" - {warning}")