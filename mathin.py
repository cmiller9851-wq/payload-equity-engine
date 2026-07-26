import dataclasses
import hashlib
import json
import math
import sys
from typing import Dict, List, Sequence, Tuple


# --- ALGEBRAIC ERRORS & STRUCTURAL DATACLASSES ---

class AlgebraError(ArithmeticError):
    """Raised when a geometric algebra invariant or rotor unit norm fails."""
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class Vector3D:
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclasses.dataclass(frozen=True, slots=True)
class Multivector:
    """
    8-component Clifford Algebra Cl(3,0) Multivector Representation:
    c = (scalar, e1, e2, e3, e12, e23, e31, e123)
    """
    _c: Tuple[float, float, float, float, float, float, float, float]

    @classmethod
    def scalar(cls, val: float) -> "Multivector":
        return cls((float(val), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    @classmethod
    def vector(cls, v: Vector3D) -> "Multivector":
        return cls((0.0, float(v.x), float(v.y), float(v.z), 0.0, 0.0, 0.0, 0.0))

    @classmethod
    def bivector(cls, e12: float, e23: float, e31: float) -> "Multivector":
        return cls((0.0, 0.0, 0.0, 0.0, float(e12), float(e23), float(e31), 0.0))

    @property
    def components(self) -> Tuple[float, ...]:
        return self._c

    def __add__(self, other: "Multivector") -> "Multivector":
        return Multivector(tuple(a + b for a, b in zip(self._c, other._c)))

    def __sub__(self, other: "Multivector") -> "Multivector":
        return Multivector(tuple(a - b for a, b in zip(self._c, other._c)))

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric Product in 3D Euclidean Space Cl(3,0)."""
        a, b = self._c, other._c
        
        # 0: scalar, 1: e1, 2: e2, 3: e3, 4: e12, 5: e23, 6: e31, 7: e123
        c0 = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7]
        c1 = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[6] + a[4]*b[2] - a[5]*b[7] - a[6]*b[3] - a[7]*b[5]
        c2 = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[5] - a[4]*b[1] + a[5]*b[3] - a[6]*b[7] - a[7]*b[6]
        c3 = a[0]*b[3] - a[1]*b[6] + a[2]*b[5] + a[3]*b[0] - a[4]*b[7] - a[5]*b[2] + a[6]*b[1] - a[7]*b[4]
        c4 = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7] + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3]
        c5 = a[0]*b[5] + a[1]*b[7] + a[2]*b[3] - a[3]*b[2] + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] + a[7]*b[1]
        c6 = a[0]*b[6] - a[1]*b[3] + a[2]*b[7] + a[3]*b[1] - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[2]
        c7 = a[0]*b[7] + a[1]*b[5] + a[2]*b[6] + a[3]*b[4] + a[4]*b[3] + a[5]*b[1] + a[6]*b[2] + a[7]*b[0]

        return Multivector((c0, c1, c2, c3, c4, c5, c6, c7))

    def reverse(self) -> "Multivector":
        """Reversal (tilde operator ~) for Clifford multivector."""
        c = self._c
        return Multivector((c[0], c[1], c[2], c[3], -c[4], -c[5], -c[6], -c[7]))

    def norm_sq(self) -> float:
        """Calculates scalar norm squared via product with reversal."""
        prod = self * self.reverse()
        return prod._c[0]

    def normalized(self, eps: float = 1e-12) -> "Multivector":
        ns = self.norm_sq()
        if ns <= eps:
            raise AlgebraError("Cannot normalize near-zero multivector.")
        scale = 1.0 / math.sqrt(ns)
        return Multivector(tuple(x * scale for x in self._c))

    def extract_vector(() -> Vector3D:
        return Vector3D(x=self._c[1], y=self._c[2], z=self._c[3])

    def to_rotor_dict(self) -> Dict[str, float]:
        return {
            "1": self._c[0],
            "e1": self._c[1],
            "e2": self._c[2],
            "e3": self._c[3],
            "e12": self._c[4],
            "e23": self._c[5],
            "e31": self._c[6],
            "e123": self._c[7],
        }


# --- DOMAIN MATHEMATICAL LOGIC ---

def golden_ratio() -> float:
    return (1.0 + math.sqrt(5.0)) / 2.0


def golden_angle_rad() -> float:
    return math.pi * (3.0 - math.sqrt(5.0))


def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(1, n):
        a, b = b, a + b
    return b


def update_bayes_belief(prior: float, p_true: float, p_false: float, eps: float = 1e-12) -> float:
    denom = (p_true * prior) + (p_false * (1.0 - prior))
    if abs(denom) < eps:
        return prior
    return (p_true * prior) / denom


def construct_golden_angle_rotor(
    plane_bivector: Multivector, angle_scale: float, eps: float = 1e-12
) -> Tuple[Multivector, float]:
    b_unit = plane_bivector.normalized(eps=eps)
    theta = golden_angle_rad() * float(angle_scale)
    half = theta / 2.0

    c, s = math.cos(half), math.sin(half)
    
    # Rotor = cos(theta/2) - bivector * sin(theta/2)
    rotor = Multivector.scalar(c) - (Multivector.scalar(s) * b_unit)

    if abs(rotor.norm_sq() - 1.0) > eps:
        raise AlgebraError("Rotor unit norm violation: non-isometric scaling detected.")

    return rotor, theta


def apply_sandwich_rotation(v: Vector3D, rotor: Multivector) -> Vector3D:
    v_mv = Multivector.vector(v)
    rotated_mv = rotor * v_mv * rotor.reverse()
    return Vector3D(x=rotated_mv._c[1], y=rotated_mv._c[2], z=rotated_mv._c[3])


# --- DETERMINISTIC PIPELINE EXECUTION ENGINE ---

def execute_ga_bayesian_pipeline(
    tx_id: str,
    prior: float,
    p_true: float,
    p_false: float,
    fib_n: int,
    input_vector: Vector3D,
    bivector_plane: Multivector,
) -> Dict[str, object]:
    posterior = update_bayes_belief(prior, p_true, p_false)
    phi = golden_ratio()
    fib_val = fibonacci(fib_n)

    # Scale: F_n * phi * phi_term (using posterior belief)
    scale = float(fib_val) * phi * (1.0 + posterior)

    rotor, golden_rad = construct_golden_angle_rotor(bivector_plane, angle_scale=posterior)
    rotated_v = apply_sandwich_rotation(input_vector, rotor)

    event_payload = {
        "tx_id": tx_id,
        "posterior": posterior,
        "phi": phi,
        "fib": fib_val,
        "scale": scale,
        "golden_angle_rad": golden_rad,
        "rotor": rotor.to_rotor_dict(),
        "rotated_vector": rotated_v.to_dict(),
        "rotor_norm_sq": rotor.norm_sq(),
    }

    return event_payload


def emit_execution_telemetry(
    run_id: str,
    timestamp_utc: str,
    event_outputs: List[Dict[str, object]],
) -> Dict[str, object]:
    # Cryptographic SHA-256 state transition digest
    canonical_bytes = json.dumps(event_outputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    execution_hash = hashlib.sha256(canonical_bytes).hexdigest()

    total_records = len(event_outputs)
    posterior_sum = sum(float(e["posterior"]) for e in event_outputs)
    scale_sum = sum(float(e["scale"]) for e in event_outputs)
    last_tx_id = event_outputs[-1]["tx_id"] if event_outputs else ""

    return {
        "telemetry_metadata": {
            "schema_version": "2.1.0",
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "status": "COMPLETED",
            "execution_hash": execution_hash,
        },
        "aggregate_metrics": {
            "total_records": total_records,
            "posterior_sum": posterior_sum,
            "scale_sum": scale_sum,
            "last_tx_id": last_tx_id,
        },
        "event_outputs": event_outputs,
    }


# --- VERIFICATION RUNNER ---

if __name__ == "__main__":
    # Test Parameters
    e12_plane = Multivector.bivector(e12=1.0, e23=0.0, e31=0.0)
    v_in = Vector3D(x=1.0, y=0.0, z=0.0)

    event = execute_ga_bayesian_pipeline(
        tx_id="tx_1001",
        prior=0.5,
        p_true=0.8,
        p_false=0.2,
        fib_n=6,  # fib(6) = 8
        input_vector=v_in,
        bivector_plane=e12_plane,
    )

    telemetry_payload = emit_execution_telemetry(
        run_id="run_1719300000",
        timestamp_utc="2026-07-25T21:32:00Z",
        event_outputs=[event],
    )

    json_output = json.dumps(telemetry_payload, indent=2)
    sys.stdout.write(json_output + "\n")
