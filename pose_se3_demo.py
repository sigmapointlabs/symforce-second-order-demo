"""
Pose2_SE2 second-order demo.

Compares SymForce's default epsilon-handled Pose2_SE2 round-trip against
an s-basis variant (Taylor polynomials in s = θ²). Measures both the
generated-code op count and float64 numerical accuracy near the
singularity at θ = 0.

Run:
    python pose2_se2_demo.py

Output:
    results.txt with the comparison table
    output/*.py with the four generated functions for inspection
"""
import importlib.util
import re
import sys
from pathlib import Path

import symforce
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.geo.unsupported.pose2_se2 import Pose2_SE2
from symforce import codegen
from symforce.codegen import PythonConfig

import numpy as np


HERE = Path(__file__).parent
OUTDIR = HERE / "output"
OUTDIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────────────
# Default path: SymForce's Pose2_SE2 round-trip
# ────────────────────────────────────────────────────────────────────────

def round_trip_default(theta: sf.Scalar, t0: sf.Scalar, t1: sf.Scalar,
                       epsilon: sf.Scalar) -> sf.V3:
    """Identity expressed as Pose2_SE2.from_tangent ∘ to_tangent."""
    xi = [theta, t0, t1]
    out = Pose2_SE2.from_tangent(xi, epsilon=epsilon).to_tangent(epsilon=epsilon)
    return sf.V3(out[0], out[1], out[2])


def hessian_default(theta: sf.Scalar, t0: sf.Scalar, t1: sf.Scalar,
                    epsilon: sf.Scalar) -> sf.V1:
    """∂²(round_trip[1])/∂θ² — mathematically zero."""
    xi = [theta, t0, t1]
    out = Pose2_SE2.from_tangent(xi, epsilon=epsilon).to_tangent(epsilon=epsilon)
    return sf.V1(sf.diff(out[1], theta, 2))


# ────────────────────────────────────────────────────────────────────────
# S-basis path: Taylor polynomials in s = θ², no removable singularities
# ────────────────────────────────────────────────────────────────────────
#
# These atoms are smooth at s = 0 by construction.  Their Taylor coefficients
# are the standard ones — we just evaluate them as polynomials rather than
# letting SymPy reconstruct (sin θ)/θ etc. through the singular path.

def A_bar(s):
    """sin(θ)/θ as polynomial in s = θ². Degree-4."""
    return (1 - s * (sf.S(1)/6 - s * (sf.S(1)/120 - s *
            (sf.S(1)/5040 - s * sf.S(1)/362880))))


def B_bar(s):
    """(1 - cos θ)/θ² as polynomial in s. Degree-4."""
    return (sf.S(1)/2 - s * (sf.S(1)/24 - s * (sf.S(1)/720 - s *
            (sf.S(1)/40320 - s * sf.S(1)/3628800))))


def round_trip_sbasis(theta: sf.Scalar, t0: sf.Scalar, t1: sf.Scalar,
                      epsilon: sf.Scalar) -> sf.V3:
    """Identity via s-basis V·V⁻¹·t.  Same shape as the default but
    using smooth atoms for the singular scalars."""
    s = theta * theta

    # V·t expressed via smooth atoms.
    # V = A_bar(s) · I + θ · B_bar(s) · J,  J = [[0,-1],[1,0]]
    a = A_bar(s)
    b_theta = theta * B_bar(s)
    Tx = a * t0 - b_theta * t1
    Ty = b_theta * t0 + a * t1

    # Recover θ from the rotation. atan2 still needs ε for the (0,0) guard;
    # this is the one place ε remains in our path.
    sin_t = sf.sin(theta)
    cos_t = sf.cos(theta)
    theta_out = sf.atan2(sin_t, cos_t, epsilon=epsilon)
    s_out = theta_out * theta_out

    # V⁻¹ = α · I + β · J where solving V·V⁻¹ = I gives:
    #     α = A_bar(s) / (2·B_bar(s)),   β = -θ/2.
    # Both numerator and denominator of α are bounded away from 0 at s=0
    # (A_bar(0)=1, B_bar(0)=½), so no removable singularity.
    alpha = A_bar(s_out) / (2 * B_bar(s_out))
    beta = -theta_out / 2

    t0_out = alpha * Tx - beta * Ty
    t1_out = beta * Tx + alpha * Ty

    return sf.V3(theta_out, t0_out, t1_out)


def hessian_sbasis(theta: sf.Scalar, t0: sf.Scalar, t1: sf.Scalar,
                   epsilon: sf.Scalar) -> sf.V1:
    """∂²(round_trip[1])/∂θ² — mathematically zero, computed via s-basis."""
    out = round_trip_sbasis(theta, t0, t1, epsilon)
    return sf.V1(sf.diff(out[1], theta, 2))


# ────────────────────────────────────────────────────────────────────────
# Codegen
# ────────────────────────────────────────────────────────────────────────

def codegen_to(func, name: str) -> tuple[Path, int]:
    """Codegen `func` into output/<name>.py. Returns (path, total_ops)."""
    cg = codegen.Codegen.function(func=func, config=PythonConfig(), name=name)
    data = cg.generate_function(output_dir=str(OUTDIR / f"_tmp_{name}"),
                                skip_directory_nesting=True)
    src_path = next(Path(p) for p in data.generated_files
                    if Path(p).suffix == ".py" and Path(p).name == f"{name}.py")
    final = OUTDIR / f"{name}.py"
    final.write_text(src_path.read_text())
    # Clean up the codegen scratch dir
    for f in (OUTDIR / f"_tmp_{name}").glob("**/*"):
        if f.is_file():
            f.unlink()
    (OUTDIR / f"_tmp_{name}").rmdir() if (OUTDIR / f"_tmp_{name}").exists() else None
    src = final.read_text()
    m = re.search(r"Total ops:\s*(\d+)", src)
    return final, int(m.group(1)) if m else -1


# ────────────────────────────────────────────────────────────────────────
# Numerical evaluation
# ────────────────────────────────────────────────────────────────────────

def load_func(path: Path, name: str):
    """Load a generated function by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("Pose2_SE2 second-order demo")
    out("=" * 70)

    # 1. Codegen all four functions ────────────────────────────────────
    out("\n[Codegen]")
    rt_d_path, rt_d_ops = codegen_to(round_trip_default, "pose2_default")
    h_d_path, h_d_ops = codegen_to(hessian_default, "pose2_default_hessian")
    rt_s_path, rt_s_ops = codegen_to(round_trip_sbasis, "pose2_sbasis")
    h_s_path, h_s_ops = codegen_to(hessian_sbasis, "pose2_sbasis_hessian")

    out(f"\n  {'function':<36} {'ops':>6}")
    out(f"  {'-' * 36}-{'-' * 6}")
    out(f"  {'round-trip (default)':<36} {rt_d_ops:>6}")
    out(f"  {'round-trip (s-basis)':<36} {rt_s_ops:>6}")
    out(f"  {'∂²(round_trip[1])/∂θ² (default)':<36} {h_d_ops:>6}")
    out(f"  {'∂²(round_trip[1])/∂θ² (s-basis)':<36} {h_s_ops:>6}")

    # 2. Load the generated functions and sweep θ ──────────────────────
    rt_d = load_func(rt_d_path, "pose2_default")
    rt_s = load_func(rt_s_path, "pose2_sbasis")
    h_d = load_func(h_d_path, "pose2_default_hessian")
    h_s = load_func(h_s_path, "pose2_sbasis_hessian")

    out("\n[Numerical sweep, f64, t0=1.0, t1=0.5, ε=1e-10]")
    out("\n  Round-trip identity error: ‖f(θ,t0,t1) − [θ,t0,t1]‖∞")
    out(f"  {'θ':>10}  {'default':>14}  {'s-basis':>14}")
    out(f"  {'-' * 10}  {'-' * 14}  {'-' * 14}")

    t0v, t1v, eps = 1.0, 0.5, 1e-10
    for theta in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        truth = np.array([theta, t0v, t1v])
        err_d = np.max(np.abs(rt_d(theta, t0v, t1v, eps) - truth))
        err_s = np.max(np.abs(rt_s(theta, t0v, t1v, eps) - truth))
        out(f"  {theta:>10.0e}  {err_d:>14.3e}  {err_s:>14.3e}")

    out("\n  Hessian-component error: |∂²(round_trip[1])/∂θ² − 0|")
    out(f"  {'θ':>10}  {'default':>14}  {'s-basis':>14}")
    out(f"  {'-' * 10}  {'-' * 14}  {'-' * 14}")

    for theta in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        err_d = abs(h_d(theta, t0v, t1v, eps)[0])
        err_s = abs(h_s(theta, t0v, t1v, eps)[0])
        out(f"  {theta:>10.0e}  {err_d:>14.3e}  {err_s:>14.3e}")

    out(f"\n[Generated code in: {OUTDIR.relative_to(HERE)}/]")
    out("\nDone.")

    (HERE / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()