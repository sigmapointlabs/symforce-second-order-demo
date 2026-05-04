# A second-order look at SymForce's epsilon recipe

Demonstration of numerical behavior at the SE(2) singularity for SymForce
v0.x's `Pose2_SE2`. Runs in under a minute against a SymForce install.

## What this shows

`Pose2_SE2.from_tangent ∘ to_tangent` is the identity function on
`[θ, t₀, t₁]`. Its second derivative with respect to θ is the constant
zero. We compile both with SymForce's standard codegen path and measure
how close the runtime values are to truth as θ → 0.

The default version's Hessian error grows as θ⁻⁴: at θ = 10⁻⁶ it
returns 44 (in the units of t₀) for what should be zero. The growth
comes from `_tmp3**(-3)` in the generated code, where `_tmp3` is
`sin θ + ε·sign(sin θ)` — division by an ε-scale quantity, cubed.

We also build an alternative ("s-basis") variant: same SymForce
pipeline, same codegen, but the singular scalars `sin(θ)/θ` and
`(1-cos θ)/θ²` are written as Taylor polynomials in `s = θ²`. These
are smooth at s = 0 by construction, so no `ε·sign(...)` machinery
is needed for them. The Hessian error becomes θ-independent at
~1.7×10⁻¹¹.

This is not a bug report. SymForce's first-order behavior — what
nearly all factor graph problems use — is correct everywhere. The
issue appears only at second order, and only for problems that
linearize near the identity.

## Run it

```bash
# from a SymForce repo with the venv activated:
python pose2_se2_demo.py
```

The script:

1. Compiles four functions through `symforce.codegen` (default and
   s-basis, round-trip and Hessian-component)
2. Counts the operations in each generated file
3. Loads the generated functions and sweeps θ from 1 to 10⁻⁸ in f64
4. Writes a results table to `results.txt`

Generated code lands in `output/` for inspection.

## Result

Round-trip identity error: ‖f(θ,t0,t1) − [θ,t0,t1]‖∞
θ         default        s-basis

1e+00       1.421e-10       4.310e-08
   1e-01       1.499e-10       9.983e-12
   1e-02       1.500e-10       1.000e-12
   1e-03       1.500e-10       1.000e-13
   1e-04       1.503e-10       1.000e-14
   1e-05       1.496e-10       1.000e-15
   1e-06       1.055e-10       1.110e-16
   1e-07       1.898e-10       5.551e-17
   1e-08       5.100e-09       1.000e-18

Hessian-component error: |∂²(round_trip[1])/∂θ² − 0|
θ         default        s-basis

1e+00       2.391e-11       3.760e-06
   1e-01       3.329e-11       1.398e-11
   1e-02       2.177e-11       1.642e-11
   1e-03       7.597e-09       1.664e-11
   1e-04       2.617e-05       1.666e-11
   1e-05       4.123e-03       1.667e-11
   1e-06       4.442e+01       1.667e-11
   1e-07       3.999e+03       1.667e-11
   1e-08       4.854e+07       1.667e-11


The s-basis trades a bounded ~10⁻⁶ error far from the singularity for
many orders of magnitude better behavior near it. For applications
that linearize near the identity, this is the desirable trade.

## What's not here

- **Pose3.** The SE(3) case is more involved — additional cancellations
  in the V/V⁻¹ coupling block — and the comparison needs more care to
  be fair. Forthcoming.
- **A drop-in patch for SymForce.** The s-basis here is built outside
  `Pose2_SE2`, not as a registered alternative. Wiring it through
  SymForce's symbolic atom mechanism is a larger piece of work.

## Repo

- `pose2_se2_demo.py` — the script
- `output/` — committed generated code (regenerated each run)
- `results.txt` — committed output of last clean run

## Contact

If you've hit second-order issues on small-rotation factors and want to
compare notes: info@sigmapointlabs.com

— Sigma Point Labs