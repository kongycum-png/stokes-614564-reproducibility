# Original equations (1)–(34): audit

The original equation numbering is retained below. Algebraic checks are stored in theory/equation_algebra_checks.json; field-level comparisons are indexed in EVIDENCE_INDEX.md. Algebraic identities alone do not bound the complete propagated field.

| Original | Definition/result | Finding and applicable conditions |
|---|---|---|
| 1 | Circular basis | Orthonormal under exp(-i omega t); S3=2 Im(Ex* Ey) with this convention. Jones check agrees to floating-point precision. |
| 2 | Airy envelope | Finite transverse power for alpha>0. f(0) is exponentially small, not exactly zero; the input is an L2 field, and a globally smooth vortex core is not assumed. |
| 3 | Two-channel input | Equal amplitudes 1/sqrt2 produce equal input powers. |
| 4 | Radial representation | Algebraically identical to (3) for P=1; common m phase is retained. |
| 5 | q+,q-,P,m | q+=m-1, q-=m+1; P=1. Generalization needs m±P and different domain roots. |
| 6 | Azimuthal separation | Follows from rotational symmetry of the scalar propagator; signed negative q has the same radial propagation as |q| for this source convention. |
| 7 | Reduced RS radial integral | Reproduced. This is the source-radius-expanded kernel; it is not the unexpanded RS integral or exact-kz propagation. |
| 8 | D and A | Correct for the stated reduced kernel, with physical wavenumber k. The exp(ikD) factor must be retained except for a common carrier. |
| 9 | Propagated transverse field | Consistent with equal input channel amplitudes and azimuthal phases. |
| 10 | I+,I- | Factors 1/2 are required and used in every stored signal. |
| 11 | S0,S3 | S3 is the transverse circular intensity difference. It is not the full dual-symmetric spin density. |
| 12 | Leading Bessel difference | The recurrence identity is exact; its use as the propagated radial profile is a leading inward-caustic approximation. |
| 13 | First proxy boundary | First positive Jm' zero for m>0. It is a proxy for a Stokes sign change, not an intensity zero. A fixed x boundary is a moving physical radius. |
| 14 | Q and Pin | Input-normalized signed local intensity difference; Q=T pbar3. It includes collected power and mean circular imbalance. |
| 15 | Half-height event | Requires an interior nondegenerate maximum and preceding rising crossing. An endpoint cannot replace the maximum. |
| 16 | Relative advance | Delta=h1-hm; positive means earlier for m. Single-event error and difference error are separate. |
| 17 | Scaled variables | R0=r0/w, R=rho/w, zeta=z/(kw^2), eta=2alpha sqrtR0 are dimensionless. |
| 18 | Inward fold representation | Asymptotic, with endpoint/outward contributions outside the local fold series. Partitioned calculations are recorded in data/hankel_branches/. No uniform full-field remainder is asserted. |
| 19 | Slow amplitude | Full g sqrt(u) Jq and the first Debye correction must be expanded, not just sqrt(u)Jq. |
| 20 | g and saddle location | s*=R0+(u+i alpha)^2. The next radial Airy-moment amplitude correction is also audited below. |
| 21 | Airy control X | Completing the cubic with u0=zeta/2-i alpha gives X=R0-zeta^2/4+i alpha zeta; the alpha^2 terms cancel. |
| 22 | zc,Lc,tau | Units and substitution checked. Delta z=Lc Delta tau. |
| 23 | X(tau) | Exact substitution into (21), including the complex O(R0^-1) correction. |
| 24 | Airy-Hankel series | Moments are i^(-n) Ai^(n). Field truncation and a consistently squared power series are different approximations. |
| 25 | Fq' | Direct differentiation and an independent five-point check agree. |
| 26 | Fq'' | Bessel equation gives u^-3/2(q^2-x^2-1/4)Jq; independently checked. |
| 27 | I0 | I0=2m Jm(b0)^2 for m>0. Numerical quadrature covers m=1,2,3,4,6,8,12,16. |
| 28 | I1 | I1/I0=-1/2 at b0 only. A different boundary has an additional boundary term. |
| 29 | I11 | I11/I0=1/4 at b0. Full-amplitude ell=J+xJ' changes the corresponding integral to zero. |
| 30 | D integral | Agrees with independent quadrature. For the full amplitude the second cross integral is (D-1/4)I0, whose -1/4 part is common. |
| 31 | D expression | D=m^2+3/4-b0^2; contains a numerically evaluated special-function zero but no propagation-fitted parameter. Monotonicity proof applies for m>0. |
| 32 | Q asymptotic factorization | Supported through eps^2 for the specified proxy, full amplitude, fixed finite m, bounded tau. Common terms affect individual events and higher-order differences. Generic boundaries introduce an order-dependent eps term. |
| 33 | Keta | Valid only for an existing nondegenerate leading Airy maximum and rising crossing. The numerical double-stationary point near eta=0.936525 is recorded separately; no global absence theorem is claimed. |
| 34 | Delta law | Conditional leading law for the proxy and defined events. Actual roots, generic boundaries, fixed-alpha paths, large m, or near-degenerate events require their own analysis. |

## Next radial Airy-moment amplitude term

Writing a=alpha-iu and s*=R0-a^2, the tilted Airy generating integral has central second moment 2a. Expanding sqrt(R0-x) about x=a^2 gives the multiplicative correction -a/(4 s*^2). In the fold variables it is i eps^3/16-i eps^4 t/16+O(eps^5). The constant imaginary term is a common field phase. After coherent multiplication and squaring, the additional normalized signal at eps^4 is -Re(A* A')/8, independent of m at the proxy boundary. The independent polynomial calculation confirms this common term and its cancellation from order differences through eps^4. It must not be omitted in a claim about individual fourth-order event coefficients.

The original N=4 numerical truncation is retained as a reproduced historical approximation. Its event error is not an error bound for the full revised field or a substitute for the separate propagation-model comparison.
