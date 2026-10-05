# Full-amplitude audit and boundary prediction (derivation; propagation checks pending)

Conventions follow submitted Eqs. (1)–(11). Let ε=R0^(-1/2), a=ζ/2=ε^(-1)+τε/2,
u0=a−iηε/2 and x=aR. All τ-dependent factors must remain when locating events.

At fixed τ and x, define lq=Jq+xJq′=hq+Jq/2. Direct expansion of g sqrt(u) Jq(uR)
about u0 gives, after removing the same nonzero factor for every q,

Hq = Ai(X0) Jq − iε Ai′(X0) lq + O(ε²), X0=−τ+iη.

At b0=j′m,1, ∫x(J−l−−J+l+)dx=I1+I0/2=0 and
∫x(l−²−l+²)dx=I11+I1+I0/4=0. The second-derivative cross term is
I2+I1+I0/4=(Dm−1/4)I0. The terms from the g derivatives are therefore
order independent *after integration to this specific boundary*. They cannot
be discarded pointwise or for another boundary. The complex argument shift
−iηε²x/2 contributes no real intensity correction at O(ε²).

The full caustic prefactor has squared modulus proportional to
exp[−2ητε−ητ²ε³/2], after removing τ-independent factors. This common O(ε)
term changes each event and participates in the O(ε³) order difference.
Radial Jacobian is proportional to (1+τε²/2)^(-2). Both are retained in
`code/formal_asymptotics.py`. Thus Eq. (32) is supported through O(ε²) within
the inward fold branch, for fixed m, bounded τ and a simple selected domain.
It does not yet establish a uniform remainder for the complete propagation.

## Arbitrary fixed scaled boundary

I0(b)=2m Jm(b)².
I1(b)=−I0/2+b²Bm(b)/2, with Bm=J(m−1)²−J(m+1)².
Hence the full O(ε) coefficient becomes
2 Im[Ai*Ai′]/|Ai|² × bJm′(b)/Jm(b).
At a generic b, this is order dependent and cannot enter a common Gη.
Near b=b0(1+δ), bJm′/Jm=(m²−b0²)δ+O(δ²).
This is a prediction for the complete normalized signal; the event response
still requires its peak and crossing perturbation. No boundary independence is asserted.

## Actual zero, when simple and on the same branch

Set v(τ)=Im[Ai′(−τ+iη)/Ai(−τ+iη)]. Write the normalized integral density as
H0=xBm, H1=2v x(J−l−−J+l+). At b0:
H1(b0)=v b0 H0′(b0), H0′(b0)=4m(m²/b0²−1)Jm(b0)²<0.
The root shift is b1=−b0 v, and the moving-upper-limit correction is
−ε² H1(b0)²/[2H0′(b0)]. Dividing by I0 gives
ε²(b0²−m²)v²=ε²(3/4−Dm)v².
Thus the order-dependent part for the *actual* root is
ε² Dm[τ−v(τ)²], with the remaining terms common to all m.
For a rising fraction γ, the candidate leading coefficient becomes
K_actual=γf(p){[p−v(p)²]−[h−v(h)²]}/f′(h).
The coefficient was recorded before root-propagation tests in
`analytic_preregistration.json`: it differs from K_proxy. This statement is
conditional on the inward branch, simple root, and nondegenerate peak/crossing.

## Dm monotonicity for positive real order

Use quadratic form ∫r|y′|²dr+m²∫|y|²dr/r in L²(rdr) on (0,1), with
natural Neumann condition y′(1)=0. For every m in a compact subset of (0,∞)
its form domain is the same intersection of the two finite-energy domains;
forms depend analytically on m². The lowest eigenvalue is simple and its
regular eigenfunction is Jm(j′m,1 r), positive before the first derivative zero.
Near 0, y=O(r^m), so both required integrals converge and the variational
boundary term r y y′ vanishes. Hellmann–Feynman therefore yields
λ′(m)=2m∫y²dr/r /∫ry²dr and D′(m)=2m−λ′(m)<0 because 1/r>r on (0,1).
This proof excludes m=0; it does not give a finite-R0 uniform high-order bound.
Numerical integral checks are stored independently in the preregistration.

## High-order work

`formal_asymptotics.py` implements exact finite polynomial algebra in ε and
the fold variable t through ε⁴, integrates t^n with (−i)^n Ai^(n), and squares
consistently in ε. No propagation-fit coefficients enter. The opposite Hankel
branch and lower spectral endpoint are not included in this local expansion;
comparison with direct fields is necessary before attributing residuals.
