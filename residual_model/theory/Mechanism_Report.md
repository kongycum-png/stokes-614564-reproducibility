---
title: Finite radius mechanism of the eta 0.35 axial event residual
subtitle: Manuscript 614564 reproducible evidence report
date: 4 October 2026
---

# Result and scope

The nonmonotonic scaled advance at $\eta=0.35$, $m=3$ is reproduced by an independent finite angular Airy integral and explained by the higher order terms of the same smooth peak and half-height event. The fourth order event series does not include enough terms to locate this minimum. Its non-fitted extension gives $c_5=3.36680300$ and $c_6=13.35298604$, followed by smaller corrections at the radii in question. The fourteenth order prediction locates the minimum at $R_0=36.37128$; the full Fresnel integral gives $36.37127$, and the manuscript's reduced R-S integral gives $36.37130$. The last two digits are numerical estimates, not certified interval endpoints.

This report provides an exact integral reduction, a rigorous bound on the artificial negative-source extension, coefficient recursion, order ablations, new-radius predictions, independent positive-source quadrature and arbitrary precision checks. The integral identities and source-extension bound are analytic results. The reported event locations, absence of a branch change over the tested range and finite-order error envelopes are numerical evidence. We do not claim a computer-assisted interval proof of a unique global minimum for every radius or a universal result for other apertures, orders or damping parameters.

# Model and event definition

The study retains the original radial envelope, the fixed-$\eta$ path $\alpha=\eta/(2\sqrt{R_0})$, $m=1$ as the reference, $P=1$, the proxy aperture $b_m=j'_{m,1}$, and the first stationary maximum in $-3\leq\tau\leq5$. The crossing is the nearest preceding rising half-height. No physical zero-contour root is selected in this observable. The quantity being explained is $R_0\Delta$, not the error of a fitted coefficient.

$$\zeta=2\sqrt{R_0}+\tau/\sqrt{R_0},\qquad R=\rho/w=2x/\zeta.$$

Apart from common phases and physical constants, the Fresnel radial integral is

$$I_q(R,\zeta)=\int_0^\infty s\,\operatorname{Ai}(R_0-s)e^{\alpha(R_0-s)}e^{is^2/(2\zeta)}J_q(Rs/\zeta)\,ds.$$

The reduced R-S model replaces $\zeta$ in the phase and Bessel argument by $d=\sqrt{\zeta^2+(R/kw)^2}$ and multiplies the integral by $\zeta/d^2$. The values used are $w=0.08$ mm, $\lambda=0.0006328$ mm and $kw=2\pi w/\lambda$. The original finite-radius data and all earlier delivery files are preserved.

# Exact compact reduction

## The Airy convolution

For $\alpha>0$ define the one-dimensional Fresnel propagation of the apodized Airy function:

$$F(y,\zeta)=\frac{1}{\sqrt{2\pi i\zeta}}\int_{-\infty}^{\infty}\operatorname{Ai}(t)e^{\alpha t}e^{i(y-t)^2/(2\zeta)}\,dt.$$

The initial profile has Fourier transform $\exp[(\alpha-ik)^3/3]$. Multiplication by $\exp(-i\zeta k^2/2)$ and completion of the cubic phase give

$$F(y,\zeta)=\operatorname{Ai}(y-\zeta^2/4+i\alpha\zeta)e^{\Phi(y,\zeta)},$$

$$\Phi(y,\zeta)=\alpha(y-\zeta^2/2)+i[\zeta(y+\alpha^2)/2-\zeta^3/12].$$

This is the finite-energy Airy solution of Siviloglou and Christodoulides, Eq. (5) [1]. Setting $t=R_0-s$, $a=R\cos\theta$ and $y=R_0-a$ converts the full-line source integral without its factor $s$ to $\sqrt{2\pi i\zeta}\,e^{-ia^2/(2\zeta)}F(R_0-a,\zeta)$. Applying $i\zeta\partial_a$ restores the factor $s$. The sign follows directly by differentiating $\exp(-ias/\zeta)$.

For integer $q$, the Bessel angular identity [2] is

$$J_q(v)=\frac{i^q}{2\pi}\int_0^{2\pi}e^{-iv\cos\theta}e^{iq\theta}\,d\theta.$$

Let $X=R_0-\zeta^2/4+i\alpha\zeta$. After removing a nonzero common factor and normalizing by $\zeta^2/2$, the angular field is

$$\mathcal H_q=\frac{1}{2\pi}\int_0^{2\pi}e^{iq\theta}G(a,\zeta)\left[B(a,\zeta)\operatorname{Ai}(X-a)-\frac{2i}{\zeta}\operatorname{Ai}'(X-a)\right]d\theta,$$

$$G(a,\zeta)=e^{-i\zeta a/2-\alpha a-ia^2/(2\zeta)},\qquad B(a,\zeta)=1-\frac{2i\alpha}{\zeta}+\frac{2a}{\zeta^2}.$$

The normalized proxy signal is therefore

$$\widetilde Q_m=\frac{V e^{-2\eta\tau\epsilon-\eta\tau^2\epsilon^3/2}}{I_{0,m}}\int_0^{b_m}x\left(|\mathcal H_{m-1}|^2-|\mathcal H_{m+1}|^2\right)dx,$$

where $\epsilon=R_0^{-1/2}$, $V=1+\tau\epsilon^2/2$ and $I_{0,m}=2mJ_m^2(b_m)$. This normalization differs from $Q_m$ by a positive factor independent of $\tau$, so it leaves peaks and half-height events unchanged. The $V$ and exponential factors must remain during event extraction.

For reduced R-S propagation, use $d$ instead of $\zeta$ inside $\mathcal H_q$. Since $d$ depends on $x$, its signal prefactor must be kept inside the $x$ integral:

$$W_{\mathrm{RS}}(x)=\frac{d\epsilon}{2}e^{-2\eta\tau\epsilon-\eta\tau^2\epsilon^3/2-\alpha(R/kw)^2},$$

$$\widetilde Q_m^{\mathrm{RS}}=\frac{1}{I_{0,m}}\int_0^{b_m}xW_{\mathrm{RS}}(x)\left(|\mathcal H_{m-1}(d)|^2-|\mathcal H_{m+1}(d)|^2\right)dx.$$

This expression retains the exact geometry of the manuscript's reduced kernel. It is not a full Maxwell or unexpanded Rayleigh-Sommerfeld calculation. The angular identity sums the complete field coherently and introduces no arbitrary spectral partition.

## Controlled negative-source extension

Replacing the physical interval $s\geq0$ by the full real line adds an integral over $s=-v<0$. Since $|J_q(t)|\leq1$ for real $t$ and integer $q$, the positive-axis Airy bound [3] and convexity of $(R_0+v)^{3/2}$ give, for $\alpha<\sqrt{R_0}$,

$$|\delta I_q|\leq\frac{e^{\alpha R_0-2R_0^{3/2}/3}}{2\sqrt\pi R_0^{1/4}(\sqrt{R_0}-\alpha)^2}.$$

Indeed, $\operatorname{Ai}(u)\leq e^{-2u^{3/2}/3}/(2\sqrt\pi u^{1/4})$ for $u>0$, and $2(R_0+v)^{3/2}/3\geq2R_0^{3/2}/3+\sqrt{R_0}v$. The remaining integral is $\int_0^\infty v e^{-(\sqrt{R_0}-\alpha)v}dv$. The resulting dimensionless source-integral bounds are $1.1541\times10^{-36}$ at $R_0=24$, $4.8580\times10^{-76}$ at 40 and $2.433\times10^{-275}$ at 96. These are bounds before the known propagation prefactor, not estimates obtained from cancellation. This artificial extension cannot account for the observed $10^{-2}$ changes in $R_0\Delta$.

# Coefficients and event recursion

The angular formula admits direct expansion around $\epsilon=0$. In the Fresnel case,

$$a=\frac{\epsilon x\cos\theta}{V},\quad X=-\tau+i\eta+\epsilon^2(-\tau^2/4+i\eta\tau/2),\quad \zeta a/2=x\cos\theta.$$

Expand $V^{-1}$ geometrically, the two remaining exponentials by power-series multiplication, and $\operatorname{Ai}(X-a)$ about $-\tau+i\eta$. Its derivatives satisfy $A^{(k)}=XA^{(k-2)}+(k-2)A^{(k-3)}$ for $k\geq3$, with $A''=XA$. The field coefficients $H_{q,n}$ then determine every intensity coefficient by the convolution $\sum_{j=0}^{n}H_{q,j}H_{q,n-j}^*$. The common signal prefactor is convolved at the same total order. There is no truncation by derivative number followed by uncontrolled squaring.

Write $Q_m(\tau,\epsilon)=\sum_n q_{m,n}(\tau)\epsilon^n$, $p_m=p_0+\sum_{n\geq1}p_{m,n}\epsilon^n$ and $h_m=h_0+\sum_{n\geq1}h_{m,n}\epsilon^n$. For a simple peak and rising crossing, coefficient extraction gives

$$p_{m,n}=-\frac{[\epsilon^n]\,\partial_\tau Q_m(p_0+\sum_{j<n}p_{m,j}\epsilon^j,\epsilon)}{q_{m,0}''(p_0)},$$

$$h_{m,n}=\frac{\tfrac12[\epsilon^n]Q_m(p_0+\sum_{j<n}p_{m,j}\epsilon^j,\epsilon)-[\epsilon^n]Q_m(h_0+\sum_{j<n}h_{m,j}\epsilon^j,\epsilon)}{q_{m,0}'(h_0)}.$$

The unknown $p_{m,n}$ does not enter the peak height at this order because $q_{m,0}'(p_0)=0$. This proves the recursion under its nondegeneracy assumptions. The coefficients $c_n=h_{1,n}-h_{3,n}$ are fixed by the model and event definition. They do not depend on a chosen set of propagation radii. The normalized full-line signal is analytic locally in $\epsilon$ where $V\neq0$; the implicit function theorem supplies local analytic events while their denominators remain nonzero. The source extension contributes a beyond-all-orders term to the physical half-line integral. These local facts alone do not certify a global convergence radius for the event series.

| $n$ | $c_n$ | $n$ | $c_n$ |
|---|---:|---|---:|
|2|4.41467828893|3|-1.44016807989|
|4|2.58765834920|5|3.36680299947|
|6|13.3529860411|7|21.1128977205|
|8|-49.8888463954|9|91.5175673968|
|10|237.809936151|11|-385.651361911|
|12|738.534340134|13|-809.692872335|
|14|3130.68031359|1|0 analytically|

The computed $c_1$ is $1.2\times10^{-15}$. The first three nonzero coefficients reproduce the independent earlier fold derivation. Changing the Cauchy derivative-contour radius from 0.7 to 0.9 changes no coefficient through order 14 by more than $8.3\times10^{-10}$. Separate order-10 checks refine the radial and angular quadratures from 32/64 to 48/96 nodes. Cauchy extraction evaluates derivatives of an analytic formula; it does not fit propagated event values.

# Why the minimum was missed

The scaled event series is $R_0\Delta=\sum_{n\geq2}c_n R_0^{1-n/2}$. At fourth order it contains only $A_3+B_3/\sqrt{R_0}+C_3/R_0$. Its stationary point is $(2C_3/|B_3|)^2=12.91361$, outside the observed intermediate-radius minimum. Adding the positive fifth- and sixth-order terms shifts the minimum toward the full calculation.

| Event truncation | Predicted minimum $R_0$ | Minimum $R_0\Delta$ |
|---|---:|---:|
|4|12.91361|4.21429595|
|5|24.97050|4.25708537|
|6|34.41707|4.27232531|
|8|35.78792|4.27406412|
|10|36.41148|4.27452784|
|12|36.37256|4.27450296|
|14|36.37128|4.27450221|
|Full Fresnel|36.37127|4.27450221|
|Reduced R-S|36.37130|4.27450539|

These are order controls without coefficient fitting. The high orders do not merely improve a regression statistic: they predict the location and depth of the minimum and the sign of its curvature. The decomposition below additionally identifies which operations generate the missing terms.

For each order $n$, split the new signal coefficient into (i) $H_0H_n^*+H_nH_0^*$, (ii) lower-field products and the common prefactor, and (iii) inherited nonlinear terms from the two implicit event equations. Applying the linear crossing functional $[q_n(p_0)/2-q_n(h_0)]/q_0'(h_0)$ to (i) and (ii), and assigning the remainder of $h_n$ to (iii), gives the following exact algebraic grouping in this declared expansion convention.

|Order|New field term|Lower products and prefactor|Nonlinear event terms|Total $c_n$|
|---|---:|---:|---:|---:|
|5|1.04031657|23.80479484|-21.47830841|3.36680300|
|6|22.98782712|54.38661509|-64.02145617|13.35298604|

The large signed cancellations rule out interpreting $c_5$ or $c_6$ as the power carried by a separate branch. They are coherent finite-radius field corrections processed through peak normalization and a moving half-height. As an event control, forcing the target's normalization peak to the reference peak changes $R_0\Delta$ to 4.52744, 4.42926 and 4.36945 at 24, 40 and 96; the dip over these three points disappears. This artificial control isolates the importance of the peak condition but is not proposed as a replacement observable. Removing $c_5$ or $c_6$ from the fixed analytic series likewise displaces the minimum; all values are retained in coefficient_decomposition.json.

# Predictions and independent numerical evidence

All coefficients were generated solely from the analytic identity; the generators never load propagation data. Predictions at the new radii were evaluated with these fixed coefficients. New controls are $R_0=20,28,36,44,48,64,80$; the report also recomputes the original radii and extends to 256. The maximum fourteenth-order error in $R_0\Delta$ against the full Fresnel identity on the new controls is $2.84\times10^{-7}$, attained at 20. Over sampled $24\leq R_0\leq96$ it is below $7.1\times10^{-8}$. The small reduced R-S/Fresnel difference is retained separately, rather than absorbed into a fitted high-order term.

| $R_0$ | Reduced R-S $R_0\Delta$ | Fourteenth-order Fresnel error in $R_0\Delta$ | R-S minus Fresnel |
|---|---:|---:|---:|
|24|4.286137374|$+7.02\times10^{-8}$|$3.225\times10^{-6}$|
|36|4.274510642|$-2.94\times10^{-9}$|$3.178\times10^{-6}$|
|40|4.274925216|$-2.94\times10^{-9}$|$3.172\times10^{-6}$|
|48|4.277668295|$-1.75\times10^{-9}$|$3.166\times10^{-6}$|
|64|4.285437979|$-4.95\times10^{-10}$|$3.161\times10^{-6}$|
|96|4.299868386|$-5.99\times10^{-11}$|$3.161\times10^{-6}$|

The angular implementation uses Gauss-Legendre integration in $x$ and periodic trapezoidal integration in angle. The independent positive-source implementation evaluates the original oscillatory radial integral by composite Gauss-Legendre quadrature, with exact reduced R-S geometry and without a spectral transform, phase expansion, radial interpolation or axial spline. Separate tests halve the source panels and extend the exponential cutoff. Refining panels from 0.5 to 0.25 and cutoff from 22 to 26 changes the scaled advance by at most $4.64\times10^{-10}$. The finest direct-source results differ from the compact reduced R-S evaluation by at most $1.41\times10^{-12}$ in $R_0\Delta$. The arbitrary precision implementation uses a third angular evaluation: an Airy Taylor series in $a$ and analytic angular moments expressed as Bessel derivatives. It uses mpmath at 35 and 50 decimal digits, 20/28 radial nodes and 36/48 angular-series terms. At the two event locations for each of $m=1,3$ and $R_0=24,40,96$, the maximum signal difference between these precision settings is $6.0\times10^{-33}$; the double-precision angular implementation differs by at most $1.45\times10^{-15}$. These are measured numerical differences, not interval-certified bounds.

The first peaks and preceding rising crossings continue smoothly across the 16 tested radii from 16 to 256. The minimum crossing slope of the normalized signal is 0.1035; the minimum magnitude of the selected peak curvature is 0.2561. All detected later maxima are retained in event_continuation.csv. The selected first maxima remain separated from them. Thus the tested minimum is not associated with a detected event switch, peak degeneracy or a selected-zero jump.

![Mechanism evidence. (a) Non-fitted event expansions and full reduced R-S values. (b) Truncation errors against the full Fresnel identity and the separately retained geometry correction. (c) First peak and rising half-height continuation. (d) Signed lower-order contributions to the scaled advance.](../figures/residual_mechanism.png){width=6.2in}

# Error accounting and interpretation

The signed residual admits the exact bookkeeping identity

$$\Delta_{\mathrm{RS}}-\Delta_4=(\Delta_{\mathrm{RS}}-\Delta_{\mathrm F})+\sum_{n=5}^{14}c_nR_0^{-n/2}+(\Delta_{\mathrm F}-\Delta_{14}).$$

The first term is the model geometry correction, the second is the analytic high-order event contribution and the third is the retained series remainder. The source-extension bound and numerical evaluation error are reported separately. This identity closes the residual without allocating coherent powers to arbitrary spectral windows.

|Contribution|Evidence and scale|Interpretation|
|---|---|---|
|Source extension|Analytic bound $1.16\times10^{-36}$ on $I_q$ at 24|Rigorous positive-source extension control|
|Angular and radial quadratures|48/96 to 72/128 nodes: scaled-advance change below $6.2\times10^{-14}$|Measured numerical convergence|
|Independent source quadrature|Source refinement change below $4.64\times10^{-10}$ in scaled advance|Direct check of the original integral|
|Arbitrary precision signal evaluation|Maximum double/50-digit difference $1.45\times10^{-15}$|Independent Airy/Bessel implementation|
|Fourteenth-order remainder|Below $7.1\times10^{-8}$ in sampled $R_0\Delta$, 24--96|Finite-order approximation, not quadrature noise|
|Reduced R-S versus Fresnel|$3.16$--$3.23\times10^{-6}$ in $R_0\Delta$, 24--96|Separate kernel geometry correction|
|Observed dip and recovery|24 to 40: $-0.01121216$; 40 to 96: $+0.02494317$|Resolved finite-radius behavior|

The mechanism supported here is a smooth, coherent high-order correction to the fixed-proxy signal and its peak-normalized event. It does not require a root change or a numerical artifact, and the independently derived terms quantitatively account for the minimum. A unique division of this correction into inward, outward and endpoint powers remains representation dependent; such a division is neither claimed nor needed for this residual closure. The conclusion is specific to the declared model and event. The broader aperture sensitivity, actual-zero branch limitations, model validity and experimental caveats in the manuscript remain applicable.

# Reproduction and provenance

Run the scripts from the residual_mechanism_research directory with Python, NumPy, SciPy, mpmath and Matplotlib. The full recipe is in RUN_REPRODUCTION.sh. compact_airy.py and compact_rs.py implement the identities; analytic_series.py generates the coefficients; study.py and the N14 driver generate the radius predictions; direct_check.py and direct_events.py evaluate the original positive-source integral; high_precision.py uses arbitrary precision angular moments; decomposition.py gives the signed coefficient grouping. JSON and CSV files retain parameters, event roots, slopes, curvatures and control results. Source-code and output hashes are supplied with the verification package.

The previous analytic and manuscript sources are cited by path in source_manifest.json. No archived manuscript or dataset was overwritten. R1-M3, R1-m4 and the associated R1-M6 error accounting are addressed. The data availability statement and historical AI-use facts still require author confirmation; no declarations have been invented.

# Primary sources examined

1. G. A. Siviloglou and D. N. Christodoulides, Accelerating finite energy Airy beams, Optics Letters 32, 979--981 (2007), doi: [10.1364/OL.32.000979](https://doi.org/10.1364/OL.32.000979). Full local paper read, especially Fourier spectrum Eq. (3) and propagated field Eq. (5). Supplies the one-dimensional convolution identity, not the new event coefficients.
2. NIST Digital Library of Mathematical Functions, [10.9.2](https://dlmf.nist.gov/10.9.E2) and [9.5](https://dlmf.nist.gov/9.5), Bessel angular representation and Airy contour identities. Accessed 4 October 2026. These provide the exact transform identities.
3. NIST Digital Library of Mathematical Functions, [9.7.15](https://dlmf.nist.gov/9.7.E15), positive-real Airy inequality. Accessed 4 October 2026. Used for the explicit extension bound.
4. N. K. Efremidis and D. N. Christodoulides, Abruptly autofocusing waves, Optics Letters 35, 4045--4047 (2010), doi: [10.1364/OL.35.004045](https://doi.org/10.1364/OL.35.004045). Full local paper read. Establishes circular Airy autofocusing, but contains no proof of the present proxy half-height residual.
5. T. Geng, M. Li and H. Guo, Orbit-induced localized spin angular momentum of vector circular Airy vortex beam in the paraxial regime, Optics Express 29, 14069--14077 (2021). [Publisher record](https://opg.optica.org/oe/abstract.cfm?uri=oe-29-9-14069). Full local paper read for the radial channels. Its propagation reduction is retained; its local spin interpretation does not itself explain the present finite-radius event minimum.
6. C. Chester, B. Friedman and F. Ursell, An extension of the method of steepest descents, Proceedings of the Cambridge Philosophical Society 53, 599--611 (1957), doi: [10.1017/S0305004100032655](https://doi.org/10.1017/S0305004100032655). Full local paper examined for coalescing saddles and uniform Airy expansion. A finite truncation is not a bound on this event functional; the compact reduction avoids that inference.
