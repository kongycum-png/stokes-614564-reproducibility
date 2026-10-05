# Maxwell completion of a prescribed transverse spectrum (P=1, m>=1)
For the common retained propagating band 0<=u<=umax<kw, let Hq(u) be the real Hankel transform of the same radial input. Let a=m-1, b=m+1, kappa=sqrt((kw)^2-u^2), and let the propagation factor be exp[i kappa z/w] (the common carrier is omitted in stored fields).

With Ex=(Ua exp(ia phi)+Ub exp(ib phi))/2 and Ey=i(Ua exp(ia phi)-Ub exp(ib phi))/2, the longitudinal completion is
Ez= (i/2) exp(im phi) integral u^2 (Hb-Ha)/kappa Jm(u r/w) phase du.
This follows from Ez=i div_t(E_t)/kz and the two Bessel derivative recurrences. Ex and Ey are preserved within the *same stated band*; no transverse projection is applied.

Define
Va=integral u [kappa/(kw) Ha +u^2/(2 kw kappa)(Ha-Hb)] Ja phase du,
Vb=integral u [kappa/(kw) Hb +u^2/(2 kw kappa)(Hb-Ha)] Jb phase du.
Then the scaled magnetic field Z0 H has transverse components
Hx=(-i Va exp(ia phi)+i Vb exp(ib phi))/2,
Hy=(Va exp(ia phi)+Vb exp(ib phi))/2,
and Hz= - exp(im phi)/(2kw) integral u^2(Ha+Hb) Jm phase du.
The normalized axial Poynting density 2 Z0 <S_z> is
Fz=Re(Ua Va*+Ub Vb*)/2.
The transverse electric S3 remains (|Ua|^2-|Ub|^2)/2; it is not replaced by a three-dimensional Stokes parameter.

Report separately: the longitudinal electric fraction in each ROI; maximum local fraction above a declared transverse-intensity floor; integrated flux correction relative to transverse S0; full-plane spectral energy and flux integrals. The finite band avoids the grazing singularity and its omitted power must be quantified. The FFTLog wraparound spectrum at very large u is a numerical artifact and cannot be interpreted as a measured grazing tail. Uniform quadrature including u=0 is used for the retained Parseval power, since the raw logarithmic grid misses the finite q=0 low-frequency interval.
