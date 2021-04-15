# Walkthrough

## Basic Overview
We're given an absorption spectra of a nanoparticle, our job is to find out all the structural parameters of the nanoparticle. The structural parameters are radii of inner and outer shell of the nanoparticle (r1, r2), and permittivity values (or materials) of the three layers in between (e1, e2, e3).
<br/>
Throughout the experiment, A_abs is plotted w.r.t. wavelength with an wavelength range of 180nm to 1940nm with 1nm interval, containing 1761 samples of A_abs. The input dimension of our model is thus 1761.

## Terms

lambd &rarr; Wavelength (fixed throughout experiment) <br/>
lambd = [180, 181, ......, 1940] <br/>

x &rarr; Input spectra [samples of A_abs] <br/>
y &rarr; Ground truth radii values [r1, r2] <br/>
y_pred &rarr; Predicted radii values [r1_pred, r2_pred] <br/>
x_pred &rarr; Predicted Spectra [samples of A_abs_pred] <br/>
