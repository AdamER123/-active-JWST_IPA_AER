Work for group "Investigating Protostellar Accretion" (PI S. Tom Megeath). The data is "3D" consistent of an image taking at many wavelengths (sometimes called multi-wavelength or hyper-spectral cubes).

"extended" files are initial attempts at distinguishing local min and local max as well as measuring properties throughout images.
"MIRI_13CO_averages" notebook discusses attempts at identifying lines from lab spectral data and matching them in noise.
*.py files generally incorporate pytorch and fitting many spectral lines simultaneously per advice from Dan Watson. This sped up the fitting process for image production from virtually infinite time to ~3 minutes per cube.
NIRSpec_plotter.py helps with plotting 3-color images for those that prefer them
*baseline* and *continuum* use techniques like least squares and non-linear smoothers to best extract a baseline to spectra. The math comes from a library, PyBaselines, which was used for Raman spectroscopy.
The *Paper* work incorporates many methods from basic physical models to apply the spectral measurements and achieve the final paper results.

Publication(s): 
https://ui.adsabs.harvard.edu/abs/2023arXiv231207807R/abstract
https://ui.adsabs.harvard.edu/abs/2023arXiv231003803F/abstract
