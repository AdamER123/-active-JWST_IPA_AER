#A code to fit 1D spectra from baseline-subtracted image cubes from the JWST nirspec ifu
#By Adam E. Rubinstein

# relevant packages you'll need are here, in addition to stable versions of torch and torchimize:
from glob import glob
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from spectral_cube import SpectralCube
import sys

#this code block is for reading in an example data cube, can be turned into functions (see next block)

'''
ALTERNATIVE TO WORK ON (FOR UPDATE)
You can use spectral_cube.spectral_axis.value
'''

# Read in a 3-D IFU datacube of interest, and header
#first, note the path
protostar_folders = ['IRAS16253', 'B335', 'HOPS153', 'HOPS370', 'IRAS20126']
protostar_ind = 0
cube_file_list = [glob('Baseline_Subtracted/' + i + '*.fits')[0].replace('\\', '/') for i in protostar_folders] #change the wildcard '*' here!

# Read in a 3-D IFU datacube of interest, and header
#first, note the path
cube_file = cube_file_list[protostar_ind]
hdul = fits.open(cube_file)
cube = fits.getdata(cube_file)
nirspec_cube = SpectralCube.read(hdul[0]) #accessing the cube for data  
header_cube = hdul[0].header

# define the wavelength grid (microns) from the header
# offset_list = [1.95e-3, 2e-3, 9.75e-4, 1.05e-3, 1.825e-3] #done by hand
offset_list = [1.85e-3, 2e-3, 1.1e-3, 1.1e-3, 2.25e-3] #experimental round 2 with initial changes to baselines
wave_offset = offset_list[protostar_ind] #unit = microns, alt value is 2e-4 microns, while cdelt is about E-9 m or E-3 mic...
wave_factor = 1 #/1.001 #for wavelength calibrating
wave_units = 1e6 #to convert from meters to microns
wave = wave_factor * wave_units * nirspec_cube.spectral_axis.value + wave_offset


# make a simple 1d spectrum of the central region, taking median to attempt to account for cosmic rays
# central_sources = [(48,45), (44,43), (47, 52), (41, 45), (46, 48)]
# central_sources = [(48,46), (44,44), (44, 41), (41, 45), (38, 42)] #this one has fewer artifacts
central_sources = [(46.57459417809592, 45.12978229),  (46.73250708463416, 43.13112798), (46.47088442936513, 46.6279981), (41.71119797770727, 43.61467905), (43.38667807448542, 43.15705917)]

# center_widths = [5, 5, 5, 5, 5]
# flux1 = np.median(cube[:, central_sources[protostar_ind][0]-center_widths[protostar_ind]:central_sources[protostar_ind][1]+center_widths[protostar_ind],\
#                        central_sources[protostar_ind][0]-center_widths[protostar_ind]:central_sources[protostar_ind][1]+center_widths[protostar_ind]], \
#                         axis=(1,2)) #for an example of a single pixel
flux1 = cube[:, int(central_sources[protostar_ind][0]), int(central_sources[protostar_ind][1]) ].astype(np.float32) #for an example of a single pixel
flux1 = np.nan_to_num(flux1)


'''
Now we try fitting multiple lines and spectral features at once with a function at same time as continuum!

After following the same continuum steps, the next steps are:
1) Read in line list for each species. Start with CO, CO2, and atomic lines. 
1a) Line list consists of: Center wavelengths, FWHM, maxima, profiles, that were FIT by Edwin Bogaert et al 
2) Implement LM fitter, and try to fit continuum + line to the observed spectrum. Starting with CO, CO2, and ices should converge quickly.
'''

#reformat and compute ions (aka atomic lines), again, non-simultaneous
line_list_path = 'Line list 2.1 for python.xlsx' #first, note formatting properties from excel file, order: continuum, ices, ions, hydrogen, CO
unres_line_sheets = [1,2,3]
skiprow_list = [None, range(66, 526), range(117, 285)] #IMPORTANT: likely need to modify by hand: range(213, 286) for CO is 3-2; 117 to 213 is 2-1
unres_line_list = [pd.read_excel(line_list_path, sheet_name=unres_line_sheets[i], skiprows=skiprow_list[i]) 
                   for i in range(len(unres_line_sheets))] #read in excel here, separate reads for each sheet is best
unres_line_list = [i[i['Wv, microns']<= max(wave)] for i in unres_line_list] #IMPORTANT: filtering out MIRI wavelengths
# unres_line_list = [i[i['Wv, microns'] >= 3.6] for i in unres_line_list] #IMPORTANT: filtering out wavelengths we can't do yet
wave_col = 'Wv, microns' #all columns: Wv, microns
unres_wavelengths = np.hstack(np.array([i[wave_col].to_numpy(dtype=np.float32) for i in unres_line_list], dtype=object)) #I couldn't figure out a simpler way...

#write out most of the possible useful lists for our case
ion_wavelengths = unres_line_list[0]['Wv, microns'].to_numpy()
hydrogen_wavelengths = unres_line_list[1]['Wv, microns'].to_numpy()
CO_wavelengths = unres_line_list[2]['Wv, microns'].to_numpy()
ion_species = unres_line_list[0]['Species'].to_numpy() #for ions
hydrogen_species = unres_line_list[1]['Species'] + ' v=' + unres_line_list[1]['Transition, v']  + ' ' + unres_line_list[1]['Transition, J']
hydrogen_species = hydrogen_species.to_numpy() #for H2
CO_species = unres_line_list[2]['Species'] + ' v=' + unres_line_list[2]['Transition, v']  + ' ' + unres_line_list[2]['Transition, J']
CO_species = CO_species.to_numpy() #for CO

#and making some connected lists of species and wavelengths (useful for referencing)
import itertools, re
line_wave_list = unres_wavelengths
species_list = list(itertools.chain([re.sub(" ","", i) for i in ion_species], hydrogen_species, CO_species))

#setting up generic functions to fit unresolved lines
#x is the full list of wavelengths, offset is the line's wavelength, and delta is the width of the gaussian
# import jax.numpy as jnp #need to swap this with numpy!
import torch

#onto guesses
amp_guesses = torch.rand(len(unres_wavelengths), dtype=torch.float) #converted to torch
amp_guesses[:len(unres_wavelengths)] = 15
# extra_params = torch.rand(2, dtype=torch.float) #3 baseline, 1 wavelength offset
# wave_min_list = [4.46, 4.452, 4.466, 4.452, 4.442] #where we want to divide the line forest for purpose of fitting a baseline, usually around 4.3 to 4.4 microns
# wave_min = wave_min_list[protostar_ind]
# amp_guesses[-1] = 1.0 + 1e-3 #factor to apply to widths

#pre-calculating some values that are helpful
''' CONSIDERATIONS FOR THE RESOLVING POWER R VS THE FWHM OF GAUSSIANS
Why you might have guessed R was 2700 by accident, but it's not...
For example, sigma = wavelength / 2700
And for a Gaussian, FWHM = 2.35 * sigma = 2.35 * wavelength / 2700
R = lambda / FWHM = lambda / (2.35 * wavelength / 2700) 
R = 2700 / 2.35 = ~1150 
So now to reverse this work...
'''
#taken from https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters
jwst_g395m_table_filename = 'jwst_nirspec_g395m_disp.fits'
hdul = fits.open(jwst_g395m_table_filename)  # open a FITS file
data = hdul[1].data
'''ColDefs(
name = 'WAVELENGTH'; format = 'E'; unit = 'MICRONS'
name = 'DLDS'; format = 'E'; unit = 'MICRONS/PIXEL'
name = 'R'; format = 'E'; unit = 'RESOLUTION'
'''
R_factor = 1.15
jwst_res_spline = UnivariateSpline(data['WAVELENGTH'], data['R']*R_factor)
R = jwst_res_spline(torch.tensor(wave, dtype=torch.float)) #mode for NIRSpec has R ~ 1000 ~ lambda / FWHM ? #1400
FWHM = torch.tensor(wave, dtype=torch.float) / R  #this is delta lambda
sigma = FWHM / np.sqrt(8.0*np.log(2.0)) #property of gaussians
sigma_sq = sigma*sigma #pre-calculating sigma^2

#computing exponential part of gaussian to speed things up
unres_exp = torch.stack([torch.exp(-torch.pow(torch.tensor(wave, dtype=torch.float) - i, 2) / (2 * sigma_sq)).float() \
                            for i in unres_wavelengths], dim=0)

#set up your residual function
#here, we produce a total spectrum to be fit, which takes x=wavelength as input
def residual_spectrum_fit(amp_guesses):
    #need to divide the parameters array by hand, curve_fit only takes 1d param arrays
    amp_unres = amp_guesses[:len(unres_wavelengths)] # torch.tensor(amp_guesses[:len(unres_wavelengths)], dtype=torch.float).flatten()

    '''
    trying to think in 3d
    orig cube = n x m x o, where n = m = 90 and o = 1341 (and there are 5 cubes)
    wave = 1 x 1 x o
    unres_wavelength = p, where p = 158
    amp_guesses = n x m x p
    so for p lines we want to make a sub-cube the size of amp_guesses (ices need separate handling)
    '''
    #take advantage of array shaping and indexing matrices to multiply in parallel (fastest possible)
    # unres_fit = torch.sum(torch.stack([amp_unres[count]*unres_exp[count] for count in range(len(amp_unres))], dim=0), dim=0, dtype=torch.float)
    unres_fit = torch.einsum('i,il->l', amp_unres.float(), unres_exp.float()).float()

    resid = (torch.tensor(flux1, dtype=torch.float) - unres_fit.float()).float()
    return resid

#other option is torchimize sponsored by facebook; I think easier to run since comes with CUDA
#testing torchimize instead: https://github.com/hahnec/torchimize
#thankfully the jacobian comes built in
from torchimize.functions import lsq_lma, gradient_descent, lsq_gna
import time

#profiling code before we get to a complicated step
# see https://docs.python.org/3/library/profile.html for details
# import cProfile
# cProfile.run('lsq_lma(amp_guesses, function=residual_spectrum_fit, ftol=1e-2)')
# sys.exit()
# or to profile an entire script from termiinal: python3 -m cProfile -o NIRSpec_1pix_profile.txt NIRSpec_1pix_lineFit.py
# to convert, see pstats (https://stackoverflow.com/questions/8283112/cprofile-saving-data-to-file-causes-jumbles-of-characters)
# https://github.com/baverman/flameprof for plotting

# compiled_spectral_fitter = torch.compile(lsq_lma)
fit_method = 'gradient_descent'
t0 = time.time()
coeffs_list = gradient_descent(amp_guesses, function=residual_spectrum_fit, ftol=1e-5, gtol=1e-5, ptol=1e-5)
# coeffs_list = lsq_lma_parallel(amp_guesses, function=residual_spectrum_fit, ftol=1e-2)
t1 = time.time()
print('start: ', t0, ', end: ', t1, ', diff: ', t1-t0)


#redistribute fluxes for select lines, tough to do, so doing this by hand
#for [FeII] at 4.89, the contaminting line is 
#for H2 v=0, S8, the contaminting line is CO v=1, P37

# contaminted_line_list = [4.8891387, 5.053115]
# co_contaminant_list = ['P 23', 'P 37']
# for i in range(len(contaminted_line_list)):
#     #locate relevant indices for contaminated lines
#     contaminated_wave_ind = np.where(line_wave_list == [j for j in line_wave_list if j > contaminted_line_list[i]-0.0001 and j < contaminted_line_list[i]+0.0001])[0][0]
#     co_ind = species_list.index([j for j in species_list if co_contaminant_list[i] in j][0])

#     #compute reference values for the amount of contamination
#     co_average = (coeffs_list[-1][co_ind-1] + coeffs_list[-1][co_ind+1])/2.0 #take advantage that the average of the co line should change continuously
#     contaminated_line_leak = coeffs_list[-1][co_ind] - co_average

#     # and assign them to the appropriate index in our amplitude cube
#     coeffs_list[-1][co_ind] = co_average
#     coeffs_list[-1][contaminated_wave_ind] += contaminated_line_leak


#reconstruct fitted spectrum for convenience of plotting
def total_spectrum_forplot(amp_guesses):
    #need to divide the parameters array by hand, curve_fit only takes 1d param arrays
    amp_unres = amp_guesses[:len(unres_wavelengths)] # torch.tensor(amp_guesses[:len(unres_wavelengths)], dtype=torch.float).flatten()

    '''
    trying to think in 3d
    orig cube = n x m x o, where n = m = 90 and o = 1341 (and there are 5 cubes)
    wave = 1 x 1 x o
    unres_wavelength = p, where p = 158
    amp_guesses = n x m x p
    so for p lines we want to make a sub-cube the size of amp_guesses (ices need separate handling)
    '''
    #take advantage of array shaping and indexing matrices to multiply in parallel (fastest possible)
    # unres_fit = torch.sum(torch.stack([amp_unres[count]*unres_exp[count] for count in range(len(amp_unres))], dim=0), dim=0, dtype=torch.float)
    unres_fit = torch.einsum('i,il->l', amp_unres.float(), unres_exp.float()).float()

    return torch.Tensor.numpy(unres_fit) #summing along 0 axis to compress profiles into single arrays, this time output to numpy for convenience

template_fit = total_spectrum_forplot(coeffs_list[-1]) #calling function for plottable version of fit

#sample plot to check our fit
fig, (ax, ax_resid) = plt.subplots(nrows=2, ncols=1, figsize=(15,10), sharex=True, gridspec_kw={'height_ratios': [3, 1]}) #setup fig, axes
ax.plot(wave, flux1, color='xkcd:grassy green', label='Baseline-Subtracted Line Emission')
# ax.scatter(wave[continuum_matches], flux1[continuum_matches], color='k', marker='o', s=35, zorder=10, label='Measured Continuum') #plot the measured anchor points
ax.plot(wave, template_fit, 'goldenrod', label='Fitted Lines', linestyle='--')
ax.axhline(0, color='k', linestyle=':') #a horizontal line at zero
ax.tick_params(axis='both', which='major', labelsize=25)

#add to this a residual plot
ax_resid.plot(wave, flux1 - template_fit, color='xkcd:twilight blue', label='Residual', zorder=100)
ax_resid.axhline(0, color='k', linestyle=':') #a horizontal line at zero
ax_resid.set_xlabel(r'$\rm \lambda \ (\mu m)$', fontsize=40)
ax_resid.tick_params(axis='both', which='major', labelsize=25)

#general formatting plot
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax_resid.get_legend_handles_labels())] #https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
ax.legend(handles, labels, loc='best', fontsize=25)
# ax.set_xlim(min(wave), max(wave))
# ax.set_yscale('log')
# ax.set_ylim(min(flux1), max(flux1)) #if doing log, then you need min = about 1e-1
# ax.set_xlim(4.3, max(wave))
# ax.set_xticks(np.arange(4.3, 5.25, 0.1))
# ax.set_ylabel('Intensity (MJy/sr)', fontsize=40)
# ax.set_title(protostar_folders[protostar_ind] + ' Central Source, Median Filtered, Baseline Subtracted', fontsize=30) #cube_file.split('/')[0] = continuum_subtracted

# fig.savefig('Spectra1D_tests/' + protostar_folders[protostar_ind] + '_Spectrum1D_basefit_offset' + str(wave_offset) + '_Rinterp' + str(R_factor) + '_' + fit_method + '.pdf')
fig.savefig('CentralSource_CO/' + protostar_folders[protostar_ind] + '_Spectrum1D_offset' + str(wave_offset) + '_Rinterp' + str(R_factor) + '_' + fit_method + '.pdf')