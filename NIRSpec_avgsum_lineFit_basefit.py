#A code to fit spectral lines for a 1D spectrum extracted via apertuer sum over baseline-subtracted cubes from the JWST nirspec ifu
#By Adam E. Rubinstein

# relevant packages you'll need are here, in addition to stable versions of torch and torchimize:
from glob import glob
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import sys

#this code block is for reading in an example data cube, can be turned into functions
#listing all protostars
protostar_folders = ['16253'] #, 'B335', '153', '370', '20126']
protostar_ind = 0
# cube_file_list = [glob('Baseline_Subtracted/' + i + '*.txt')[0].replace('\\', '/') for i in protostar_folders] #change the wildcard '*' here!
cube_file_list = glob('Baseline_Subtracted/*' + protostar_folders[0] + '*.txt')[0].replace('\\', '/')

# use data from reading in the simple .txt file with columns wavelength, intensity
# cube_file = cube_file_list[protostar_ind]
data = np.genfromtxt(cube_file_list, dtype=float, delimiter=',', skip_header=1)
wave_list = data[:, 0]
flux_list = data[:, 1]

# define the wavelength grid (microns)
#[FeI] - CO v10 P23 = 4.8891387 - 4.886926 = 0.002127 mic
# offset_list = [1.825e-3, 2e-3, 9.75e-4, 1.05e-3, 1.825e-3] #done by hand
# offset_list = [7.5e-4, 0.0002, -0.0005, -0.0006, 0.00045] #experimental including baseline fits with some influence by dan
offset_list = [1e-4, 1e-3, 1e-3, 1e-3, 1e-3] #experimental
# offset_list = [1.85e-3, 2e-3, 1.1e-3, 1.1e-3, 2.25e-3]  #best ones I found so far
wave_offset = offset_list[protostar_ind] #unit = microns, alt value is 2e-4 microns, while cdelt is about E-9 m or E-3 mic...
wave_units = 1e6 #if you need to convert from meters to microns
# wave = wave_units * jwst_cube.spectral_axis.value + wave_offset
wave_list -= wave_offset

'''
Now we try fitting multiple lines and spectral features at once with a function at same time as continuum!

After following the same continuum steps, the next steps are:
1) Read in line list for each species. Start with CO, CO2, and atomic lines. 
1a) Line list consists of: Center wavelengths, FWHM, maxima, profiles, that were FIT by Edwin Bogaert et al 
2) Implement LM fitter, and try to fit continuum + line to the observed spectrum. Starting with CO, CO2, and ices should converge quickly.
'''

#reformat and compute ions (aka atomic lines), again, non-simultaneous
line_list_path = 'Line list 2.2 for python.xlsx' #first, note formatting properties from excel file, order: continuum, ices, ions, hydrogen, CO

unres_line_sheets = [1,2,3]
skiprow_list = [None, range(66, 526), None] #IMPORTANT: likely need to modify by hand: range(213, 286) for CO is 3-2; 117 to 213 is 2-1
unres_line_list = [pd.read_excel(line_list_path, sheet_name=unres_line_sheets[i], skiprows=skiprow_list[i]) 
                   for i in range(len(unres_line_sheets))] #read in excel here, separate reads for each sheet is best
unres_line_list = [i[i['Wv, microns']<= max(wave_list)] for i in unres_line_list] #IMPORTANT: filtering out MIRI wavelengths
# unres_line_list = [i[i['Wv, microns'] >= 3.6] for i in unres_line_list] #IMPORTANT: filtering out wavelengths we can't do yet
wave_col = 'Wv, microns' #all columns: Wv, microns
unres_wavelengths = np.hstack(np.array([i[wave_col].to_numpy(dtype=np.float64) for i in unres_line_list], dtype=object)) #I couldn't figure out a simpler way...

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
amp_guesses[:len(unres_wavelengths)] = 2.5e-17

#trying to make a more accurate initial guess based on probing the spectrum around the wavelength for each line
# central_sources = [(47,45), (45,43), (42, 45), (42, 38), (43, 43)] #based on line to cont ratio
# amp_numpy = np.array([2.0 * jwst_cube._data[ np.abs(wave - i).argmin(), \
#                                                central_sources[protostar_ind][0], central_sources[protostar_ind][1]] for i in unres_wavelengths])
# amp_guesses = torch.from_numpy(amp_numpy)

#trying fits with more parameters
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
R_factor = 1.5
jwst_res_spline = UnivariateSpline(data['WAVELENGTH'], data['R']*R_factor)
R = 1550 # jwst_res_spline(torch.tensor(wave_list, dtype=torch.float)) #mode for NIRSpec has R ~ 1000 ~ lambda / FWHM
FWHM = torch.tensor(wave_list, dtype=torch.float) / R  #this is delta lambda
# ''' (currently not working)
# For the very brightest lines, we want to fix the FWHM or sigma by hand (not R!)...including...
# HI BrA 4.052, FeII 4.115, H2 S10 4.409, FeII 4.434, HeI 4.6066, HI PfB 4.653, H2 S9 4.69, FeII 4.889, H2 v=0 S8 5.053
# '''
# bright_line_species = ['HI(BrA)', 'FeII', 'H2 v=0 - 0 S 10', 'FeII', 'HeI', 'HI(PfB)', 'H2 v=0 - 0 S 9', 'FeII', 'H2 v=0 - 0 S 8']
# bright_line_wave = [4.0522694, 4.1149943, 4.40979113, 4.4348337, 4.6066, 4.65378, 4.694613954, 4.8891388, 5.053115061]
# bright_line_fwhm = 0.004
# for i in range(len(bright_line_species)):
#     for j in range(len(species_list)):
#         if bright_line_species[i] == species_list[j] and line_wave_list[j] > bright_line_wave[i]-0.0001 and \
#         line_wave_list[j] < bright_line_wave[i]+0.0001: #search by label for the line we can measure and check as well as by line
#             FWHM[j] = bright_line_fwhm
sigma = FWHM / np.sqrt(8.0*np.log(2.0)) #property of gaussians
sigma_sq = sigma*sigma #pre-calculating sigma^2

#computing exponential part of gaussian to speed things up (see other scripts if more intuitive)
unres_exp = torch.stack([torch.exp(-torch.pow(torch.tensor(wave_list, dtype=torch.float) - i, 2) / (2 * sigma_sq)).float() \
                             for i in unres_wavelengths], dim=0)


#set up your residual function for purpose of fitting
def residual_spectrum_fit(amp_guesses, jwst_spectrum=torch.tensor([])):
    #here, we produce a total spectrum to be fit, which takes x=wavelength as input
    #need to divide the parameters array by hand, curve_fit only takes 1d param arrays
    amp_unres = amp_guesses[:len(unres_wavelengths)]

    '''
    trying to think in 3d
    orig cube = n x m x o, where n = m = 90 and o = 1341 (and there are 5 cubes)
    wave = 1 x 1 x o
    unres_wavelength = p, where p = 158
    amp_guesses = n x m x p
    so for p lines we want to make a sub-cube the size of amp_guesses (ices need separate handling)
    '''
    #this one takes advantage of array shaping and matrices
    # basically sum[A_ijk B_il] over i => C_ljk (formally may require writing an operator to rotate indices, but that's done implicitly here)
    unres_fit = torch.einsum('i,il->l', amp_unres.float(), unres_exp.float()).float()

    return ((jwst_spectrum).float() - unres_fit.float()).float()  #depending on your other function, need to do *params


#testing torchimize instead: https://github.com/hahnec/torchimize
#thankfully the jacobian comes built in
from torchimize.functions import gradient_descent #lsq_lma

#profiling code before we get to a complicated step, in case you want to check your run time and failure points...
# see https://docs.python.org/3/library/profile.html for details
# import cProfile
# cProfile.run('lsq_lma(amp_guesses, function=residual_spectrum_fit, ftol=1e-2)')
# sys.exit()
# or to profile an entire script from termiinal: python3 -m cProfile -o NIRSpec_1pix_profile.txt NIRSpec_1pix_lineFit.py
# to convert, see pstats (https://stackoverflow.com/questions/8283112/cprofile-saving-data-to-file-causes-jumbles-of-characters)
# https://github.com/baverman/flameprof for plotting

#open and read a data file
jwst_data = torch.from_numpy(np.nan_to_num(np.array(flux_list, dtype=np.float64)))
amp_cube = torch.from_numpy(np.zeros(len(amp_guesses))) #assume square, initialize as zeros

#fitting line amplitudes
#note format of arrays is wavelength, x, y
import time
t0 = time.time()
coeffs_list = gradient_descent(amp_cube[:].float(), function=residual_spectrum_fit, args=(jwst_data.float(),), ftol=1e-5, gtol=1e-5, ptol=1e-5) #, l=1.0) #fit lines for cube
t1 = time.time()
print('start: ', t0, ', end: ', t1, ', diff: ', t1-t0)

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
fig, ax = plt.subplots(figsize=(15,10)) #setup fig, axes #make a figure to plot various locations on an image
ax.plot(wave_list, flux_list, color='xkcd:grassy green', label='Baseline-Subtracted')
ax.plot(wave_list, template_fit, 'goldenrod', label='Fitted Lines', linestyle='--')
ax.axhline(0, color='k', linestyle=':') #a horizontal line at zero
ax.tick_params(axis='both', which='major', labelsize=25)

#add to this a residual plot
ax.plot(wave_list, flux_list - template_fit, color='xkcd:twilight blue', label='Residual', zorder=100)

#general formatting plot
ax.legend(loc='best', fontsize=25)
# ax.set_xlim(min(wave_list), max(wave_list))
# ax.set_yscale('log')
# ax.set_ylim(min(flux1), max(flux1)) #if doing log, then you need min = about 1e-1
ax.set_xlim(4.3, max(wave_list))
ax.set_xticks(np.arange(4.3, 5.25, 0.1))
ax.set_ylabel(r'$\rm Flux \ (erg \ {cm}^{-2} \ {s}^{-1})$', fontsize=40)
ax.set_xlabel(r'$\rm \lambda \ (\mu m)$', fontsize=40)
# ax.set_title(protostar_folders[protostar_ind] + ' Central Source, Median Filtered, Baseline Subtracted', fontsize=30) #cube_file.split('/')[0] = continuum_subtracted

fig.savefig('Spectra1D_tests/' + protostar_folders[protostar_ind] + '_Spectrum1D_basefit_offset' + str(wave_offset) + '_Rinterp' + str(R_factor) + '.pdf')
# fig.savefig('CentralSource_CO/' + protostar_folders[protostar_ind] + '_Spectrum1D_offset' + str(wave_offset) + '_Rinterp' + str(R_factor) + '_' + fit_method + '.pdf')