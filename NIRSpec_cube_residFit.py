#A code to fit spectral cubes from baseline-subtracted image cubes from the JWST nirspec ifu
#By Adam E. Rubinstein

# relevant packages you'll need are here, in addition to stable versions of torch and torchimize:
from glob import glob
from astropy.io import fits
from astropy import units as u
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from spectral_cube import SpectralCube
import sys

#this code block is for reading in an example data cube, can be turned into functions
#listing all protostars
protostar_folders = ['IRAS16253', 'B335', 'HOPS153', 'HOPS370', 'IRAS20126']
protostar_ind = 4
cube_file_list = [glob('Cube_Fit_Results/' + i + '*resid.fits')[0].replace('\\', '/') for i in protostar_folders] #change the wildcard '*' here!

# Read in a 3-D IFU datacube of interest, and header
#first, note the path
cube_file = cube_file_list[protostar_ind]
hdul = fits.open(cube_file)
jwst_cube = SpectralCube.read(hdul[0]) #accessing the cube for data  

# define the wavelength grid (microns)
#[FeI] - CO v10 P23 = 4.8891387 - 4.886926 = 0.002127 mic
# offset_list = [1.825e-3, 2e-3, 9.75e-4, 1.05e-3, 1.825e-3] #done by hand
offset_list = [1.85e-3, 2e-3, 1.1e-3, 1.1e-3, 2.25e-3]  #experimental
# offset_list = [7.5e-5, 0.0002, -0.0005, -0.0006, 0.00045] #experimental including baseline fits with some influence by dan
wave_offset = offset_list[protostar_ind] #unit = microns, alt value is 2e-4 microns, while cdelt is about E-9 m or E-3 mic...
wave_units = 1e6 #if you need to convert from meters to microns
wave = wave_units * jwst_cube.spectral_axis.value + wave_offset

'''
Now we try fitting multiple lines and spectral features at once with a function at same time as continuum!

After following the same continuum steps, the next steps are:
1) Read in line list for each species. Start with CO, CO2, and atomic lines. 
1a) Line list consists of: Center wavelengths, FWHM, maxima, profiles, that were FIT by Edwin Bogaert et al 
2) Implement LM fitter, and try to fit continuum + line to the observed spectrum. Starting with CO, CO2, and ices should converge quickly.
'''

#reformat and compute ions (aka atomic lines), again, non-simultaneous
line_list_path = 'Line list 2.1 for python.xlsx' #first, note formatting properties from excel file, order: continuum, ices, ions, oh, CO
unres_line_sheets = [3]
skiprow_list = [range(1,117)] #IMPORTANT: likely need to modify by hand: range(213, 286) for CO is 3-2; 117 to 213 is 2-1
unres_line_list = [pd.read_excel(line_list_path, sheet_name=unres_line_sheets[i], skiprows=skiprow_list[i]) 
                   for i in range(len(unres_line_sheets))] #read in excel here, separate reads for each sheet is best
unres_line_list = [i[i['Wv, microns']<= max(wave)] for i in unres_line_list] #IMPORTANT: filtering out MIRI wavelengths
# unres_line_list = [i[i['Wv, microns'] >= 3.6] for i in unres_line_list] #IMPORTANT: filtering out wavelengths we can't do yet
wave_col = 'Wv, microns' #all columns: Wv, microns
unres_wavelengths = np.hstack(np.array([i[wave_col].to_numpy(dtype=np.float32) for i in unres_line_list], dtype=object)) #I couldn't figure out a simpler way...

#write out most of the possible useful lists for our case 
CO_wavelengths = unres_line_list[0]['Wv, microns'].to_numpy()
CO_species = unres_line_list[0]['Species'] + ' v=' + unres_line_list[0]['Transition, v']  + ' ' + unres_line_list[0]['Transition, J']
CO_species = CO_species.to_numpy() #for CO
# oh_wavelengths = unres_line_list[1]['Wv, microns'].to_numpy()
# oh_species = unres_line_list[1]['Species'] + ' v=' + unres_line_list[1]['Transition, v']  + ' ' + unres_line_list[1]['Transition, J']
# oh_species = oh_species.to_numpy() #for H2


#and making some connected lists of species and wavelengths (useful for referencing)
import itertools, re
line_wave_list = unres_wavelengths
species_list = CO_species #list(itertools.chain([re.sub(" ","", i) for i in oh_species], CO_species))

#setting up generic functions to fit unresolved lines
#x is the full list of wavelengths, offset is the line's wavelength, and delta is the width of the gaussian
# import jax.numpy as jnp #need to swap this with numpy!
import torch

#onto guesses
amp_guesses = torch.rand(len(unres_wavelengths), dtype=torch.float) #converted to torch
amp_guesses[:len(unres_wavelengths)] = 5
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
R = 1200 #jwst_res_spline(torch.tensor(wave, dtype=torch.float)) #mode for NIRSpec has R ~ 1000 ~ lambda / FWHM
FWHM = torch.tensor(wave, dtype=torch.float) / R  #this is delta lambda
sigma = FWHM / np.sqrt(8.0*np.log(2.0)) #property of gaussians
sigma_sq = sigma*sigma #pre-calculating sigma^2

#computing exponential part of gaussian to speed things up (see other scripts if more intuitive)
unres_exp = torch.stack([torch.exp(-torch.pow(torch.tensor(wave, dtype=torch.float) - i, 2) / (2 * sigma_sq)).float() \
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
    # basically sum[Aijk Bil] over i = Cljk (perhaps requiring an operator to rotate indices)
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
jwst_data = torch.from_numpy(np.nan_to_num(np.array(jwst_cube._data, dtype=np.float32))) #need to take out nans, otherwise things break
amp_cube = torch.from_numpy(np.zeros((len(amp_guesses), jwst_data.shape[1], jwst_data.shape[2]))) #assume square, initialize as zeros
hdul.close()

#fitting line amplitudes
#note format of arrays is wavelength, x, y
import time
stacked_coeffs = torch.from_numpy(np.zeros(amp_cube.shape)).float()
t0 = time.time()
for row in range(jwst_data.shape[1]):
    for col in range(jwst_data.shape[2]):
        coeffs_list = gradient_descent(amp_cube[:, row, col].float(), function=residual_spectrum_fit, args=(jwst_data[:, row, col].float(),), ftol=1e-5, gtol=1e-5, ptol=1e-5, l=1.0) #fit lines for cube
        stacked_coeffs[:, row, col] = coeffs_list[-1].float()
t1 = time.time()
print('start: ', t0, ', end: ', t1, ', diff: ', t1-t0)

#pre-corrections and initial setup to check fitting process...
amp_unres_round1 = stacked_coeffs[:len(unres_wavelengths), :, :] #with the coeffs fit and saved, use coeffs_list in our cube as the amp_cube now
unres_fit_round1 = torch.einsum('ijk,il->ljk', amp_unres_round1.float(), unres_exp.float()).float()
protostar_id = protostar_folders[protostar_ind] + '_residFit' # cube_file.split('/')[-1][:-5] # for reference later when saving files

#with the coeffs, use coeffs_list[-1] as the amp_cube now
amp_unres = stacked_coeffs[:len(unres_wavelengths), :, :]

#saving cube of amplitudes
from astropy.wcs import WCS
amp_header_copy = jwst_cube.header #editing header beforehand in memory because it isn't right, need to separately define by saving the wavelength list
amp_header_copy['CRVAL3'] = 0
amp_header_copy['CDELT3'] = 1
amp_cube = SpectralCube(data=stacked_coeffs*u.MJy/u.sr, wcs=WCS(amp_header_copy))
amp_cube_savepath = 'Cube_Fit_Results/'
amp_cube_name = protostar_id + '_amp.fits'
amp_cube.write(amp_cube_savepath+amp_cube_name, format='fits', overwrite=True)
print('Saved: ', amp_cube_savepath+amp_cube_name)

#need a line list to interpret
line_wave_filename = 'residFit_amp_wavelength_list.txt'
with open(amp_cube_savepath + line_wave_filename, 'w') as f:
    f.write('Line Number, Wavelength (um), Species \n') #write column header
    for count, (wavelength, line_name) in enumerate(zip(line_wave_list, species_list)):
            f.write(str(count) + ',' + str(wavelength) + ',' + line_name + '\n') #write wavelengths

#reading in original line fits to check subtraction...
cube_file_list = [glob('Ices/' + i + '*_GasLines_corrected.fits')[0].replace('\\', '/') for i in protostar_folders] #change the wildcard '*' here!
orig_path = cube_file_list[protostar_ind]
hdul = fits.open(orig_path)
orig_cube = SpectralCube.read(hdul[0]) #accessing the cube for data  
orig_line_fit = orig_cube._data[:, :, :]

#also gets us the cube of pure line emission; BEFORE flux redistribution...primarily for leiden group -> Ices directory
total_gas_fits = orig_line_fit + unres_fit_round1.numpy()
precorrec_cube = SpectralCube(data=total_gas_fits*u.MJy/u.sr, wcs=jwst_cube.wcs)
precorrec_cube_savepath = 'Ices/'
precorrec_cube_name = protostar_id + '_GasLines_corrected_V2.fits'
precorrec_cube.write(precorrec_cube_savepath+precorrec_cube_name, format='fits', overwrite=True)
print('Saved: ', precorrec_cube_savepath+precorrec_cube_name)


#reading in original data files to check subtraction...
cube_file_list = [glob(i + '/' + '*.fits')[0].replace('\\', '/') for i in protostar_folders] #change the wildcard '*' here!
orig_path = cube_file_list[protostar_ind]
hdul = fits.open(orig_path)
orig_cube = SpectralCube.read(hdul[1]) #accessing the cube for data  
orig_data = orig_cube._data[1:-1, :, :] #need to invoke original cutoff...
err_data = hdul['ERR'].data[1:-1, :, :] #needed for chi^2...converting to torch 

# #first, saving line-subtracted cubes; to be done BEFORE flux redistribution...primarily for leiden group -> Ices directory
noline_data = orig_data - total_gas_fits
noline_cube = SpectralCube(data=noline_data*u.MJy/u.sr, wcs=jwst_cube.wcs)
noline_cube_savepath = 'Ices/'
noline_cube_name = protostar_id + '_noGasLines_V2.fits'
noline_cube.write(noline_cube_savepath+noline_cube_name, format='fits', overwrite=True)
print('Saved: ', noline_cube_savepath+noline_cube_name)



#to save residuals and comparisons, we need something that sums over all lines at a given wavelength, like we did for purpose of fitting
#redo einsum here b/c we want a different summation...
#basically sum[Aijk Bil] over i = Cljk (perhaps requiring an operator to rotate indices)
unres_fit = torch.einsum('ijk,il->ljk', amp_unres.float(), unres_exp.float()).float()

#saving residuals
residual_data = jwst_data - unres_fit #determining residual, careful to sum along wavelength axis
resid_cube = SpectralCube(data=residual_data*u.MJy/u.sr, wcs=jwst_cube.wcs)
resid_cube_savepath = 'Cube_Fit_Results/'
resid_cube_name = protostar_id + '_resid_V2.fits'
resid_cube.write(resid_cube_savepath+resid_cube_name, format='fits', overwrite=True)
print('Saved: ', resid_cube_savepath+resid_cube_name)

#a new fits file to be saved, maybe viewed in DS9
# may need to modify the function for a particular image, but generally useful if you want to modify lots of images with a few variations (e.g. headers)
def fits_saver(array, wcs_header, name, save_path):
    '''
    array is a 2d array of data - could be from reprojecting one image onto another or from convolution...
    wcs_header is a header containing the wcs coords of the image that we projected onto or of the orig image (if from the convolution)
    name can be the path to some image you're using. It will get string split at the / character, and the func only takes the last element of that splitting
    save_path is the folder you want to save to...recommended to also add something to the start of the images names to make it clear what you did to them (e.g. 'Regridded/regrid_')
    '''
    
    #just setup an fits HDU from the data
    hdu_new = fits.PrimaryHDU(array, header=wcs_header)
    hdul = fits.HDUList([hdu_new])
    
    #saving the file
    hdul.writeto(save_path+name, overwrite=True)     
    return (save_path+name)

#a chi^2 like quantity
#note chi^2 here! to be precise, this goodness of fit is determined using reduced chi^2, 
# this is LIKE the pearson's chi^2 test statistic, but that formally is dependent on number of counts in a distribution
#the general concept is we are trying to measure our squared residuals relative to our variance
#(this both eliminates any negatives in our sum and gives us a "fair" comparison against the noise)
#then traditionally you scale a factor for the number of parameters you needed relative to the amount of data you're fitting (to be a bit more generous)
chi2_fits_savepath = 'Cube_Fit_Results/'
# flux_with_snr = 0.2 * jwst_data
len_first_fit = 181 #double check this number or omit subtracting N_guesses
chi2_like = (np.sum((residual_data.numpy())**2. / (err_data)**2., axis=0) ) / (len(err_data) - len(amp_guesses) - len_first_fit) #for now, I assume a 10% uncertainty on my fitted values...note the len of smooth list and -2 comes from the fixed parameters involved in fitting
chi2_fits_name = protostar_id + '_chi2_unc=' + 'errMap_V2.fits' # +str(flux_snr)+'_flux.fits'
fits_path = fits_saver(chi2_like, WCS(jwst_cube.header).to_header(), chi2_fits_name, chi2_fits_savepath) #saving
print('Saved: ', fits_path) #I use to confirm file has right path


#redistribute fluxes for select lines, tough to do, so doing this by hand
#for [FeII] at 4.89, the contaminting line is 
#for H2 v=0, S8, the contaminting line is CO v=1, P37
#the full list of contaminted lines...
# H2 v=0 S18 # H2 v=0 S14 # HI BrA # HI # H2 v=0 S10 # H2 v=1 - 1 S 11 # FeII # H2 v=1 - 0 O 9 # HeI # U1
# HI # H2  v=1 - 1 S 10 # H2 v=0 S9  # FeII # H2 v=1-1 S9 # H2 v=0 - 0 S 8 # Either HI or something molecular
# contaminated_line_list = [3.4378672, 3.723689, \
#                          4.054046, 4.376458, 4.40979113, 4.4166107, 4.4348335, 4.575481, 4.6066, 4.618, 4.65378, 4.694613954, 4.8891387, 4.954, \
#                             5.053115, 5.12865] #these are the lines that are affected (only noted by wavelength, easier to look up)
# contaminant_list = ['H2 v=0 - 0 S 18', 'H2 v=0 - 0 S 14', 'H2 v=2 - 1 O 7', \
#                     'R 48', 'R 39', 'R 38', 'R 34', 'R 11', 'R 6', 'R 5', 'R 1', \
#                         'P 3', 'P 23', 'P 29', 'P 37', 'P 43'] #the line that we consider to be "contaminating" our line (really just easiest to check values for)
# co_preContam_list = ['H2 v=0 - 0 S 17',  'H2 v=0 - 0 S 13', 'H2 v=2 - 1 O 5', \
#                      'R 47', 'R 37', 'R 37', 'R 33', 'R 10', 'R 4', 'R 3', 'R 0', \
#                         'P 2', 'P 22', 'P 28', 'P 36', 'P 42', ] #the line before or neighboring the contaminator
# co_postContam_list = ['H2 v=0 - 0 S 19', 'H2 v=0 - 0 S 15', 'H2 v=2 - 1 O 9', \
#                       'R 49', 'R 41', 'R 39', 'R 35', 'R 12', 'R 8', 'R 7', 'R 2', \
#                         'P 4', 'P 24', 'P 30', 'P 38', 'P 44'] #the line in the series after our contaminator

# redist_stacked_coeffs = torch.clone(stacked_coeffs) #make sure before looping to save this as a separate variable

# #looping through our contaminators
# for i in range(len(contaminated_line_list)):
#     #locate relevant indices for contaminated lines
#     line_ind = np.where(line_wave_list == [j for j in line_wave_list if j > contaminated_line_list[i]-0.0001 and j < contaminated_line_list[i]+0.0001])[0][0] #search line with overlap by numerical wavelength
#     contam_ind = species_list.index([j for j in species_list if contaminant_list[i] in j][0]) #search by label for the line we can measure and check
#     print('Checking for Flux Redistribution: ', line_ind, line_wave_list[line_ind], contam_ind, species_list[contam_ind], line_wave_list[contam_ind])

#     #compute reference values for the amount of contamination
#     contam_ind_pre = species_list.index([j for j in species_list if co_preContam_list[i] in j][0]) #search by label
#     contam_ind_post = species_list.index([j for j in species_list if co_postContam_list[i] in j][0]) #search by label
#     contam_average = (redist_stacked_coeffs[contam_ind_pre,:,:] + redist_stacked_coeffs[contam_ind_post,:,:])/2.0 #take advantage that the average of the co line should change continuously
#     contaminated_line_leak = redist_stacked_coeffs[contam_ind,:,:] - contam_average #take average out of the relevant line BEFORE REASSIGNMENT

#     #check if the line would be negative. if so, need to instead set it to just the amount of leakage (even if small, should be positive)
#     #first case is if the sum is positive, just add them (simple)
#     redist_stacked_coeffs[line_ind,:,:] = torch.where(redist_stacked_coeffs[line_ind,:,:] + contaminated_line_leak >= 0, \
#                                                       redist_stacked_coeffs[line_ind,:,:] + contaminated_line_leak, redist_stacked_coeffs[line_ind,:,:])
#     #next case is if one is negative, set to 0
#     redist_stacked_coeffs[line_ind,:,:] = torch.where(redist_stacked_coeffs[line_ind,:,:] < 0, \
#                                                       0, redist_stacked_coeffs[line_ind,:,:])
#     #last case is if one is positive but the original is negative, then just use the leakage (worst case noise?)
#     redist_stacked_coeffs[line_ind,:,:] = torch.where(contaminated_line_leak > redist_stacked_coeffs[line_ind,:,:], \
#                                                       contaminated_line_leak, redist_stacked_coeffs[line_ind,:,:])
#     #(implicit extra case is if the original positive but leakage negative, ignore the leakage...)
#     #once we've done all this, then we're safe to reassign the contaminated index with the average value...
#     redist_stacked_coeffs[contam_ind,:,:] = contam_average #assign the average to the CO index now that we've stored the varaible

#slice cube for each summed line and save 
#basically sum[Aijk Bil] over i = Cljk (perhaps requiring an operator to rotate indices)
# redist_unres_fit = torch.einsum('ijk,il->ljk', amp_unres.float(), unres_exp.float()).float() # torch.einsum('ijk,il->iljk', redist_stacked_coeffs[:len(unres_wavelengths), :, :].float(), unres_exp.float()).float()
# line_profile_cube = torch.clone(redist_unres_fit) #filling in cube of lines, producing 2D slices by summing along axis for each line

# #now trying to save profiles given the FWHM for each line
# profile_savepath = 'Line_Profiles/' #set folder
# prof_header_copy = jwst_cube.header #editing header beforehand in memory because it isn't right, need to separately define by saving the wavelength list
# prof_header_copy['CUNIT3'] = 'um'
# R_numpy = jwst_res_spline(line_wave_list) #remaking R and FWHM for line list and appropriate datatype
# FWHM_numpy = line_wave_list / R_numpy  #this is delta lambda
# fwhm_buffer = 2 #just in case you want a bit of a buffer around the line...

# #reading in continuum for ratios
# ratio_savepath = 'Line_Cont_Ratios/'
# #except the line needs to be diminished by the ices
# # ...if I do a smooth polynomial continuum, then the ices need to diminish the lines! but could divide by the baseline?
# baseline_path = 'Baseline/' + protostar_folders[protostar_ind] + '_NIRspec_cube_tophat_jcbd.fits' # + '_NIRspec_cube_pspline_asls_cont.fits'
# hdul = fits.open(baseline_path)
# baseline_cube = SpectralCube.read(hdul[0]) #accessing the cube for data  
# baseline_data = baseline_cube._data

# #looping through and saving profiles
# for count, (wavelength, line_name, channel_width) in enumerate(zip(line_wave_list, species_list, FWHM_numpy)):  
#     #for each slice of a cube...
#     #if you'd like to mod this to convert a profile to a saved cube for each line, you might want this:
#     line_ind = np.argmin(np.abs(wave - wavelength)) #find index from line's lambda
#     pos_fwhm_ind = np.argmin(np.abs(wave - wavelength - 2.*channel_width)) + fwhm_buffer #indices for channel width
#     neg_fwhm_ind = np.argmin(np.abs(wave - wavelength + 2.*channel_width)) - fwhm_buffer

#     #using indices to grab correct data
#     line_profile = line_profile_cube[count, neg_fwhm_ind:pos_fwhm_ind, :, :] #slicing out a profile at width ~5 or 10 sigma...
#     prof_header_copy['CRVAL3'] = wave[neg_fwhm_ind] #correcting the wavelength for beginning of a given line profile
#     profile_cube = SpectralCube(data=line_profile*u.MJy/u.sr, wcs=WCS(prof_header_copy))

#     #now can save profile
#     profile_name = protostar_id + '_'+line_name+ '_'+str(wavelength)+'_profile.fits'
#     profile_cube.write(profile_savepath+profile_name, format='fits', overwrite=True)

#     #and then also save the line to continuum ratio as images
#     line_cont_ratio = torch.sum(line_profile / baseline_data[neg_fwhm_ind:pos_fwhm_ind, :, :], axis=0)
#     name = protostar_folders[protostar_ind] + '_'+line_name+ '_'+str(wavelength)+'_cont_ratio.fits'
#     fits_path = fits_saver(line_cont_ratio, prof_header_copy, name, ratio_savepath) #saving
# print('Saved: Line Profiles')



#slice cube for each summed line and save 
#basically sum[Aijk Bil] over i = Cljk (perhaps requiring an operator to rotate indices)
'''
on units: delta_lambda/lambda = delta_nu / nu = 1/R
per sr...can just convert to per pix and then sum that...
'''
cube_units = 5.370000028051437e-6 / torch.tensor(wave[None,:], dtype=torch.float)**2. #conversions here include MJy, c, Jy to cgs, and spectral resolution, but then divide by lambda^2 for correct units...
sum_unres_fit = torch.einsum('ijk,il->ijk', stacked_coeffs[:len(unres_wavelengths), :, :].float(), cube_units.float()*unres_exp.float()).float()
line_sum_cube = torch.clone(sum_unres_fit) #filling in cube of lines, producing 2D slices by summing along axis for each line

savepath = 'Line_Images/' #set folder
for count, (wavelength, line_name) in enumerate(zip(line_wave_list, species_list)):  
    #now can save summed line
    line_slice = line_sum_cube[count]
    name = protostar_id + '_'+line_name+ '_'+str(wavelength)+'.fits'
    header_edit = jwst_cube.header #need to edit header a bit
    header_edit['BUNIT'] = 'erg cm-2 s-1 sr-1' 
    header_edit['CRVAL3'] = wavelength
    header_edit['CUNIT3'] = 'um'
    header_edit['HISTORY'] = 'The following steps apply: converted to cgs units (erg/s/cm^2/sr), continuum is subtracted, all line profiles simultaneously fit, a given line profile is summed.'
    fits_path = fits_saver(line_slice, header_edit, name, savepath) #saving
    # print(fits_path) #I use to confirm file has right path
print('Saved: Summed Line Maps')