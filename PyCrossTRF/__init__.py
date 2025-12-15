print('Cross TRF package load')

# # Import modules and expose their contents
# from .cross_trf import *
# from .h_block_cv import *
# from .sieve_bs_pooledols_torch import * # The module you want to fix
# from .utils import *
# from .plotting import *

# library management module that refreshing library w/o restart instance.
from importlib import import_module, reload

# # Estimation module - OLS with processed data
estimation_module = import_module('PyCrossTRF.cross_trf')
from .cross_trf import CTRF, recover_ctrf, calc_ctrf, calc_ctrf_comb
reload(estimation_module)



# # cross validation module
cv_module = import_module('PyCrossTRF.h_block_cv')
from .h_block_cv import CV_h_block
reload(cv_module)

# # # cross validation module
# cv_module_refac = import_module('PyCrossTRF.h_block_cv_refac')
# from .h_block_cv_refac import *
# reload(cv_module_refac)

# # # cross validation module
# cv_module_torch = import_module('PyCrossTRF.h_block_cv_torch')
# from .h_block_cv_torch import *
# reload(cv_module_torch)


# # # sieve bootstrap module
# sbs_module = import_module('PyCrossTRF.sieve_bs_pooledols')
# from .sieve_bs_pooledols import *
# reload(sbs_module)

# # sieve bootstrap module
sbs_module = import_module('PyCrossTRF.sieve_bs_pooledols_torch')
from .sieve_bs_pooledols_torch import SieveBootstrap
reload(sbs_module)



# # util module
utilities_module = import_module('PyCrossTRF.utils')
from .utils import *
reload(utilities_module)


# # plotting module
plotting_module = import_module('PyCrossTRF.plotting')
from .plotting import Plot
reload(plotting_module)



# # Nadaraya-Watson kernel regression module
nwkernal_modul = import_module('PyCrossTRF.nadaraya_watson')
from .nadaraya_watson import NadarayaWatson
reload(nwkernal_modul)


# # Nadaraya-Watson kernel regression module
torch_kde = import_module('PyCrossTRF.kde_gaussian_pytorch')
from .kde_gaussian_pytorch import kde_torch_gaussian
reload(torch_kde)





