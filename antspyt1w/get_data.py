"""
Get local ANTsPyT1w data (Unified Version - V0 Legacy & V1 Backends)
"""

__all__ = ['get_data', 'get_backend', 'set_backend', 'set_global_version', 'get_global_version', 
    'set_global_scientific_computing_random_seed', 'get_global_scientific_computing_random_seed', 
    'ap_segmentation_to_dataframe','hierarchical', 'random_basis_projection', 'deep_dkt','deep_tissue_segmentation',
    'deep_brain_parcellation', 'deep_mtl', 'label_hemispheres','brain_extraction',
    'hemi_reg', 'region_reg', 't1_hypointensity', 'zoom_syn', 'merge_hierarchical_csvs_to_wide_format',
    'map_intensity_to_dataframe', 'deep_nbm', 'map_cit168', 'deep_cit168','minimal_sr_preprocessing', 
    'deep_hippo', 'resnet_grader']

from pathlib import Path
import os
from os.path import exists
import pandas as pd
import math
import pickle
import sys
import numpy as np
import random
import re
import functools
from operator import mul
from scipy.sparse.linalg import svds
from PyNomaly import loop
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as ss
import warnings
import ants
from multiprocessing import Pool

# -------------------------------------------------------------------------
# BACKENDS & VERSIONING
# -------------------------------------------------------------------------
try:
    import antspynet
except ImportError:
    antspynet = None

try:
    import antstorch
except ImportError:
    antstorch = None

_GLOBAL_BACKEND = None
if antspynet is not None:
    _GLOBAL_BACKEND = 'antspynet'
elif antstorch is not None:
    _GLOBAL_BACKEND = 'antstorch'

_GLOBAL_VERSION = 0 # 0 = Legacy (local models/tf), 1 = Unified Backends API

def set_global_version(version: int):
    """
    Définit la version de l'API à utiliser.
    0 : Fonctions locales d'origine (nécessite TensorFlow).
    1 : Fonctions déléguées aux backends ANTsPyNet / ANTsTorch.
    """
    global _GLOBAL_VERSION
    if version not in [0, 1]:
        raise ValueError("La version globale doit être 0 ou 1.")
    _GLOBAL_VERSION = version

def get_global_version():
    return _GLOBAL_VERSION

def set_backend(backend_name: str, antstorch_device=None):
    global _GLOBAL_BACKEND
    if backend_name not in ['antspynet', 'antstorch']:
        raise ValueError("The backend should be 'antspynet' or 'antstorch'.")
    if backend_name == 'antspynet' and antspynet is None:
        raise ImportError("antspynet is not installed.")
    if backend_name == 'antstorch' and antstorch is None:
        raise ImportError("antstorch is not installed.")
    
    if backend_name == "antstorch" and antstorch_device is not None:
        antstorch.set_default_device(antstorch_device)
        
    _GLOBAL_BACKEND = backend_name

def get_backend():
    return _GLOBAL_BACKEND

# -------------------------------------------------------------------------
# RANDOM SEED
# -------------------------------------------------------------------------
_GLOBAL_SCIENTIFIC_SEED = None

def set_global_scientific_computing_random_seed(seed: int = 42, deterministic: bool = True):
    global _GLOBAL_SCIENTIFIC_SEED
    _GLOBAL_SCIENTIFIC_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import jax
        jax_key = jax.random.PRNGKey(seed)
        globals()["_jax_key"] = jax_key
    except ImportError:
        pass

    warnings.warn("Remember to set 'random_state=seed' in scikit-learn models.", stacklevel=2)
    print(f"[INFO] Global scientific computing seed set to {seed}")

def get_global_scientific_computing_random_seed():
    return _GLOBAL_SCIENTIFIC_SEED

set_global_scientific_computing_random_seed(1234)

# -------------------------------------------------------------------------
# DATA RETRIEVAL & UTILITIES (COMMON TO V0 & V1)
# -------------------------------------------------------------------------
def get_data(name=None, force_download=False, version=46, target_extension='.csv'):
    import os
    import shutil
    from pathlib import Path
    
    # Lazy import of keras utils for file download
    from tensorflow.keras.utils import get_file
    
    DATA_PATH = os.path.join(os.path.expanduser('~'), '.antspyt1w')
    os.makedirs(DATA_PATH, exist_ok=True)

    def mv_subfolder_files(folder, verbose=False):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if root != folder:
                    shutil.move(os.path.join(root, file), folder)
            for dir in dirs:
                if root != folder:
                    shutil.rmtree(os.path.join(root, dir))

    def download_data(version):
        url = "https://ndownloader.figshare.com/articles/14766102/versions/" + str(version)
        target_file_name = "14766102.zip"
        target_file_name_path = get_file(target_file_name, url, cache_subdir=DATA_PATH, extract=True)
        os.remove(os.path.join(DATA_PATH, target_file_name))

    if force_download:
        download_data(version=version)

    mv_subfolder_files( os.path.expanduser("~/.antspyt1w"), False )

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if root != DATA_PATH:
                shutil.move(os.path.join(root, file), DATA_PATH)
        for dir in dirs:
            if root != DATA_PATH:
                shutil.rmtree(os.path.join(root, dir))

    files = []
    for fname in os.listdir(DATA_PATH):
        if fname.endswith(target_extension):
            fname = os.path.join(DATA_PATH, fname)
            files.append(fname)

    if len(files) == 0:
        download_data(version=version)
        for fname in os.listdir(DATA_PATH):
            if fname.endswith(target_extension):
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)

    mv_subfolder_files( os.path.expanduser("~/.antspyt1w"), False )

    if name == 'all':
        return files

    datapath = None
    for fname in os.listdir(DATA_PATH):
        mystem = Path(fname).resolve().stem
        mystem = Path(mystem).resolve().stem
        mystem = Path(mystem).resolve().stem
        if name == mystem and fname.endswith(target_extension):
            datapath = os.path.join(DATA_PATH, fname)

    if datapath is None:
        warnings.warn("datapath in get_data is None - issue in downloading from figshare.")
    return datapath

def map_segmentation_to_dataframe( segmentation_type, segmentation_image ):
    mydf_fn = get_data( segmentation_type )
    mydf = pd.read_csv( mydf_fn )
    mylgo = ants.label_geometry_measures( segmentation_image )
    return pd.merge( mydf, mylgo, how='left', on=["Label"] )

def map_intensity_to_dataframe(segmentation_type, intensity_image, segmentation_image):
    if isinstance(segmentation_type, str):
        mydf_fn = get_data(segmentation_type)
        mydf = pd.read_csv(mydf_fn)
    elif isinstance(segmentation_type, pd.DataFrame):
        mydf = segmentation_type
    else:
        raise ValueError("segmentation_type must be either a string or a pandas DataFrame")

    mylgo = ants.label_stats(intensity_image, segmentation_image)
    mylgo = mylgo.rename(columns={'LabelValue': 'Label'})
    result = pd.merge(mydf, mylgo, how='left', on=["Label"])
    return result

def myproduct(lst):
    return( functools.reduce(mul, lst) )

def mahalanobis_distance( x ):
    x_minus_mu = x - np.mean(x)
    cov = np.cov(x.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    outlier = []
    C = np.sqrt(sp.stats.chi2.ppf((1-0.001), df=x.shape[1]))
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
    return { "distance": md, "outlier": outlier }

def patch_eigenvalue_ratio( x, n, radii, evdepth = 0.9, mask=None, standardize=False ):
    nptch=n
    radder=radii
    if mask is None:
        msk=ants.threshold_image( x, "Otsu", 1 )
    else:
        msk = mask.clone()
    rnk=ants.rank_intensity(x,msk,True)
    npatchvox = myproduct( radder )
    ptch0 = ants.extract_image_patches( rnk, tuple(radder), mask_image=msk,
        max_number_of_patches = nptch, return_as_array=False, randomize=False )
    ptch = []
    for k in range(len(ptch0)):
        if np.prod( ptch0[k].shape ) == npatchvox :
            ptch.append( np.reshape( ptch0[k], npatchvox ) )
    X = np.stack( ptch )
    if standardize:
        X = X - X.mean(axis=0, keepdims=True)
    thespectrum = np.linalg.svd( X, compute_uv=False )
    spectralsum = thespectrum.sum()
    spectralcumsum = np.cumsum( thespectrum )
    numer = np.argmin(  abs( spectralcumsum - evdepth * spectralsum ) )
    denom = len( thespectrum )
    return numer/denom

def loop_outlierness( random_projections, reference_projections=None, standardize=True, extent=3, n_neighbors=24, cluster_labels=None ):
    use_reference=True
    if reference_projections is None:
        use_reference = False
        reference_projections = random_projections
    nBasisUse = reference_projections.shape[1]
    if random_projections.shape[1] < nBasisUse:
        nBasisUse = random_projections.shape[1]

    refbases = reference_projections.iloc[:,:nBasisUse]
    myax=0
    refbasesmean = refbases.mean(axis=myax)
    refbasessd = refbases.std(axis=myax)
    normalized_df = refbases
    if standardize:
        normalized_df = (normalized_df-refbasesmean)/refbasessd
    if use_reference:
        temp = random_projections.iloc[:,:nBasisUse]
        if standardize:
            temp = (temp-refbasesmean)/refbasessd
        normalized_df = pd.concat( [normalized_df, temp], axis=0 ).dropna(axis=0)
    if cluster_labels is None:
        m = loop.LocalOutlierProbability(normalized_df, extent=extent, n_neighbors=n_neighbors ).fit()
    else:
        m = loop.LocalOutlierProbability(normalized_df, extent=extent, n_neighbors=n_neighbors, cluster_labels=cluster_labels).fit()
    scores = m.local_outlier_probabilities
    return scores

def random_basis_projection( x, template, type_of_transform='Similarity', refbases = None, nBasis=10, random_state = 99 ):
    template = ants.crop_image( template )
    template = ants.iMath( template, "Normalize" )
    np.random.seed(int(random_state))
    nvox = template.shape
    randbasis = np.random.randn( myproduct( nvox ), nBasis  )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    norm = ants.iMath( x, "Normalize" )
    trans = ants.registration( template, norm, type_of_transform='antsRegistrationSyNQuickRepro[t]' )
    resamp = ants.registration( template, norm,
        type_of_transform=type_of_transform, total_sigma=0.5,
        random_seed=1, initial_transform=trans['fwdtransforms'][0] )['warpedmovout']
    mydelta = resamp - template
    imat = ants.get_neighborhood_in_mask( mydelta, mydelta*0+1,[0,0,0], boundary_condition='mean' )
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    record = {}
    uproj_counter = 0
    for i in uproj[0]:
        uproj_counter += 1
        name = "RandBasisProj" + str(uproj_counter).zfill(2)
        record[name] = i
    uprojpos_counter = 0
    for i in uprojpos[0]:
        uprojpos_counter += 1
        name = "RandBasisProjPos" + str(uprojpos_counter).zfill(2)
        record[name] = i
    df = pd.DataFrame(record, index=[0])

    if refbases is None:
        refbases = pd.read_csv( get_data( "reference_basis", target_extension='.csv' ) )
    df['loop_outlier_probability'] = loop_outlierness(  df, refbases,
        n_neighbors=refbases.shape[0] )[ refbases.shape[0] ]
    mhdist = 0.0
    if nBasis == 10:
        temp = pd.concat( [ refbases, df.iloc[:,:nBasis] ], axis=0 )
        mhdist = mahalanobis_distance( temp )['distance'][ refbases.shape[0] ]
    df['mhdist'] = mhdist
    df['templateL1']=mydelta.abs().mean()
    return df

def subdivide_labels( x, verbose = False ):
    notzero=ants.threshold_image( x, 1, np.math.inf )
    ulabs = np.unique( x.numpy() )
    ulabs.sort()
    newx = x * 0.0
    for u in ulabs:
        if u > 0:
            temp = ants.threshold_image( x, u, u )
            subimg = ants.crop_image( x, temp ) * 0
            localshape=subimg.shape
            axtosplit=np.argmax(localshape)
            mid=int(np.round( localshape[axtosplit] /2 ))
            nextlab = newx.max()+1
            if verbose:
                print( "label: " + str( u ) )
            if axtosplit == 1:
                subimg[:,0:mid,:]=subimg[:,0:mid,:]+nextlab
                subimg[:,(mid):(localshape[axtosplit]),:]=subimg[:,(mid):(localshape[axtosplit]),:]+nextlab+1
            if axtosplit == 0:
                subimg[0:mid,:,:]=subimg[0:mid,:,:]+nextlab
                subimg[(mid):(localshape[axtosplit]),:,:]=subimg[(mid):(localshape[axtosplit]),:,:]+nextlab+1
            if axtosplit == 2:
                subimg[:,:,0:mid]=subimg[:,:,0:mid]+nextlab
                subimg[:,:,(mid):(localshape[axtosplit])]=subimg[:,:,(mid):(localshape[axtosplit])]+nextlab+1
            newx = newx + ants.resample_image_to_target( subimg, newx, interp_type='nearestNeighbor' ) * notzero
    return newx

def subdivide_hemi_label( x  ):
    notzero = ants.threshold_image( x, 1, 1e9 )
    localshape = ants.crop_image( x, ants.threshold_image( x, 1, 1 ) ).shape
    axtosplit = np.argmax(localshape)
    mid = int(np.round( localshape[axtosplit] /2 ))
    if axtosplit == 1:
        x[:,0:mid,:]=x[:,0:mid,:]+3
        x[:,(mid):(localshape[axtosplit]),:]=x[:,(mid):(localshape[axtosplit]),:]+5
    if axtosplit == 0:
        x[0:mid,:,:]=x[0:mid,:,:]+3
        x[(mid):(localshape[axtosplit]),:,:]=x[(mid):(localshape[axtosplit]),:,:]+5
    if axtosplit == 2:
        x[:,:,0:mid]=x[:,:,0:mid]+3
        x[:,:,(mid):(localshape[axtosplit])]=x[:,:,(mid):(localshape[axtosplit])]+5
    return x*notzero

def special_crop( x, pt, domainer ):
    pti = np.round( ants.transform_physical_point_to_index( x, pt ) )
    xdim = x.shape
    for k in range(len(xdim)):
        if pti[k] < 0: pti[k]=0
        if pti[k] > (xdim[k]-1): pti[k]=(xdim[k]-1)
    mim = ants.make_image( domainer )
    ptioff = pti.copy()
    for k in range(len(xdim)):
        ptioff[k] = ptioff[k] - np.round( domainer[k] / 2 )
    domainerlo = []
    domainerhi = []
    for k in range(len(xdim)):
        domainerlo.append( int(ptioff[k] - 1) )
        domainerhi.append( int(ptioff[k] + 1) )
    loi = ants.crop_indices( x, tuple(domainerlo), tuple(domainerhi) )
    mim = ants.copy_image_info(loi,mim)
    return ants.resample_image_to_target( x, mim )

def trim_segmentation_by_distance( segmentation, which_label, distance ):
    bseg = ants.threshold_image( segmentation, 1, segmentation.max() )
    dist = ants.iMath( bseg, "MaurerDistance" ) * (-1.0)
    disttrim = ants.threshold_image( dist, distance, dist.max() )
    tarseg = ants.threshold_image( segmentation, which_label, which_label ) * disttrim
    segmentationtrim = segmentation.clone()
    segmentationtrim[ segmentation == which_label ] = 0
    return segmentationtrim + tarseg * which_label

def localsyn(img, template, hemiS, templateHemi, whichHemi, padder, iterations, output_prefix, total_sigma=0.5):
    ihemi = img * ants.threshold_image(hemiS, whichHemi, whichHemi)
    themi = template * ants.threshold_image(templateHemi, whichHemi, whichHemi)
    loquant = np.quantile(themi.numpy(), 0.01) + 1e-6
    hemicropmask = ants.threshold_image(
        templateHemi * ants.threshold_image(themi, loquant, math.inf),
        whichHemi, whichHemi
    ).iMath("MD", padder)
    tcrop = ants.crop_image(themi, hemicropmask)
    syn = ants.registration(
        tcrop,
        ihemi,
        'antsRegistrationSyNQuickRepro[s]',
        reg_iterations=iterations,
        flow_sigma=3.0,
        total_sigma=total_sigma,
        verbose=False,
        outprefix=output_prefix,
        random_seed=1
    )
    return syn

def hemi_reg(input_image, input_image_tissue_segmentation, input_image_hemisphere_segmentation, input_template, input_template_hemisphere_labels, output_prefix, padding=10, labels_to_register=[2,3,4,5], total_sigma=0.5, is_test=False ):
    img = ants.rank_intensity( input_image )
    ionlycerebrum = brain_extraction( input_image )
    tonlycerebrum = brain_extraction( input_template )
    template = ants.rank_intensity( input_template )
    regsegits=[200,200,20]

    if is_test:
        regsegits=[8,0,0]

    input_template_hemisphere_labels = ants.resample_image_to_target(
        input_template_hemisphere_labels,
        template,
        interp_type='nearestNeighbor',
    )

    synL = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=input_image_hemisphere_segmentation,
        templateHemi=input_template_hemisphere_labels,
        whichHemi=1,
        padder=padding,
        iterations=regsegits,
        output_prefix = output_prefix + "left_hemi_reg",
        total_sigma=total_sigma,
    )
    synR = localsyn(
        img=img*ionlycerebrum,
        template=template*tonlycerebrum,
        hemiS=input_image_hemisphere_segmentation,
        templateHemi=input_template_hemisphere_labels,
        whichHemi=2,
        padder=padding,
        iterations=regsegits,
        output_prefix = output_prefix + "right_hemi_reg",
        total_sigma=total_sigma,
    )

    ants.image_write(synL['warpedmovout'], output_prefix + "left_hemi_reg.nii.gz" )
    ants.image_write(synR['warpedmovout'], output_prefix + "right_hemi_reg.nii.gz" )
    fignameL = output_prefix + "_left_hemi_reg.png"
    ants.plot(synL['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameL, black_bg=False, crop=True )
    fignameR = output_prefix + "_right_hemi_reg.png"
    ants.plot(synR['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameR, black_bg=False, crop=True )

    lhjac = ants.create_jacobian_determinant_image(synL['warpedmovout'], synL['fwdtransforms'][0], do_log=1)
    ants.image_write( lhjac, output_prefix+'left_hemi_jacobian.nii.gz' )
    rhjac = ants.create_jacobian_determinant_image(synR['warpedmovout'], synR['fwdtransforms'][0], do_log=1)
    ants.image_write( rhjac, output_prefix+'right_hemi_jacobian.nii.gz' )
    
    return {"synL":synL, "synLpng":fignameL, "synR":synR, "synRpng":fignameR, "lhjac":lhjac, "rhjac":rhjac }

def region_reg(input_image, input_image_tissue_segmentation, input_image_region_segmentation, input_template, input_template_region_segmentation, output_prefix, padding=10, total_sigma=0.5, is_test=False ):
    img = ants.rank_intensity( input_image )
    ionlycerebrum = brain_extraction( input_image )
    template = ants.rank_intensity( input_template )
    regsegits = [200,200,20]

    if min(ants.get_spacing(img)) < 0.8:
        regsegits=[200,200,200,20]
        template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )

    if is_test:
        regsegits=[20,5,0]

    input_template_region_segmentation = ants.resample_image_to_target(
        input_template_region_segmentation,
        template,
        interp_type='nearestNeighbor',
    )

    synL = localsyn(
        img=img*ionlycerebrum,
        template=template,
        hemiS=input_image_region_segmentation,
        templateHemi=input_template_region_segmentation,
        whichHemi=1,
        padder=padding,
        iterations=regsegits,
        output_prefix = output_prefix + "region_reg",
        total_sigma=total_sigma,
    )

    ants.image_write(synL['warpedmovout'], output_prefix + "region_reg.nii.gz" )
    fignameL = output_prefix + "_region_reg.png"
    ants.plot(synL['warpedmovout'],axis=2,ncol=8,nslices=24,filename=fignameL, black_bg=False, crop=True )
    lhjac = ants.create_jacobian_determinant_image(synL['warpedmovout'], synL['fwdtransforms'][0], do_log=1)
    ants.image_write( lhjac, output_prefix+'region_jacobian.nii.gz' )

    return {"synL":synL, "synLpng":fignameL, "lhjac":lhjac, 'rankimg':img*ionlycerebrum, 'template':template}

def preprocess_intensity( x, brain_extraction, intensity_truncation_quantiles=[1e-4, 0.999], rescale_intensities=True  ):
    brain_extraction = ants.resample_image_to_target( brain_extraction, x, interp_type='nearestNeighbor' )
    img = x * brain_extraction
    img = ants.iMath( img, "TruncateIntensity", intensity_truncation_quantiles[0], intensity_truncation_quantiles[1] ).iMath( "Normalize" )
    img = ants.n4_bias_field_correction( img, mask=brain_extraction, rescale_intensities=rescale_intensities, ).iMath("Normalize")
    return img

def kelly_kapowski_thickness( x, labels, label_description='dkt', iterations=45, max_thickness=6.0, verbose=False ):
    if verbose: 
        myverb=1
    else: 
        myverb=0
    seg = deep_tissue_segmentation( x )
    kkthk = ants.kelly_kapowski( s=seg['segmentation_image'],
            g=seg['probability_images'][2], w=seg['probability_images'][3],
            its=iterations, r=0.025, m=1.5, verbose=myverb )
    kkthkmask = ants.threshold_image( kkthk, 0.25, max_thickness )
    kkdf = map_intensity_to_dataframe(
                  label_description,
                  kkthk,
                  labels * kkthkmask )
    kkdf_wide = merge_hierarchical_csvs_to_wide_format( {'KK' : kkdf}, col_names = ['Mean'] )
    return { 'thickness_image' : kkthk, 'thickness_dataframe' : kkdf_wide }

def zoom_syn( target_image, template, template_segmentations, initial_registration, dilation = 4, regIterations = [25] ):
    croppertem = ants.iMath( template_segmentations[0], "MD", dilation )
    templatecrop = ants.crop_image( template, croppertem )
    cropper = ants.apply_transforms( target_image,
        croppertem, initial_registration['fwdtransforms'],
        interpolator='linear' ).threshold_image(0.5,1.e9)
    croplow = ants.crop_image( target_image,  cropper )
    synnerlow = ants.registration( croplow, templatecrop,
        'SyNOnly', gradStep = 0.20, regIterations = regIterations, randomSeed=1,
        syn_metric='cc', syn_sampling=2, total_sigma=0.5,
        initialTransform = initial_registration['fwdtransforms'] )
    orlist = []
    for jj in range(len(template_segmentations)):
      target_imageg = ants.apply_transforms( target_image, template_segmentations[jj],
        synnerlow['fwdtransforms'],
        interpolator='linear' ).threshold_image(0.5,1e9)
      orlist.append( target_imageg )
    return{ 'segmentations': orlist, 'registration': synnerlow, 'croppedimage': croplow, 'croppingmask': cropper }

def read_hierarchical( output_prefix ):
    mydataframes = {
            "rbp": None, "hemispheres":None, "tissues":None, "dktlobes":None, "dktregions":None, "dktcortex":None,
            "wmtracts_left":None, "wmtracts_right":None, "mtl":None, "bf":None, "cit168":None, "deep_cit168":None,
            "snseg":None, "cerebellum":None, "brainstem":None }

    dkt_parc = {
        "tissue_segmentation":None, "tissue_probabilities":None, "dkt_parcellation":None,
        "dkt_lobes":None, "dkt_cortex": None, "hemisphere_labels": None, "wmSNR": None, "wmcsfSNR": None }

    hierarchical_object = {
            "brain_n4_dnz": None, "brain_n4_dnz_png": None, "brain_extraction": None, "tissue_seg_png": None,
            "left_right": None, "dkt_parc": dkt_parc, "registration":None, "wm_tractsL":None, "wm_tractsR":None,
            "mtl":None, "bf":None, "deep_cit168lab":  None, "cit168lab":  None, "cit168reg":  None,
            "snseg":None, "snreg":None, "cerebellum":None, "brainstem":None, "dataframes": mydataframes }

    for myvar in hierarchical_object['dataframes'].keys():
        if hierarchical_object['dataframes'][myvar] is None and exists( output_prefix + myvar + ".csv"):
            hierarchical_object['dataframes'][myvar] = pd.read_csv(output_prefix + myvar + ".csv")

    myvarlist = hierarchical_object.keys()
    for myvar in myvarlist:
        if hierarchical_object[myvar] is None and exists( output_prefix + myvar + '.nii.gz' ):
            hierarchical_object[myvar] = ants.image_read( output_prefix + myvar + '.nii.gz' )

    myvarlist = ['tissue_segmentation', 'dkt_parcellation', 'dkt_lobes', 'dkt_cortex', 'hemisphere_labels' ]
    for myvar in myvarlist:
        if hierarchical_object['dkt_parc'][myvar] is None and exists( output_prefix + myvar + '.nii.gz' ):
            hierarchical_object['dkt_parc'][myvar] = ants.image_read( output_prefix + myvar + '.nii.gz' )

    return hierarchical_object

def write_hierarchical( hierarchical_object, output_prefix, verbose=False ):
    for myvar in hierarchical_object['dataframes'].keys():
        if hierarchical_object['dataframes'][myvar] is not None:
            hierarchical_object['dataframes'][myvar].dropna(axis=0).to_csv(output_prefix + myvar + ".csv")

    myvarlist = hierarchical_object.keys()
    r16img = ants.image_read( ants.get_data( "r16" ))
    for myvar in myvarlist:
        if hierarchical_object[myvar] is not None and type(hierarchical_object[myvar]) == type( r16img ):
            ants.image_write( hierarchical_object[myvar], output_prefix + myvar + '.nii.gz' )

    myvarlist = ['tissue_segmentation', 'dkt_parcellation', 'dkt_lobes', 'dkt_cortex', 'hemisphere_labels' ]
    for myvar in myvarlist:
        if hierarchical_object['dkt_parc'][myvar] is not None:
            ants.image_write( hierarchical_object['dkt_parc'][myvar], output_prefix + myvar + '.nii.gz' )
    return

def merge_hierarchical_csvs_to_wide_format( hierarchical_dataframes, col_names = None , identifier=None, identifier_name='u_hier_id', verbose=False ):
    if identifier is None: identifier='A'
    wide_df = pd.DataFrame( )
    icvkey='icv'
    if icvkey in hierarchical_dataframes.keys():
        temp = hierarchical_dataframes[icvkey].copy()
        temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
        temp.insert(loc=0, column=identifier_name, value=identifier)
        temp = temp.set_index(identifier_name)
        wide_df = wide_df.join(temp,how='outer')
    for myvar in hierarchical_dataframes.keys():
        if hierarchical_dataframes[myvar] is not None:
            jdf = hierarchical_dataframes[myvar].dropna(axis=0)
            jdf = jdf.loc[:, ~jdf.columns.str.contains('^Unnamed')]
            if col_names is not None :
                for col_name in col_names :
                    if jdf.shape[0] > 1 and any( jdf.columns.str.contains(col_name)):
                        varsofinterest = ["Description", col_name]
                        jdfsub = jdf[varsofinterest]
                        jdfsub.insert(loc=0, column=identifier_name, value=identifier)
                        jdfsub = jdfsub.set_index([identifier_name, 'Description'])[col_name].unstack().add_prefix(col_name + '_')
                        jdfsub.columns=jdfsub.columns
                        jdfsub = jdfsub.rename(mapper=lambda x: x.strip().replace(' ', '_').lower(), axis=1)
                        wide_df = wide_df.join(jdfsub,how='outer')
            if jdf.shape[0] > 1 and any( jdf.columns.str.contains('VolumeInMillimeters')):
                varsofinterest = ["Description", "VolumeInMillimeters"]
                jdfsub = jdf[varsofinterest]
                jdfsub.insert(loc=0, column=identifier_name, value=identifier)
                jdfsub=jdfsub.set_index([identifier_name, 'Description']).VolumeInMillimeters.unstack().add_prefix('Vol_')
                jdfsub.columns=jdfsub.columns+myvar
                jdfsub = jdfsub.rename(mapper=lambda x: x.strip().replace(' ', '_').lower(), axis=1)
                wide_df = wide_df.join(jdfsub,how='outer')
            if jdf.shape[0] > 1 and any( jdf.columns.str.contains('SurfaceAreaInMillimetersSquared')):
                varsofinterest = ["Description", "SurfaceAreaInMillimetersSquared"]
                jdfsub = jdf[varsofinterest]
                jdfsub.insert(loc=0, column=identifier_name, value=identifier)
                jdfsub=jdfsub.set_index([identifier_name, 'Description']).SurfaceAreaInMillimetersSquared.unstack().add_prefix('Area_')
                jdfsub.columns=jdfsub.columns+myvar
                jdfsub = jdfsub.rename(mapper=lambda x: x.strip().replace(' ', '_').lower(), axis=1)
                wide_df = wide_df.join(jdfsub,how='outer')
            if jdf.shape[0] > 1 and any( jdf.columns.str.contains('SurfaceAreaInMillimetersSquared')) and any( jdf.columns.str.contains('VolumeInMillimeters')):
                varsofinterest = ["Description", "VolumeInMillimeters", "SurfaceAreaInMillimetersSquared"]
                jdfsub = jdf[varsofinterest]
                jdfsub.insert(loc=0, column=identifier_name, value=identifier)
                jdfsub.insert(loc=1, column='thickness',value=jdfsub['VolumeInMillimeters']/jdfsub['SurfaceAreaInMillimetersSquared'])
                jdfsub=jdfsub.set_index([identifier_name, 'Description']).thickness.unstack().add_prefix('Thk_')
                jdfsub.columns=jdfsub.columns+myvar
                jdfsub = jdfsub.rename(mapper=lambda x: x.strip().replace(' ', '_').lower(), axis=1)
                wide_df = wide_df.join(jdfsub,how='outer')
    rbpkey='rbp'
    if rbpkey in hierarchical_dataframes.keys():
        temp = hierarchical_dataframes[rbpkey].copy()
        temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
        temp.insert(loc=0, column=identifier_name, value=identifier)
        temp = temp.set_index(identifier_name)
        wide_df = wide_df.join(temp,how='outer')
    wmhkey='wmh'
    if wmhkey in hierarchical_dataframes.keys():
        df=hierarchical_dataframes[wmhkey].copy()
        df.insert(loc=0, column=identifier_name, value=identifier)
        df = df.set_index(identifier_name)
        wmhval = df.loc[ df.Description == 'Volume_of_WMH','Value']
        wide_df.insert(loc = 0, column = 'wmh_vol', value =wmhval )
        wmhval = df.loc[ df.Description == 'Integral_WMH_probability','Value']
        wide_df.insert(loc = 0, column = 'wmh_integral_prob', value =wmhval )
        wmhval = df.loc[ df.Description == 'Log_Evidence','Value']
        wide_df.insert(loc = 0, column = 'wmh_log_evidence', value =wmhval )
        wide_df['wmh_log_evidence']=wmhval
    wide_df.insert(loc = 0, column = identifier_name, value = identifier)
    return wide_df

def label_and_img_to_sr( img, label_img, sr_model, return_intensity=False, target_range=[1,0] ):
    ulabs = np.unique( label_img.numpy() )
    ulabs.sort()
    ulabs = ulabs[1:len(ulabs)]
    ulabs = ulabs.tolist()
    if return_intensity:
        return super_resolution_segmentation_per_label(img, label_img, [2,2,2], sr_model, segmentation_numbers=ulabs, target_range=target_range, max_lab_plus_one=True  )
    else:
        return super_resolution_segmentation_per_label(img, label_img, [2,2,2], sr_model, segmentation_numbers=ulabs, target_range=target_range, max_lab_plus_one=True  )['super_resolution_segmentation']

def super_resolution_segmentation_per_label(imgIn, segmentation, upFactor, sr_model, segmentation_numbers, dilation_amount = 0, probability_images=None, probability_labels=None, max_lab_plus_one=True, target_range=[1,0], poly_order = 'hist', verbose = False):
    import re
    if type( sr_model ) == type(""):
        if re.search( 'h5', sr_model ) is not None:
            import tensorflow as tf
            sr_model=tf.keras.models.load_model( sr_model, compile=False )
            
    newspc = ( np.asarray( ants.get_spacing( imgIn ) ) ).tolist()
    for k in range(len(newspc)):
        newspc[k] = newspc[k]/upFactor[k]
    imgup = ants.resample_image( imgIn, newspc, use_voxels=False, interp_type=0 )
    imgsrfull = imgup * 0.0
    weightedavg = imgup * 0.0
    problist = []
    bkgdilate = 2
    segmentationUse = ants.image_clone( segmentation )
    segmentationUse = ants.mask_image( segmentationUse, segmentationUse, segmentation_numbers )
    segmentation_numbers_use = segmentation_numbers.copy()
    if max_lab_plus_one:
        background = ants.mask_image( segmentationUse, segmentationUse, segmentation_numbers, binarize=True )
        background = ants.iMath(background,"MD",bkgdilate) - background
        backgroundup = ants.resample_image_to_target( background, imgup, interp_type='linear' )
        segmentation_numbers_use.append( max(segmentation_numbers) + 1 )
        segmentationUse = segmentationUse + background * max(segmentation_numbers_use)

    for locallab in segmentation_numbers:
        binseg = ants.threshold_image( segmentationUse, locallab, locallab )
        sizethresh = 2
        if ( binseg == 1 ).sum() >= sizethresh :
            if probability_images is not None:
                whichprob = probability_labels.index(locallab)
                probimg = probability_images[whichprob].resample_image_to_target( binseg )
            binsegdil = ants.iMath( ants.threshold_image( segmentationUse, locallab, locallab ), "MD", dilation_amount )
            binsegdil2input = ants.resample_image_to_target( binsegdil, imgIn, interp_type='nearestNeighbor'  )
            imgc = ants.crop_image( ants.iMath(imgIn,"Normalize"), binsegdil2input )
            imgc = imgc * target_range[0] - target_range[1]
            imgchCore = ants.crop_image( binseg, binsegdil )
            if probability_images is not None:
                imgchCore = ants.crop_image( probimg, binsegdil )
            imgch = imgchCore * target_range[0] - target_range[1]
            if type( sr_model ) == type(""):
                binsegup = ants.resample_image_to_target( binseg, imgup, interp_type='linear' )
                problist.append( binsegup )
            else:
                myarr = np.stack( [imgc.numpy(),imgch.numpy()],axis=3 )
                newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
                myarr = myarr.reshape( newshape )
                pred = sr_model.predict( myarr )
                
                # Handling tf.squeeze / np.squeeze locally
                try:
                    import tensorflow as tf
                    pred_0 = tf.squeeze(pred[0]).numpy()
                    pred_1 = tf.squeeze(pred[1]).numpy() if isinstance(pred, list) else None
                except ImportError:
                    pred_0 = np.squeeze(pred[0])
                    pred_1 = np.squeeze(pred[1]) if isinstance(pred, list) else None

                imgsr = ants.from_numpy(pred_0)
                imgsr = ants.copy_image_info( imgc, imgsr )
                ants.set_spacing( imgsr,  newspc )
                
                imgsrh = ants.from_numpy(pred_1)
                imgsrh = ants.copy_image_info( imgc, imgsrh )
                ants.set_spacing( imgsrh,  newspc )
                
                if poly_order is not None:
                    if poly_order == 'hist':
                        imgsrh = ants.histogram_match_image( imgsrh, imgchCore )
                        imgsr = ants.histogram_match_image( imgsr, imgc )
                    else:
                        imgsr = ants.regression_match_image( imgsr, ants.resample_image_to_target(imgup,imgsr), poly_order=poly_order )
                        imgsrh = ants.regression_match_image( imgsrh, ants.resample_image_to_target(imgchCore,imgsr), poly_order=poly_order )
                problist.append( imgsrh )
                contribtoavg = ants.resample_image_to_target( imgsr*0+1, imgup, interp_type='nearestNeighbor' )
                weightedavg = weightedavg + contribtoavg
                imgsrfull = imgsrfull + ants.resample_image_to_target( imgsr, imgup, interp_type='nearestNeighbor' )

    if max_lab_plus_one:
        problist.append( backgroundup )

    imgsrfull2 = imgsrfull
    selector = imgsrfull == 0
    imgsrfull2[ selector  ] = imgup[ selector ]
    weightedavg[ weightedavg == 0.0 ] = 1.0
    imgsrfull2=imgsrfull2/weightedavg
    imgsrfull2[ imgup == 0 ] = 0

    for k in range(len(problist)):
        problist[k] = ants.resample_image_to_target(problist[k],imgsrfull2,interp_type='linear')

    if max_lab_plus_one:
        tarmask = ants.threshold_image( segmentationUse, 1, segmentationUse.max() )
    else:
        tarmask = ants.threshold_image( segmentationUse, 1, segmentationUse.max() ).iMath("MD",1)
    tarmask = ants.resample_image_to_target( tarmask, imgsrfull2, interp_type='nearestNeighbor' )

    segmat = ants.images_to_matrix(problist, tarmask)
    finalsegvec = segmat.argmax(axis=0)
    finalsegvec2 = finalsegvec.copy()
    for i in range(len(problist)):
        segnum = segmentation_numbers_use[i]
        finalsegvec2[finalsegvec == i] = segnum
    outimg = ants.make_image(tarmask, finalsegvec2)
    outimg = ants.mask_image( outimg, outimg, segmentation_numbers )
    seggeom = ants.label_geometry_measures( outimg )

    return { "super_resolution": imgsrfull2, "super_resolution_segmentation": outimg, "segmentation_geometry": seggeom, "probability_images": problist }


def super_resolution_segmentation_with_probabilities(img, initial_probabilities, sr_model, target_range=[1,0], verbose = False):
    srimglist = []
    srproblist = []
    mypt = 1.0 / len(initial_probabilities)

    for k in range(len(initial_probabilities)):
        if k == 0:
            srimglist.append( img )
            srproblist.append( initial_probabilities[k] )
        else:
            tempm = ants.threshold_image( initial_probabilities[k], mypt, 2 ).iMath("MD",1)
            imgc = ants.crop_image(img,tempm)
            imgch = ants.crop_image(initial_probabilities[k],tempm)
            imgcrescale = ants.iMath( imgc, "Normalize" ) * target_range[0] - target_range[1]
            imgchrescale = imgch * target_range[0] - target_range[1]
            myarr = np.stack( [ imgcrescale.numpy(), imgchrescale.numpy() ],axis=3 )
            newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
            myarr = myarr.reshape( newshape )
            pred = sr_model.predict( myarr )
            
            try:
                import tensorflow as tf
                pred_0 = tf.squeeze(pred[0]).numpy()
                pred_1 = tf.squeeze(pred[1]).numpy() if isinstance(pred, list) else None
            except ImportError:
                pred_0 = np.squeeze(pred[0])
                pred_1 = np.squeeze(pred[1]) if isinstance(pred, list) else None

            imgsr = ants.from_numpy(pred_0)
            imgsr = ants.copy_image_info( imgc, imgsr )
            newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
            ants.set_spacing( imgsr,  newspc )
            imgsr = ants.regression_match_image( imgsr, ants.resample_image_to_target(imgc,imgsr) )
            
            imgsrh = ants.from_numpy(pred_1)
            imgsrh = ants.copy_image_info( imgsr, imgsrh )
            tempup = ants.resample_image_to_target( tempm, imgsr )
            srimglist.append( imgsr )
            srproblist.append( imgsrh * tempup )

    labels = { 'sr_intensities':srimglist, 'sr_probabilities':srproblist }
    return labels

def hierarchical_to_sr( t1hier, sr_model, tissue_sr=False, blending=0.5, verbose=False ):
    img = t1hier['brain_n4_dnz']
    myvarlist = [ 'mtl', 'cit168lab', 'snseg', 'bf', 'deep_cit168lab' ]
    for myvar in myvarlist:
        t1hier[myvar]=label_and_img_to_sr( img, t1hier[myvar], sr_model )
        
    temp = label_and_img_to_sr( img, t1hier['dkt_parc']['dkt_cortex'], sr_model, return_intensity=True )
    tempupimg = temp['super_resolution']
    if blending is not None:
        tempupimg = tempupimg * (1.0 - blending ) + ants.iMath( tempupimg, "Sharpen" ) * blending
    t1hier['dkt_parc']['dkt_cortex']= temp['super_resolution_segmentation']
    t1hier['dataframes']["dktcortex"]= map_segmentation_to_dataframe( "dkt", t1hier['dkt_parc']['dkt_cortex'] )
    t1hier['dataframes']["mtl"]=map_segmentation_to_dataframe( 'mtl_description',  t1hier['mtl'] )
    t1hier['dataframes']["cit168"]=map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', t1hier['cit168lab'] )
    t1hier['dataframes']["snseg"]=map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', t1hier['snseg'] )
    t1hier['dataframes']["bf"]=map_segmentation_to_dataframe( 'nbm3CH13', t1hier['bf'] )
    t1hier['dataframes']["deep_cit168"]=map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', t1hier['deep_cit168lab'] )

    if tissue_sr:
        bmask = ants.threshold_image( t1hier['dkt_parc']['tissue_segmentation'], 1, 6 )
        segcrop = ants.image_clone( t1hier['dkt_parc']['tissue_segmentation'] )
        hemicrop = t1hier['left_right']
        segcrop[ hemicrop == 2 ] = ( segcrop[ hemicrop == 2  ] + 6 )
        segcrop = segcrop * bmask
        mysr = super_resolution_segmentation_per_label(
                    t1hier['brain_n4_dnz'], segcrop, [2,2,2], sr_model, [1,2,3,4,5,6,7,8,9,10,11,12],
                    dilation_amount=0, probability_images=None,
                    probability_labels=[1,2,3,4,5,6,7,8,9,10,11,12],
                    max_lab_plus_one=True, verbose=True )
        if blending is not None:
            mysr['super_resolution'] = mysr['super_resolution'] * (1.0 - blending ) + ants.iMath( mysr['super_resolution'], "Sharpen" ) * blending
        temp = mysr['super_resolution_segmentation']
        for k in [7,8,9,10,11,12] :
            temp[ temp == k ] = temp[ temp == k ] - 6
        t1hier['brain_n4_dnz'] = mysr['super_resolution']
        t1hier['dkt_parc']['tissue_segmentation'] = temp
    else:
        t1hier['brain_n4_dnz'] = tempupimg
        t1hier['dkt_parc']['tissue_segmentation'] = ants.resample_image_to_target(
            t1hier['dkt_parc']['tissue_segmentation'], tempupimg, 'nearestNeighbor')

    tissue = map_segmentation_to_dataframe( "tissues", ants.mask_image( t1hier['dkt_parc']['tissue_segmentation'], t1hier['dkt_parc']['tissue_segmentation'],  [1,2,3,4,5,6] ) )
    t1hier['dataframes']['tissues'] = tissue
    return t1hier

# -------------------------------------------------------------------------
# BACKEND-AWARE / VERSION-ROUTED FUNCTIONS
# -------------------------------------------------------------------------

def resnet_grader( x, weights_filename=None ):
    if get_global_version() == 0:
        import tensorflow as tf
        if weights_filename is None:
            weights_filename=get_data( 'resnet_grader', target_extension='.h5' )

        if not exists( weights_filename ):
            print("resnet_grader weights do not exist: " + weights_filename )
            return None

        mdl = antspynet.create_resnet_model_3d( [None,None,None,1], lowest_resolution = 32, number_of_outputs = 4, cardinality = 1, squeeze_and_excite = False )
        mdl.load_weights( weights_filename )

        t1 = ants.iMath( x - x.min(),  "Normalize" )
        bxt = ants.threshold_image( t1, 0.01, 1.0 )
        t1 = ants.rank_intensity( t1, mask=bxt, get_mask=True )
        templateb = ants.image_read( get_data( "S_template3_brain", target_extension='.nii.gz' ) )
        templateb = ants.crop_image( templateb ).resample_image( [1,1,1] )
        templateb = ants.pad_image_by_factor( templateb, 8 )
        templatebsmall = ants.resample_image( templateb, [2,2,2] )
        reg = ants.registration( templatebsmall, t1, 'Similarity', verbose=False )
        ilist = [[templateb]]
        nsim = 16
        uu = ants.randomly_transform_image_data( templateb, ilist, number_of_simulations = nsim, transform_type='scaleShear', sd_affine=0.075 )
        fwdaffgd = ants.read_transform( reg['fwdtransforms'][0])
        scoreNums = np.zeros( 4 )
        scoreNums[3], scoreNums[2], scoreNums[1], scoreNums[0] = 0, 1, 2, 3
        scoreNums=scoreNums.reshape( [4,1] )

        def get_grade( score, probs ):
            grade='f'
            if score >= 2.25: grade='a'
            elif score >= 1.5: grade='b'
            elif score >= 0.75: grade='c'
            probgradeindex = np.argmax( probs )
            probgrade = ['a','b','c','f'][probgradeindex]
            return [grade, probgrade, float( score )]

        gradelistNum, gradelistProb, gradeScore = [], [], []
        for k in range( nsim ):
            simtx = uu['simulated_transforms'][k]
            cmptx = ants.compose_ants_transforms( [simtx,fwdaffgd] ) 
            subjectsim = ants.apply_ants_transform_to_image( cmptx, t1, templateb, interpolation='linear' )
            subjectsim = ants.add_noise_to_image( subjectsim, 'additivegaussian', (0,0.01) )
            xarr = subjectsim.numpy()
            newshape = [1] + list( xarr.shape ) + [1]
            xarr = np.reshape(  xarr, newshape  )
            preds = mdl.predict( xarr )
            predsnum = tf.matmul(  preds, scoreNums )
            locgrades = get_grade( predsnum, preds )
            gradelistNum.append( locgrades[0] )
            gradelistProb.append( locgrades[1] )
            gradeScore.append( locgrades[2] )

        def most_frequent(List):
            return max(set(List), key = List.count)

        mydf = pd.DataFrame( {"NumericGrade": gradelistNum, "ProbGrade": gradelistProb, "NumericScore": gradeScore, 'grade': most_frequent( gradelistProb )})
        smalldf = pd.DataFrame( {'gradeLetter':  [mydf.grade[0]], 'gradeNum': [mydf.NumericScore.mean()] }, index=[0] )
        return smalldf
    else:
        if get_backend() == "antspynet":
            smalldf = antspynet.t1_grader(x)
        else:
            smalldf = antstorch.t1_grader(x) 
        return smalldf

def inspect_raw_t1( x, output_prefix, option='both' ):
    if x.dimension != 3:
        raise ValueError('inspect_raw_t1: input image should be 3-dimensional')

    x = ants.iMath( x, "Normalize" )
    csvfn, pngfn = output_prefix + "head.csv", output_prefix + "head.png"
    csvfnb, pngfnb = output_prefix + "brain.csv", output_prefix + "brain.png"

    rbh = pd.read_csv( get_data( "refbasis_head", target_extension=".csv" ) )
    rbb = pd.read_csv( get_data( "refbasis_brain", target_extension=".csv" ) )

    rbp=None
    if option == 'both' or option == 'head':
        if get_global_version() == 0 or get_backend() == 'antspynet':
            bfn = antspynet.get_antsxnet_data( "S_template3" )
        else:
            bfn = antstorch.get_antstorch_data( "S_template3" )
            
        templateb = ants.image_read( bfn ).iMath("Normalize")
        templatesmall = ants.resample_image( templateb, (2,2,2), use_voxels=False )
        lomask = ants.threshold_image( x, "Otsu", 2 ).threshold_image(1,2)
        t1 = ants.rank_intensity( x * lomask, mask=lomask, get_mask=False )
        ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfn, crop=True )
        rbp = random_basis_projection( t1, templatesmall, type_of_transform='Rigid', refbases=rbh )
        rbp.to_csv( csvfn )
        
        looper=float(rbp['loop_outlier_probability'].iloc[0])
        ttl="LOOP: " + "{:0.4f}".format(looper) + " MD: " + "{:0.4f}".format(float(rbp['mhdist'].iloc[0]))
        img = Image.open( pngfn ).copy()
        plt.figure(dpi=300)
        plt.imshow(img)
        plt.text(20, 0, ttl, color="red", fontsize=12 )
        plt.axis("off")
        plt.subplots_adjust(0,0,1,1)
        plt.savefig( pngfn, bbox_inches='tight',pad_inches = 0)
        plt.close()

    rbpb=None
    if option == 'both' or option == 'brain':
        if option == 'both':
            t1 = ants.iMath( x, "TruncateIntensity",0.001, 0.999).iMath("Normalize")
            if get_global_version() == 0 or get_backend() == 'antspynet':
                lomask = antspynet.brain_extraction( t1, "t1" )
            else:
                lomask = antstorch.brain_extraction( t1, "t1" )
            t1 = ants.rank_intensity( t1 * lomask, get_mask=True )
        else:
            t1 = ants.iMath( x, "Normalize" )
            t1 = ants.rank_intensity( t1, get_mask=True )
            
        ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfnb, crop=True )
        templateb = ants.image_read( get_data( "S_template3_brain", target_extension='.nii.gz' ) )
        templatesmall = ants.resample_image( templateb, (2,2,2), use_voxels=False )
        rbpb = random_basis_projection( t1, templatesmall, type_of_transform='Rigid', refbases=rbb )
        rbpb['evratio'] = patch_eigenvalue_ratio( t1, 512, [20,20,20], evdepth = 0.9 )
        
        grade0 = resnet_grader( t1 )['gradeNum'].iloc[0]
        msk=ants.threshold_image(t1,0.01,1.0)
        t1tx=ants.n4_bias_field_correction( t1, mask=msk )
        t1tx=ants.iMath(t1tx,'TruncateIntensity',0.001,0.98)
        grade1 = resnet_grader( t1tx )['gradeNum'].iloc[0]
        
        if grade1 > grade0:
            grade0 = grade1
        rbpb['resnetGrade'] = grade0
        rbpb.to_csv( csvfnb )
        looper = float(rbpb['loop_outlier_probability'].iloc[0])
        myevr = float( rbpb['evratio'].iloc[0] )
        mygrd = float( rbpb['resnetGrade'].iloc[0] )
        myl1 = float( rbpb['templateL1'].iloc[0] )
        ttl="LOOP: " + "{:0.4f}".format(looper) + " MD: " + "{:0.4f}".format(float(rbpb['mhdist'].iloc[0])) + " EVR: " + "{:0.4f}".format(myevr) + " TL1: " + "{:0.4f}".format(myl1) + " grade: " + "{:0.4f}".format(mygrd)
        img = Image.open( pngfnb ).copy()
        plt.figure(dpi=300)
        plt.imshow(img)
        plt.text(20, 0, ttl, color="red", fontsize=12 )
        plt.axis("off")
        plt.subplots_adjust(0,0,1,1)
        plt.savefig( pngfnb, bbox_inches='tight',pad_inches = 0)
        plt.close()

    return { "head": rbp, "head_image": pngfn, "brain": rbpb, "brain_image": pngfnb }

def icv( x ):
    if get_global_version() == 0 or get_backend() == 'antspynet':
        icvseg = antspynet.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,2)
    else:
        icvseg = antstorch.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,2)
    np.product = np.prod( ants.get_spacing( x ) ) * icvseg.sum()
    return pd.DataFrame({'icv': [np.product]})

def brain_extraction( x, dilation = 8.0, method = 'v1', deform=True, verbose=False ):
    if get_global_version() == 0:
        return antspynet.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,1)
    else:
        if get_backend() == 'antspynet':
            brain_mask = antspynet.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,1)
        else:  
            brain_mask = antstorch.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,1) 
        return brain_mask

def label_hemispheres( x, template=None, templateLR=None, reg_iterations=[200,50,2,0] ):
    if template is None or templateLR is None:
        tfn = get_data('T_template0', target_extension='.nii.gz' )
        tfnw = get_data('T_template0_WMP', target_extension='.nii.gz' )
        tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )
        template = ants.image_read( tfn )
        
        if get_global_version() == 0 or get_backend() == 'antspynet':
            template = (template * antspynet.brain_extraction(template, 't1')).iMath( "Normalize" )
        else:
            template = (template * antstorch.brain_extraction(template, 't1')).iMath( "Normalize" )
        templateLR = ants.image_read( tlrfn )

    reg = ants.registration( ants.rank_intensity(x), ants.rank_intensity(template), 'antsRegistrationSyNQuickRepro[s]', aff_metric='GC', syn_metric='CC', syn_sampling=2, reg_iterations=reg_iterations, total_sigma=0.5, random_seed = 1 )
    return( ants.apply_transforms( x, templateLR, reg['fwdtransforms'], interpolator='nearestNeighbor') )

def deep_tissue_segmentation( x, template=None, registration_map=None, atropos_prior=None, sr_model=None ):
    if get_global_version() == 0:
        dapper = antspynet.deep_atropos( x, do_denoising=False )
    else:
        if get_backend() == 'antspynet':
            dapper = antspynet.deep_atropos( [x, None, None], do_denoising=False )
        else:
            dapper = antstorch.deep_atropos( [x, None, None] )    

    if atropos_prior is not None:
        msk = ants.threshold_image( dapper['segmentation_image'], 2, 3 ).iMath("GetLargestComponent",50)
        msk = ants.morphology( msk, "close", 2 )
        mskfull = ants.threshold_image( dapper['segmentation_image'], 1, 6 )
        mskfull = mskfull - msk
        priors = dapper['segmentation_image'][1:4] if get_global_version() == 0 else dapper['probability_images'][1:4]
        for k in range( len( priors ) ):
            priors[k]=ants.image_clone( priors[k]*msk )
        aap = ants.atropos( x, msk, i=priors, m='[0.0,1x1x1]', c = '[1,0]', priorweight=atropos_prior, verbose=1  )
        dapper['segmentation_image'] = aap['segmentation'] * msk + dapper['segmentation_image'] * mskfull
        if get_global_version() == 0:
            dapper['probability_images'][1:4] = aap['probabilityimages']
        else:
            dapper['probability_images'][1:4] = aap['probabilityimages']

    return dapper

def deep_brain_parcellation( target_image, template, img6seg = None, do_cortical_propagation=False, atropos_prior=None, verbose=True):
    if verbose: print("Begin registration")
    rig = ants.registration( template, ants.rank_intensity(target_image), "antsRegistrationSyNQuickRepro[a]", aff_iterations = (500,200,0,0), total_sigma=0.5, random_seed=1 )
    rigi = ants.apply_transforms( template, target_image, rig['fwdtransforms'])

    if verbose: print("Begin Atropos tissue segmentation")
    if img6seg is None:
        mydap = deep_tissue_segmentation( target_image  )
    else:
        mydap = { 'segmentation_image': img6seg, 'probability_images': None }

    if verbose: print("Begin DKT")
    if get_global_version() == 0 or get_backend() == 'antspynet':
        dkt = antspynet.desikan_killiany_tourville_labeling( target_image, do_preprocessing=True, return_probability_images=False, do_lobar_parcellation = True, do_denoising=False)
    else:
        dkt = antstorch.desikan_killiany_tourville_labeling( target_image, do_preprocessing=True, return_probability_images=False, do_lobar_parcellation = True)

    myhemiL = ants.threshold_image( dkt['lobar_parcellation'], 1, 6 )
    myhemiR = ants.threshold_image( dkt['lobar_parcellation'], 7, 12 )
    myhemi = myhemiL + myhemiR * 2.0
    brainmask = ants.threshold_image( mydap['segmentation_image'], 1, 6 )
    myhemi = ants.iMath( brainmask, 'PropagateLabelsThroughMask', myhemi, 100, 0)

    cortprop = None
    if do_cortical_propagation:
        cortprop = ants.threshold_image( mydap['segmentation_image'], 2, 2 )
        cortlab = dkt['segmentation_image'] * ants.threshold_image( dkt['segmentation_image'], 1000, 5000  )
        cortprop = ants.iMath( cortprop, 'PropagateLabelsThroughMask', cortlab, 1, 0)

    wmseg = ants.threshold_image( mydap['segmentation_image'], 3, 3 )
    wmMean = target_image[ wmseg == 1 ].mean()
    wmStd = target_image[ wmseg == 1 ].std()
    csfseg = ants.threshold_image( mydap['segmentation_image'], 1, 1 )
    csfStd = target_image[ csfseg == 1 ].std()
    
    return {
        "tissue_segmentation":mydap['segmentation_image'], "tissue_probabilities":mydap['probability_images'],
        "dkt_parcellation":dkt['segmentation_image'], "dkt_lobes":dkt['lobar_parcellation'],
        "dkt_cortex": cortprop, "hemisphere_labels": myhemi,
        "wmSNR": wmMean/wmStd, "wmcsfSNR": wmMean/csfStd, }

def deep_mtl(t1, sr_model=None, verbose=True):
    verbose = False
    if get_global_version() == 0 or get_backend() == 'antspynet': 
        template_fn = antspynet.get_antsxnet_data("deepFlashTemplateT1SkullStripped")
    else:
        template_fn = antstorch.get_antstorch_data("deepFlashTemplateT1SkullStripped")    
        
    template = ants.image_read(template_fn)
    registration = ants.registration(fixed=template, moving=t1, type_of_transform="antsRegistrationSyNQuickRepro[a]", total_sigma=0.5, verbose=verbose)
    template_transforms = dict(fwdtransforms=registration['fwdtransforms'], invtransforms=registration['invtransforms'])
    t1_warped = registration['warpedmovout']

    if get_global_version() == 0 or get_backend() == 'antspynet':
        df = antspynet.deep_flash(t1_warped, do_preprocessing=False, use_rank_intensity = True, verbose=verbose)
    else:
        df = antstorch.deep_flash(t1_warped, do_preprocessing=False, use_rank_intensity = True, verbose=verbose)    

    if sr_model is not None:
        newprobs = super_resolution_segmentation_with_probabilities( t1_warped, df['probability_images'], sr_model )
        df['probability_images'] = newprobs['sr_probabilities']

    relabeled_image = ants.apply_transforms( fixed=t1, moving=df['segmentation_image'], transformlist=template_transforms['invtransforms'], whichtoinvert=[True], interpolator="nearestNeighbor", verbose=verbose)
    mtl_description = map_segmentation_to_dataframe( 'mtl_description', relabeled_image )
    return { 'mtl_description':mtl_description, 'mtl_segmentation':relabeled_image }

def deep_nbm( t1, nbm_weights=None, binary_mask=None, deform=False, aged_template=False, csfquantile=None, reflect=False, verbose=False ):
    if get_global_version() == 0:
        import tensorflow as tf
        from tensorflow.keras.layers import Layer
        
        if not aged_template:
            refimg = ants.resample_image( ants.rank_intensity( ants.image_read( get_data( "CIT168_T1w_700um_pad", target_extension='.nii.gz' ))), [0.5,0.5,0.5] )
            refimgseg = ants.image_read( get_data( "CIT168_basal_forebrain", target_extension='.nii.gz' ))
            refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )
        else:
            refimg = ants.resample_image( ants.rank_intensity( ants.image_read( get_data( "CIT168_T1w_700um_pad_adni", target_extension='.nii.gz' ))), [0.5,0.5,0.5] )
            refimgseg = ants.image_read( get_data( "CIT168_basal_forebrain_adni", target_extension='.nii.gz' ))
            refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )

        pt_labels, group_labels, reflection_labels, crop_size = [1,2,3,4], [0,1,2,3,4,5,6,7,8], [0,2,1,6,7,8,3,4,5], [144,96,64]

        def nbmpreprocess( img, pt_labels, group_labels, masker=None, csfquantile=None, returndef=False, reflect=False ):
            imgr = ants.rank_intensity( img ) if masker is None else ants.rank_intensity( img, mask = masker )
            if csfquantile is not None and masker is None:
                masker = ants.threshold_image( imgr, np.quantile(imgr[imgr>1e-4], csfquantile ), 1e9 )
            if masker is not None: 
                imgr = imgr * masker
            imgrsmall = ants.resample_image( imgr, [1,1,1] )
            if not reflect:
                reg = ants.registration( refimgsmall, imgrsmall, 'antsRegistrationSyNQuickRepro[s]', reg_iterations = [200,200,20],total_sigma=0.5, verbose=False )
            else:
                myref = ants.reflect_image( imgrsmall, axis=0, tx='Translation' )
                reg = ants.registration( refimgsmall, imgrsmall, 'antsRegistrationSyNQuickRepro[s]', reg_iterations = [200,200,20], total_sigma=0.5, initial_transform = myref['fwdtransforms'][0], verbose=False )
            if not returndef:
                imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'][1], interpolator='linear' )
                imgseg = ants.apply_transforms( refimg, refimgseg, reg['invtransforms'][1], interpolator='nearestNeighbor' )
            else:
                imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'], interpolator='linear' )
                imgseg = ants.image_clone( refimgseg )
            binseg = ants.mask_image( imgseg, imgseg, pt_labels, binarize=True )
            imgseg = ants.mask_image( imgseg, imgseg, group_labels, binarize=False  )
            com = ants.get_center_of_mass( binseg )
            return { "img": imgraff, "seg": imgseg, "imgc": special_crop( imgraff, com, crop_size ), "segc": special_crop( imgseg, com, crop_size ), "reg" : reg, "mask": masker }

        nLabels, number_of_outputs, number_of_channels = len( group_labels ), len(group_labels), 1
        unet0 = antspynet.create_unet_model_3d( [None, None, None, number_of_channels ], number_of_outputs = 1, number_of_layers = 4, number_of_filters_at_base_layer = 32, convolution_kernel_size = 3, deconvolution_kernel_size = 2, pool_size = 2, strides = 2, dropout_rate = 0.0, weight_decay = 0, additional_options = "nnUnetActivationStyle", mode =  "sigmoid" )
        unet1 = antspynet.create_unet_model_3d( [None, None, None, 2], number_of_outputs=number_of_outputs, mode="classification", number_of_filters=(32, 64, 96, 128, 256), convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2), dropout_rate=0.0, weight_decay=0, additional_options = "nnUnetActivationStyle")

        class myConcat(Layer):
            def call(self, x): return tf.concat(x, axis=4 )
        nextin = myConcat()( [ unet0.inputs[0], unet0.outputs[0] ] )
        unetonnet = unet1( nextin )
        unet_model = tf.keras.models.Model( unet0.inputs, [ unetonnet,  unet0.outputs[0] ] )
        unet_model.load_weights( nbm_weights )

        imgprepro = nbmpreprocess( t1, pt_labels, group_labels, csfquantile = csfquantile, returndef = deform, reflect = reflect )

        def map_back( relo, t1, imgprepro, interpolator='linear', deform = False ):
            if not deform: return ants.apply_transforms( t1, relo, imgprepro['reg']['invtransforms'][0], whichtoinvert=[True], interpolator=interpolator )
            else: return ants.apply_transforms( t1, relo, imgprepro['reg']['invtransforms'], interpolator=interpolator )

        physspaceBF = imgprepro['imgc']
        tfarr1 = tf.cast( physspaceBF.numpy() ,'float32' )
        newshapeBF = list( tfarr1.shape )
        newshapeBF.insert(0,1)
        newshapeBF.insert(4,1)
        tfarr1 = tf.reshape(tfarr1, newshapeBF )
        snpred = unet_model.predict( tfarr1 )
        segpred, sigmoidpred = snpred[0], snpred[1]
        snpred1_image = ants.from_numpy( sigmoidpred[0,:,:,:,0] )
        snpred1_image = ants.copy_image_info( physspaceBF, snpred1_image )
        snpred1_image = map_back( snpred1_image, t1, imgprepro, 'linear', deform )
        bint = ants.threshold_image( snpred1_image, 0.5, 1.0 )
        probability_images = []
        for jj in range(number_of_outputs-1):
            temp = ants.from_numpy( segpred[0,:,:,:,jj+1] )
            temp = ants.copy_image_info( physspaceBF, temp )
            temp = map_back( temp, t1, imgprepro, 'linear', deform )
            probability_images.append( temp )
        image_matrix = ants.image_list_to_matrix(probability_images, bint)
        segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1)
        segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
        relabeled_image = ants.image_clone(segmentation_image)
        if not reflect:
            for i in range(1,len(group_labels)): relabeled_image[segmentation_image==(i)] = group_labels[i]
        else:
            for i in range(1,len(group_labels)): relabeled_image[segmentation_image==(i)] = reflection_labels[i]

        bfsegdesc = map_segmentation_to_dataframe( 'nbm3CH13', relabeled_image )
        return { 'segmentation':relabeled_image, 'description':bfsegdesc, 'mask': imgprepro['mask'], 'probability_images': probability_images }

    else:
        if get_backend() == 'antspynet':
            nbm = antspynet.nbm_labeling(t1, verbose=verbose)
        else:
            nbm = antstorch.nbm_labeling(t1, verbose=verbose)
        bfsegdesc = map_segmentation_to_dataframe( 'nbm3CH13', nbm['segmentation_image'] )
        return { 'segmentation':nbm['segmentation_image'], 'description':bfsegdesc, 'probability_images': nbm['probability_images'] }

def deep_cit168( t1, binary_mask=None, syn_type='antsRegistrationSyNQuickRepro[s]', priors=None, verbose=False):
    if get_global_version() == 0:
        import tensorflow as tf
        from tensorflow.keras.layers import Layer

        def tfsubset( x, indices ):
            with tf.device('/CPU:0'):
                outlist = []
                for k in indices: outlist.append( x[:,:,:,int(k)] )
                return tf.stack( outlist, axis=3 )

        registration = True
        cit168seg = t1 * 0
        myprior = ants.image_read(get_data("det_atlas_25_pad_LR_adni", target_extension=".nii.gz"))
        nbmtemplate = ants.image_read( get_data( "CIT168_T1w_700um_pad_adni", target_extension=".nii.gz" ) )
        nbmtemplate = ants.resample_image( nbmtemplate, [0.5,0.5,0.5] )
        templateSmall = ants.resample_image( nbmtemplate, [2.0,2.0,2.0] )
        orireg = ants.registration( fixed = templateSmall, moving = ants.iMath( t1, "Normalize" ), type_of_transform=syn_type, total_sigma=0.5, verbose=False )
        image = ants.apply_transforms( nbmtemplate, ants.iMath( t1, "Normalize" ), orireg['fwdtransforms'][1] )
        image = ants.iMath( image, "TruncateIntensity",0.001,0.999).iMath("Normalize")
        patchSize = [ 160,160,112 ]
        if priors is None: 
            priortosub = ants.apply_transforms( image, myprior, orireg['invtransforms'][1], interpolator='nearestNeighbor' )
        else: 
            priortosub = ants.apply_transforms( image, priors, orireg['fwdtransforms'][1], interpolator='nearestNeighbor' )
        bmask = ants.threshold_image( priortosub, 1, 999 )
        pt = list( ants.get_center_of_mass( bmask ) )
        pt[1] = pt[1] + 10.0  

        physspaceCIT = special_crop( image, pt, patchSize) 

        if binary_mask is not None:
            binary_mask_use = ants.apply_transforms( nbmtemplate, binary_mask, orireg['fwdtransforms'][1] )
            binary_mask_use = special_crop( binary_mask_use, pt, patchSize)

        for sn in [True,False]:
            if sn:
                group_labels = [0,7,8,9,23,24,25,33,34]
                newfn=get_data( "deepCIT168_sn", target_extension=".h5" )
            else:
                group_labels = [0,1,2,5,6,17,18,21,22]
                newfn=get_data( "deepCIT168", target_extension=".h5" )

            number_of_outputs, number_of_channels = len(group_labels), len(group_labels)

            unet0 = antspynet.create_unet_model_3d( [None, None, None, number_of_channels ], number_of_outputs = 1, number_of_layers = 4, number_of_filters_at_base_layer = 32, convolution_kernel_size = 3, deconvolution_kernel_size = 2, pool_size = 2, strides = 2, dropout_rate = 0.0, weight_decay = 0, additional_options = "nnUnetActivationStyle", mode =  "sigmoid" )
            unet1 = antspynet.create_unet_model_3d( [None, None, None, 2], number_of_outputs=number_of_outputs, mode="classification", number_of_filters=(32, 64, 96, 128, 256), convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2), dropout_rate=0.0, weight_decay=0, additional_options = "nnUnetActivationStyle")

            class mySplit(Layer):
                def call(self, x): return tf.split(x, 9, axis=4 )
            temp = mySplit()( unet0.inputs[0] )
            temp[1] = unet0.outputs[0]

            class myConcat(Layer):
                def call(self, x): return tf.concat(x, axis=4 )
            newmult = myConcat()( temp[0:2] )
            unetonnet = unet1( newmult )
            unet_model = tf.keras.models.Model( unet0.inputs, [ unetonnet,  unet0.outputs[0] ] )
            unet_model.load_weights( newfn )
            
            nbmprior = special_crop( priortosub, pt, patchSize).numpy() 
            imgnp = tf.reshape( physspaceCIT.numpy(), [160, 160, 112,1])
            nbmprior = tf.one_hot( nbmprior, 35 )
            nbmprior = tfsubset( nbmprior, group_labels[1:len(group_labels)] )
            imgnp = tf.reshape( tf.concat( [imgnp,nbmprior], axis=3), [1,160, 160, 112,9])

            nbmpred = unet_model.predict( imgnp )
            segpred, sigmoidpred = nbmpred[0], nbmpred[1]

            def map_back_cit( relo, t1, orireg, interpolator='linear' ):
                return ants.apply_transforms( t1, relo, orireg['invtransforms'][0], whichtoinvert=[True], interpolator=interpolator )

            nbmpred1_image = ants.from_numpy( sigmoidpred[0,:,:,:,0] )
            nbmpred1_image = ants.copy_image_info( physspaceCIT, nbmpred1_image )
            nbmpred1_image = map_back_cit( nbmpred1_image, t1, orireg, 'linear'  )
            if binary_mask is not None: 
                nbmpred1_image = nbmpred1_image * binary_mask
            bint = ants.threshold_image( nbmpred1_image, 0.5, 1.0 )

            probability_images = []
            for jj in range(1,len(group_labels)):
                temp = ants.from_numpy( segpred[0,:,:,:,jj] )
                temp = ants.copy_image_info( physspaceCIT, temp )
                temp = map_back_cit( temp, t1, orireg, 'linear'  )
                probability_images.append( temp )

            image_matrix = ants.image_list_to_matrix(probability_images, bint)
            segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1)
            segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
            relabeled_image = ants.image_clone(segmentation_image*0.)
            for i in np.unique(segmentation_image.numpy()):
                if i > 0 :
                    temp = ants.threshold_image(segmentation_image,i,i)
                    if group_labels[int(i)] < 33: 
                        temp = ants.iMath( temp, "GetLargestComponent",1)
                    relabeled_image = relabeled_image + temp*group_labels[int(i)]
            cit168seg = cit168seg + relabeled_image

        cit168segdesc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', cit168seg ).dropna(axis=0)
        return { 'segmentation':cit168seg, 'description':cit168segdesc }

    else:
        if get_backend() == 'antspynet':
            cit168seg = antspynet.cit168_labeling(t1, verbose=verbose)
        else:
            cit168seg = antstorch.cit168_labeling(t1, verbose=verbose)
        cit168segdesc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', cit168seg ).dropna(axis=0)
        return { 'segmentation':cit168seg, 'description':cit168segdesc }

def minimal_sr_preprocessing( x, imgbxt=None ):
    if x.dimension != 3: raise ValueError('hierarchical: input image should be 3-dimensional')

    tfn = get_data('T_template0', target_extension='.nii.gz' ), 
    tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )
    templatea = ants.image_read( tfn )
    
    if get_global_version() == 0:
        templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
    else:
        if get_backend() == 'antspynet': 
            template_mask = antspynet.brain_extraction( templatea, 't1' )
        else: 
            template_mask = antstorch.brain_extraction( templatea, 't1' )    
        templatea = ( templatea * template_mask ).iMath( "Normalize" )
        
    templatealr = ants.image_read( tlrfn )
    if imgbxt is None: 
        imgbxt = brain_extraction(x)
        
    img = preprocess_intensity( ants.iMath( x, "Normalize" ), imgbxt )
    img = ants.iMath( img, "Normalize" )
    mylr = label_hemispheres( img, templatea, templatealr )
    return img, mylr * imgbxt

def hierarchical( x, output_prefix, labels_to_register=[2,3,4,5], imgbxt=None, img6seg=None, cit168=False, is_test=False, atropos_prior=None, sr_model=None, verbose=True ):
    if x.dimension != 3: 
        raise ValueError('hierarchical: input image should be 3-dimensional')

    tfn = get_data('T_template0', target_extension='.nii.gz' )
    tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )

    if get_global_version() == 0 or get_backend() == 'antspynet': 
        bfn = antspynet.get_antsxnet_data( "croppedMni152" )
    else: 
        bfn = antstorch.get_antstorch_data( "croppedMni152" )    

    templatea = ants.image_read( tfn )
    if get_global_version() == 0:
        templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
    else:
        if get_backend() == 'antspynet': 
            template_brain_mask = antspynet.brain_extraction( templatea, 't1' )    
        else: 
            template_brain_mask = antstorch.brain_extraction( templatea, 't1' ) 
        templatea = ( templatea * template_brain_mask ).iMath( "Normalize" )

    templatealr = ants.image_read( tlrfn )
    templateb = ants.image_read( bfn )
    
    if get_global_version() == 0:
        templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
    else:
        if get_backend() == 'antspynet': 
            template_brain_mask = antspynet.brain_extraction( templateb, 't1' )    
        else: 
            template_brain_mask = antstorch.brain_extraction( templateb, 't1' ) 
        templateb = ( templateb * template_brain_mask ).iMath( "Normalize" )

    if imgbxt is None:
        if get_global_version() == 0:
            imgbxt =  antspynet.brain_extraction( x, modality="t1threetissue")['segmentation_image'].threshold_image(1,1)
        else:
            if get_backend() == 'antspynet': 
                imgbxt = antspynet.brain_extraction( x, modality="t1threetissue")['segmentation_image']
            else: 
                imgbxt = antstorch.brain_extraction( x, modality="t1threetissue")['segmentation_image']     
            imgbxt = ants.threshold_image( imgbxt, 1, 1 )  
        img = preprocess_intensity( ants.iMath( x, "Normalize" ), imgbxt )
    else:
        img = ants.iMath( x, "Normalize" )

    templatesmall = ants.resample_image( templateb, (91,109,91), use_voxels=True )
    myqc = inspect_raw_t1( img, output_prefix=output_prefix, option='brain' )
    img = ants.iMath( img, "Normalize" )

    myparc = deep_brain_parcellation( x, templateb, img6seg = img6seg, atropos_prior = atropos_prior, do_cortical_propagation = not is_test, verbose=verbose )

    cit168lab, cit168reg, cit168lab_desc = None, None, None
    cit168adni = ants.image_read( get_data( "CIT168_T1w_700um_pad_adni",target_extension='.nii.gz') ).iMath("Normalize")
    cit168labT = ants.image_read( get_data( "det_atlas_25_pad_LR_adni", target_extension='.nii.gz' ) )
    cit168labStem = ants.image_read( get_data( "CIT168_T1w_700um_pad_adni_brainstem", target_extension='.nii.gz' ) )

    cit168reg = region_reg( input_image = img, input_image_tissue_segmentation=myparc['tissue_segmentation'], input_image_region_segmentation=imgbxt, input_template=cit168adni, input_template_region_segmentation=ants.threshold_image( cit168adni, 0.15, 1 ), output_prefix=output_prefix + "_CIT168RRSYN", padding=10, total_sigma=0.5, is_test=not cit168 )['synL']
    cit168lab = ants.apply_transforms( img, cit168labT, cit168reg['invtransforms'], interpolator = 'nearestNeighbor' )
    cit168lab_desc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', cit168lab ).dropna(axis=0)

    mylr = label_hemispheres( img, templatea, templatealr )

    hemi = map_segmentation_to_dataframe( "hemisphere", myparc['hemisphere_labels'] )
    tissue = map_segmentation_to_dataframe( "tissues", myparc['tissue_segmentation'] )
    dktl = map_segmentation_to_dataframe( "lobes", myparc['dkt_lobes'] )
    dktp = map_segmentation_to_dataframe( "dkt", myparc['dkt_parcellation'] )
    dktc = map_segmentation_to_dataframe( "dkt", myparc['dkt_cortex'] ) if not is_test else None

    tissue_seg_png = output_prefix + "_seg.png"
    ants.plot( img*255, myparc['tissue_segmentation'], axis=2, nslices=21, ncol=7, alpha=0.6, filename=tissue_seg_png, crop=True, black_bg=False )

    myicv = icv( x )
    braintissuemask =  ants.threshold_image( myparc['tissue_segmentation'], 2, 6 )
    
    myhypo, hippLR = None, None
    if get_global_version() == 0:
        deep_bf = deep_nbm( img * braintissuemask, get_data("deep_nbm_rank",target_extension='.h5'), csfquantile=None, aged_template=True, verbose=verbose )
    else:
        deep_bf = deep_nbm( img * braintissuemask, verbose=verbose )

    if is_test:
        mydataframes = { "icv": myicv, "rbp": myqc['brain'], "tissues":tissue, "dktlobes":dktl, "dktregions":dktp, "dktcortex":dktc, "bf":deep_bf['description'] }
        outputs = { "brain_n4_dnz": img, "brain_extraction": imgbxt, "left_right": mylr, "dkt_parc": myparc, "bf":deep_bf['segmentation'], "dataframes": mydataframes }
        return outputs

    wm_tractsL, wm_tractsR, wmtdfL, wmtdfR, reg = None, None, None, None, None
    if labels_to_register is not None and not is_test:
        reg = hemi_reg( input_image = img, input_image_tissue_segmentation = myparc['tissue_segmentation'], input_image_hemisphere_segmentation = mylr, input_template=templatea, input_template_hemisphere_labels=templatealr, output_prefix = output_prefix + "_SYN", labels_to_register = labels_to_register, is_test=is_test )
        wm_tracts = ants.image_read( get_data( "wm_major_tracts", target_extension='.nii.gz' ) )
        wm_tractsL = ants.apply_transforms( img, wm_tracts, reg['synL']['invtransforms'], interpolator='nearestNeighbor' ) * ants.threshold_image( mylr, 1, 1  )
        wm_tractsR = ants.apply_transforms( img, wm_tracts, reg['synR']['invtransforms'], interpolator='nearestNeighbor' ) * ants.threshold_image( mylr, 2, 2  )
        wmtdfL = map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsL )
        wmtdfR = map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsR )

    deep_flash = deep_mtl(img, sr_model = sr_model )

    if get_global_version() == 0:
        deep_cit = deep_cit168( img, priors = cit168lab, binary_mask = braintissuemask )
    else:
        deep_cit = deep_cit168( img, verbose=verbose )

    if not is_test:
        tbinseg = ants.mask_image( cit168labT, cit168labT, [7,9,23,25,33,34], binarize=True)
        tbinseg = ants.morphology( tbinseg, "dilate", 14 )
        ibinseg = ants.apply_transforms( img, tbinseg, cit168reg['invtransforms'], interpolator='nearestNeighbor')
        snreg = region_reg( img, myparc['tissue_segmentation'], ibinseg, cit168adni, tbinseg, output_prefix=output_prefix + "_SNREG", padding = 4, is_test=False )['synL']
        tbinseg = ants.mask_image( cit168labT, cit168labT, [7,9,23,25,33,34], binarize=False)
        snseg = ants.apply_transforms( img, tbinseg, snreg['invtransforms'], interpolator = 'nearestNeighbor' )
        snseg = snseg * ants.threshold_image( myparc['tissue_segmentation'], 2, 6 )
        snseg_desc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', snseg ).dropna(axis=0)
    else :
        snseg, snseg_desc = None, None

    brainstemseg = ants.apply_transforms( img, cit168labStem, cit168reg['invtransforms'], interpolator = 'nearestNeighbor' )
    brainstemseg = brainstemseg * braintissuemask
    brainstem_desc = map_segmentation_to_dataframe( 'CIT168_T1w_700um_pad_adni_brainstem', brainstemseg )
    brainstem_desc = brainstem_desc.loc[:, ~brainstem_desc.columns.str.contains('^Side')]

    if get_global_version() == 0 or get_backend() == 'antspynet':
        cereb = antspynet.cerebellum_morphology( x, compute_thickness_image=False, verbose=False, do_preprocessing=True )
        maskc = ants.threshold_image(cereb['cerebellum_probability_image'], 0.5, 1, 1, 0)
        cereb = antspynet.cerebellum_morphology( x, cerebellum_mask=maskc, compute_thickness_image=False, verbose=False, do_preprocessing=True )
    else:   
        cereb = antstorch.cerebellum_morphology( x, compute_thickness_image=False, verbose=False, do_preprocessing=True )
        maskc = ants.threshold_image(cereb['cerebellum_probability_image'], 0.5, 1, 1, 0)
        cereb = antstorch.cerebellum_morphology( x, cerebellum_mask=maskc, compute_thickness_image=False, verbose=False, do_preprocessing=True )

    cereb_desc = map_segmentation_to_dataframe( 'cerebellum', cereb['parcellation_segmentation_image'] ).dropna(axis=0)

    mydataframes = {
        "rbp": myqc['brain'], "icv": myicv, "hemispheres":hemi, "tissues":tissue, "dktlobes":dktl,
        "dktregions":dktp, "dktcortex":dktc, "wmtracts_left":wmtdfL, "wmtracts_right":wmtdfR,
        "mtl":deep_flash['mtl_description'], "bf":deep_bf['description'], "cit168":cit168lab_desc,
        "deep_cit168":deep_cit['description'], "snseg":snseg_desc, "brainstem": brainstem_desc, "cerebellum": cereb_desc }
        
    outputs = {
        "brain_n4_dnz": img, "brain_n4_dnz_png": myqc['brain_image'], "brain_extraction": imgbxt,
        "tissue_seg_png": tissue_seg_png, "left_right": mylr, "dkt_parc": myparc, "registration":reg,
        "wm_tractsL":wm_tractsL, "wm_tractsR":wm_tractsR, "mtl":deep_flash['mtl_segmentation'],
        "bf":deep_bf['segmentation'], "deep_cit168lab": deep_cit['segmentation'], "cit168lab": cit168lab,
        "cit168reg": cit168reg, "snseg":snseg, "snreg":snreg, "brainstem": brainstemseg,
        "cerebellum":cereb['parcellation_segmentation_image'], "dataframes": mydataframes }

    if get_global_version() == 0:
        if myhypo is not None:
            mydataframes["wmh"] = myhypo['wmh_summary']
            outputs["white_matter_hypointensity"] = myhypo['wmh_probability_image']
        if hippLR is not None:
            mydataframes["hippLR"] = hippLR['description']
            outputs["hippLR"] = hippLR['segmentation']

    return outputs

