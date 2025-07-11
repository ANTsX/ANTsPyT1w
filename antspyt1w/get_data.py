"""
Get local ANTsPyT1w data
"""

__all__ = ['get_data','map_segmentation_to_dataframe','hierarchical',
    'random_basis_projection', 'deep_dkt','deep_hippo','deep_tissue_segmentation',
    'deep_brain_parcellation', 'deep_mtl', 'label_hemispheres','brain_extraction',
    'hemi_reg', 'region_reg', 't1_hypointensity', 'zoom_syn', 'merge_hierarchical_csvs_to_wide_format',
    'map_intensity_to_dataframe', 'deep_nbm', 'map_cit168', 'deep_cit168','minimal_sr_preprocessing']

from pathlib import Path
import os
from os.path import exists
import pandas as pd
import math
import os.path
from os import path
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
import antspynet
import tensorflow as tf
from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.antspyt1w/')
# Internal global variable to hold the seed
_GLOBAL_SCIENTIFIC_SEED = None

def set_global_scientific_computing_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set a global seed for reproducibility across Python, NumPy, and scientific ML frameworks.

    Parameters
    ----------
    seed : int
        The seed to use for all random number generators.
    deterministic : bool
        If True, sets flags for deterministic behavior in supported libraries (PyTorch, TensorFlow).
    """
    global _GLOBAL_SCIENTIFIC_SEED
    _GLOBAL_SCIENTIFIC_SEED = seed

    # Python's built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Set Python hash seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    # PyTorch
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

    # JAX
    try:
        import jax
        jax_key = jax.random.PRNGKey(seed)
        globals()["_jax_key"] = jax_key
    except ImportError:
        pass

    warnings.warn("Remember to set 'random_state=seed' in scikit-learn models.", stacklevel=2)
    print(f"[INFO] Global scientific computing seed set to {seed}")


def get_global_scientific_computing_random_seed():
    """
    Retrieve the most recently set global seed.

    Returns
    -------
    int or None
        The seed set by `set_global_scientific_computing_seed()`, or None if not yet set.
    """
    return _GLOBAL_SCIENTIFIC_SEED

set_global_scientific_computing_random_seed(1234)

def get_data(name=None, force_download=False, version=46, target_extension='.csv'):
    """
    Get ANTsPyT1w data filename

    The first time this is called, it will download data to ~/.antspyt1w.
    After, it will just read data from disk.  The ~/.antspyt1w may need to
    be periodically deleted in order to ensure data is current.

    Arguments
    ---------
    name : string
        name of data tag to retrieve
        Options:
            - 'all'
            - 'dkt'
            - 'hemisphere'
            - 'lobes'
            - 'tissues'
            - 'T_template0'
            - 'T_template0_LR'
            - 'T_template0_LobesBstem'
            - 'T_template0_WMP'
            - 'T_template0_Symmetric'
            - 'T_template0_SymmetricLR'
            - 'PPMI-3803-20120814-MRI_T1-I340756'
            - 'simwmseg'
            - 'simwmdisc'
            - 'wmh_evidence'
            - 'wm_major_tracts'

    force_download: boolean

    version: version of data to download (integer)

    Returns
    -------
    string
        filepath of selected data

    Example
    -------
    >>> import ants
    >>> ppmi = ants.get_ants_data('ppmi')
    """
    import os
    import shutil
    from pathlib import Path
    import tensorflow as tf
    DATA_PATH = os.path.join(os.path.expanduser('~'), '.antspyt1w')
    os.makedirs(DATA_PATH, exist_ok=True)

    def mv_subfolder_files(folder, verbose=False):
        """
        Move files from subfolders to the parent folder.

        Parameters
        ----------
        folder : str
            Path to the folder.
        verbose : bool, optional
            Print information about the files and folders being processed (default is False).

        Returns
        -------
        None
        """
        import os
        import shutil
        for root, dirs, files in os.walk(folder):
            if verbose:
                print(f"Processing directory: {root}")
                print(f"Subdirectories: {dirs}")
                print(f"Files: {files}")
            
            for file in files:
                if root != folder:
                    if verbose:
                        print(f"Moving file: {file} from {root} to {folder}")
                    shutil.move(os.path.join(root, file), folder)
            
            for dir in dirs:
                if root != folder:
                    if verbose:
                        print(f"Removing directory: {dir} from {root}")
                    shutil.rmtree(os.path.join(root, dir))

    def download_data(version):
        url = "https://ndownloader.figshare.com/articles/14766102/versions/" + str(version)
        target_file_name = "14766102.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract=True)
        os.remove(os.path.join(DATA_PATH, target_file_name))

    if force_download:
        download_data(version=version)

    mv_subfolder_files( os.path.expanduser("~/.antspyt1w"), False )

    # Move files from subdirectories to the main directory
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
        os.listdir(DATA_PATH)
        warnings.warn("datapath in get_data is None - must be some issue in downloading the data from figshare. try calling get_data in the setup of your system.")
    return datapath


def map_segmentation_to_dataframe( segmentation_type, segmentation_image ):
    """
    Match the segmentation to its appropriate data frame.  We do not check
    if the segmentation_type and segmentation_image match; this may be indicated
    by the number of missing values on output eg in column VolumeInMillimeters.

    Arguments
    ---------
    segmentation_type : string
        name of segmentation_type data frame to retrieve
        Options:
            - 'dkt'
            - 'lobes'
            - 'tissues'
            - 'hemisphere'
            - 'wm_major_tracts'

    segmentation_image : antsImage with same values (or mostly the same) as are
        expected by segmentation_type

    Returns
    -------
    dataframe

    """
    mydf_fn = get_data( segmentation_type )
    mydf = pd.read_csv( mydf_fn )
    mylgo = ants.label_geometry_measures( segmentation_image )
    return pd.merge( mydf, mylgo, how='left', on=["Label"] )


def map_intensity_to_dataframe(segmentation_type, intensity_image, segmentation_image):
    """
    Match intensity values within segmentation labels to an appropriate data frame.

    Arguments
    ---------
    segmentation_type : string or pandas.DataFrame
        If string:
            Name of the segmentation type data frame to retrieve using get_data.
            Options include data available via get_data (e.g., "lobes").
        If pandas.DataFrame:
            A pre-existing DataFrame containing label information.

    intensity_image : ants.ANTsImage
        ANTsImage with intensity values to summarize.

    segmentation_image : ants.ANTsImage
        ANTsImage with labels (or mostly the same values) as expected by segmentation_type.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the intensity values within segmentation labels.
    """
    if isinstance(segmentation_type, str):
        # Retrieve the data frame based on the segmentation type string
        mydf_fn = get_data(segmentation_type)
        mydf = pd.read_csv(mydf_fn)
    elif isinstance(segmentation_type, pd.DataFrame):
        # Use the provided DataFrame
        mydf = segmentation_type
    else:
        raise ValueError("segmentation_type must be either a string or a pandas DataFrame")

    # Get label statistics for the intensity and segmentation images
    mylgo = ants.label_stats(intensity_image, segmentation_image)

    # Rename the label column to match expected name for merging
    mylgo = mylgo.rename(columns={'LabelValue': 'Label'})

    # Merge the segmentation DataFrame with the label statistics
    result = pd.merge(mydf, mylgo, how='left', on=["Label"])
    
    return result


def myproduct(lst):
    return( functools.reduce(mul, lst) )

def mahalanobis_distance( x ):
    """
    Calculate mahalanobis distance from a dataframe

    Arguments
    ---------
    x : dataframe of random projections or other data

    Returns
    -------
    dictionary of distances and outlierness categories

    """
    # M-Distance
    x_minus_mu = x - np.mean(x)
    cov = np.cov(x.values.T)                           #Covariance
    inv_covmat = sp.linalg.inv(cov)                     #Inverse covariance
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    #Flag as outlier
    outlier = []
    #Cut-off point
    C = np.sqrt(sp.stats.chi2.ppf((1-0.001), df=x.shape[1]))    #degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return { "distance": md, "outlier": outlier }


def patch_eigenvalue_ratio( x, n, radii, evdepth = 0.9, mask=None, standardize=False ):
    """
    Patch-based eigenvalue ratio calculation.  Computes the ratio (percent) of the eigenvalues needed to reach evdepth explained variance.

    Arguments
    ---------
    x : input image - if t1, likely better if brain extracted

    n : number of patches

    radii : list of radius values

    evdepth : value in zero to one

    mask : optional antsImage

    standardize : boolean standardizes the patch matrix if True (subtract mean)

    Returns
    -------
    an eigenvalue ratio in the range of [0,1]

    """
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
    # u, s, v = svds(X , min(X.shape)-1 )
    thespectrum = np.linalg.svd( X, compute_uv=False )
    spectralsum = thespectrum.sum()
    targetspec = spectralsum * evdepth
    spectralcumsum = np.cumsum( thespectrum )
    numer = np.argmin(  abs( spectralcumsum - evdepth * spectralsum ) )
    denom = len( thespectrum )
    # print( str(numer) +  " " + str(denom) )
    return numer/denom

def loop_outlierness( random_projections, reference_projections=None,
    standardize=True, extent=3, n_neighbors=24, cluster_labels=None ):
    """
    Estimate loop outlierness for the input.

    Arguments
    ---------
    random_projections : output of random_basis_projection or a similar function

    reference_projections : produced in the same way as random_projections

    standardize: boolean will subtract mean and divide by sd of the reference data

    extent: an integer value [1, 2, 3] that controls the statistical
        extent, e.g. lambda times the standard deviation from the mean (optional,
        default 2)

    n_neighbors: the total number of neighbors to consider w.r.t. each
        sample (optional, default 10)

    cluster_labels: a numpy array of cluster assignments w.r.t. each
        sample (optional, default None)

    Returns
    -------
    loop outlierness probability

    """
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
        m = loop.LocalOutlierProbability(normalized_df,
            extent=extent,
            n_neighbors=n_neighbors ).fit()
    else:
        m = loop.LocalOutlierProbability(normalized_df,
            extent=extent,
            n_neighbors=n_neighbors,
            cluster_labels=cluster_labels).fit()
    scores = m.local_outlier_probabilities

    return scores



def random_basis_projection( x, template,
    type_of_transform='Similarity',
    refbases = None,
    nBasis=10, random_state = 99 ):
    """
    Produce unbiased data descriptors for a given image which can be used
    to assist data inspection and ranking.  can be used with any image
    brain extracted or not, any modality etc.   but we assume we can
    meaningfully map to a template, at least with a low-dimensional
    transformation, e.g. Translation, Rigid, Similarity.

    Arguments
    ---------
    x : antsImage

    template : antsImage reference template

    type_of_transform: one of Translation, Rigid, Similarity, Affine

    refbases : reference bases for outlierness calculations

    nBasis : number of variables to derive

    random_state : seed

    Returns
    -------
    dataframe with projections and an outlierness estimate.

    the outlierness estimate is based on a small reference dataset of young controls.
    the user/researcher may want to use a different reference set.  see the
    function loop_outlierness for one way to do that.

    """
    template = ants.crop_image( template )
    template = ants.iMath( template, "Normalize" )
    np.random.seed(int(random_state))
    nvox = template.shape
    # X = np.random.rand( nBasis+1, myproduct( nvox ) )
    # u, s, randbasis = svds(X, k=nBasis)
    # if randbasis.shape[1] != myproduct(nvox):
    #    raise ValueError("columns in rand basis do not match the nvox product")

    randbasis = np.random.randn( myproduct( nvox ), nBasis  )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    norm = ants.iMath( x, "Normalize" )
    trans = ants.registration( template, norm,
        type_of_transform='antsRegistrationSyNQuickRepro[t]' )
    resamp = ants.registration( template, norm,
        type_of_transform=type_of_transform, total_sigma=0.5,
        # aff_metric='GC',
        random_seed=1, initial_transform=trans['fwdtransforms'][0] )['warpedmovout']
#    mydelta = ants.from_numpy( ( resamp - template ).abs() )
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


def resnet_grader( x, weights_filename = None ):
    """
    Supervised grader / scoring of t1 brain

    Arguments
    ---------

    x : antsImage of t1 brain

    weights_filename : optional weights filename

    Returns
    -------
    two letter grades

    """

    if weights_filename is None:
        weights_filename=get_data( 'resnet_grader', target_extension='.h5' )

    if not exists( weights_filename ):
        print("resnet_grader weights do not exist: " + weights_filename )
        return None

    mdl = antspynet.create_resnet_model_3d( [None,None,None,1],
        lowest_resolution = 32,
        number_of_outputs = 4,
        cardinality = 1,
        squeeze_and_excite = False )
    mdl.load_weights( weights_filename )


    t1 = ants.iMath( x - x.min(),  "Normalize" )
    bxt = ants.threshold_image( t1, 0.01, 1.0 )
    t1 = ants.rank_intensity( t1, mask=bxt, get_mask=True )
    templateb = ants.image_read( get_data( "S_template3_brain", target_extension='.nii.gz' ) )
    templateb = ants.crop_image( templateb ).resample_image( [1,1,1] )
    templateb = ants.pad_image_by_factor( templateb, 8 )
    templatebsmall = ants.resample_image( templateb, [2,2,2] )
    reg = ants.registration( templatebsmall, t1, 'Similarity', verbose=False )
    ilist = list()
    refimg=templateb
    ilist.append( [refimg] )
    nsim = 16
    uu = ants.randomly_transform_image_data( refimg, ilist,
            number_of_simulations = nsim,
            transform_type='scaleShear', sd_affine=0.075 )
    fwdaffgd = ants.read_transform( reg['fwdtransforms'][0])
    scoreNums = np.zeros( 4 )
    scoreNums[3]=0
    scoreNums[2]=1
    scoreNums[1]=2
    scoreNums[0]=3
    scoreNums=scoreNums.reshape( [4,1] )

    def get_grade( score, probs ):
        grade='f'
        if score >= 2.25:
                grade='a'
        elif score >= 1.5:
                grade='b'
        elif score >= 0.75:
                grade='c'
        probgradeindex = np.argmax( probs )
        probgrade = ['a','b','c','f'][probgradeindex]
        return [grade, probgrade, float( score )]

    gradelistNum = []
    gradelistProb = []
    gradeScore = []
    for k in range( nsim ):
            simtx = uu['simulated_transforms'][k]
            cmptx = ants.compose_ants_transforms( [simtx,fwdaffgd] ) # good
            subjectsim = ants.apply_ants_transform_to_image( cmptx, t1, refimg, interpolation='linear' )
            subjectsim = ants.add_noise_to_image( subjectsim, 'additivegaussian', (0,0.01) )
            xarr = subjectsim.numpy()
            newshape = list( xarr.shape )
            newshape = [1] + newshape + [1]
            xarr = np.reshape(  xarr, newshape  )
            preds = mdl.predict( xarr )
            predsnum = tf.matmul(  preds, scoreNums )
            locgrades = get_grade( predsnum, preds )
            gradelistNum.append( locgrades[0] )
            gradelistProb.append( locgrades[1] )
            gradeScore.append( locgrades[2] )

    def most_frequent(List):
        return max(set(List), key = List.count)

    mydf = pd.DataFrame( {
            "NumericGrade": gradelistNum,
            "ProbGrade": gradelistProb,
            "NumericScore": gradeScore,
            'grade': most_frequent( gradelistProb )
        })

    smalldf = pd.DataFrame( {
        'gradeLetter':  [mydf.grade[0]],
        'gradeNum': [mydf.NumericScore.mean()]
        }, index=[0] )
    # print( mydf.Num.value_counts() )
    # print( mydf.Prob.value_counts() )
    return smalldf



def inspect_raw_t1( x, output_prefix, option='both' ):
    """
    Quick inspection and visualization of a raw T1 whole head image,
    whole head and brain or both.  The reference data was developed with
    option both and this will impact results.  For the head image, outlierness
    is estimated vi a quick extraction from background.  Variability in this
    extraction (Otsu) may lead to reliability issues.

    Arguments
    ---------

    x : antsImage of t1 whole head

    output_prefix: a path and prefix for outputs

    option : string both, brain or head

    Returns
    -------
    two dataframes (head, brain) with projections and outlierness estimates.

    """

    if x.dimension != 3:
        raise ValueError('inspect_raw_t1: input image should be 3-dimensional')

    x = ants.iMath( x, "Normalize" )
    csvfn = output_prefix + "head.csv"
    pngfn = output_prefix + "head.png"
    csvfnb = output_prefix + "brain.csv"
    pngfnb = output_prefix + "brain.png"

    # reference bases
    rbh = pd.read_csv( get_data( "refbasis_head", target_extension=".csv" ) )
    rbb = pd.read_csv( get_data( "refbasis_brain", target_extension=".csv" ) )

    # whole head outlierness
    rbp=None
    if option == 'both' or option == 'head':
        bfn = antspynet.get_antsxnet_data( "S_template3" )
        templateb = ants.image_read( bfn ).iMath("Normalize")
        templatesmall = ants.resample_image( templateb, (2,2,2), use_voxels=False )
        lomask = ants.threshold_image( x, "Otsu", 2 ).threshold_image(1,2)
        t1 = ants.rank_intensity( x * lomask, mask=lomask, get_mask=False )
        ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfn, crop=True )
        rbp = random_basis_projection( t1, templatesmall,
            type_of_transform='Rigid',
            refbases=rbh )
        rbp.to_csv( csvfn )
        # fix up the figure
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

    # same for brain
    rbpb=None
    evratio=None
    if option == 'both' or option == 'brain':
        if option == 'both':
            t1 = ants.iMath( x, "TruncateIntensity",0.001, 0.999).iMath("Normalize")
            lomask = antspynet.brain_extraction( t1, "t1" )
            t1 = ants.rank_intensity( t1 * lomask, get_mask=True )
        else:
            t1 = ants.iMath( x, "Normalize" )
            t1 = ants.rank_intensity( t1, get_mask=True )
        ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfnb, crop=True )
        templateb = ants.image_read( get_data( "S_template3_brain", target_extension='.nii.gz' ) )
        templatesmall = ants.resample_image( templateb, (2,2,2), use_voxels=False )
        rbpb = random_basis_projection( t1,
            templatesmall,
            type_of_transform='Rigid',
            refbases=rbb )
        rbpb['evratio'] = patch_eigenvalue_ratio( t1, 512, [20,20,20], evdepth = 0.9 )
        grade0 = resnet_grader( t1 ).gradeNum[0]
        msk=ants.threshold_image(t1,0.01,1.0)
        t1tx=ants.n4_bias_field_correction( t1, mask=msk )
        t1tx=ants.iMath(t1tx,'TruncateIntensity',0.001,0.98)
        grade1 = resnet_grader( t1tx ).gradeNum[0]
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

    return {
        "head": rbp,
        "head_image": pngfn,
        "brain": rbpb,
        "brain_image": pngfnb,
        }

def icv( x ):
    """
    Intracranial volume (ICV) estimation from a brain extracted image

    Arguments
    ---------
    x : antsImage raw t1

    Returns
    -------
    ICV in cubic millimeters

    """
    icvseg = antspynet.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,2)
    np.product = np.prod( ants.get_spacing( x ) ) * icvseg.sum()
    return pd.DataFrame({'icv': [np.product]})

def brain_extraction( x, dilation = 8.0, method = 'v1', deform=True, verbose=False ):
    """
    quick brain extraction for individual images

    x: input image

    dilation: amount to dilate first brain extraction in millimeters

    method: version currently v0 or any other string gives two different results

    deform: map the image to the training hypersphere before running the extraction

    verbose: boolean

    """
    return antspynet.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,1)

    if deform:
        reorient_template_file_name_path = antspynet. get_antsxnet_data("S_template3" )
        template = ants.image_read( reorient_template_file_name_path )
        reg = ants.registration( template, x, 'antsRegistrationSyNQuickRepro[s]', 
             total_sigma=0.5 )

    closedilmm = 5.0
    spacing = ants.get_spacing(x)
    spacing_product = spacing[0] * spacing[1] * spacing[2]
    spcmin = min( spacing )
    dilationRound = int(np.round( dilation / spcmin ))
    closedilRound = int(np.round( closedilmm / spcmin ))
    if deform:
        xn3 = ants.n3_bias_field_correction( reg['warpedmovout'], 8 ).n3_bias_field_correction( 4 )
        xn3 = ants.iMath(xn3, "TruncateIntensity",0.001,0.999).iMath("Normalize")
    else:
        xn3 = ants.n3_bias_field_correction( x, 8 ).n3_bias_field_correction( 4 )
        xn3 = ants.iMath(xn3, "TruncateIntensity",0.001,0.999).iMath("Normalize")
    if method == 'v0':
        if verbose:
            print("method v0")
        bxtmethod = 't1combined[' + str(closedilRound) +']' # better for individual subjects
        bxt = antspynet.brain_extraction( xn3, bxtmethod ).threshold_image(2,3).iMath("GetLargestComponent",50).iMath("FillHoles")
        if deform:
            bxt = ants.apply_transforms( x, bxt, reg['invtransforms'], interpolator='nearestNeighbor' )
        return bxt
    if method == 'v1':
        if verbose:
            print("method v1")
        bxt = antspynet.brain_extraction( xn3, 't1' ).threshold_image(0.5,3).iMath("GetLargestComponent",50).iMath("FillHoles")
        if deform:
            bxt = ants.apply_transforms( x, bxt, reg['invtransforms'], interpolator='nearestNeighbor' )
        return bxt
    if verbose:
        print("method candidate")
    bxt0 = antspynet.brain_extraction( xn3, "t1" ).threshold_image(0.5,1.0).iMath("GetLargestComponent",50).morphology( "close", closedilRound ).iMath("FillHoles")
    bxt0dil = ants.iMath( bxt0, "MD", dilationRound )
    image = ants.iMath( xn3 * bxt0dil,"Normalize")*255
    # no no brainer
    model = antspynet.create_nobrainer_unet_model_3d((None, None, None, 1))
    weights_file_name = antspynet.get_pretrained_network("brainExtractionNoBrainer")
    model.load_weights(weights_file_name)
    image_array = image.numpy()
    image_robust_range = np.quantile(image_array[np.where(image_array != 0)], (0.02, 0.98))
    threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]
    thresholded_mask = ants.threshold_image(image, -10000, threshold_value, 0, 1)
    thresholded_image = image * thresholded_mask
    image_resampled = ants.resample_image(thresholded_image, (256, 256, 256), use_voxels=True)
    image_array = np.expand_dims(image_resampled.numpy(), axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    brain_mask_array = np.squeeze(model.predict(image_array))
    brain_mask_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(brain_mask_array))
    brain_mask_image = ants.resample_image(brain_mask_resampled, image.shape, use_voxels=True, interp_type=1)
    brain_mask_image = brain_mask_image * ants.threshold_image( brain_mask_image, 0.5, 1e9 )
    minimum_brain_volume = round(649933.7/spacing_product)
    bxt1 = ants.label_clusters(brain_mask_image, minimum_brain_volume)
    nblabels = np.unique( bxt1.numpy() )# .astype(int)
    maxol = 0.0
    bestlab = bxt0

    for j in range(1,len(nblabels)):
        temp = ants.threshold_image( bxt1, j, j )
        tempsum = ( bxt0 * temp ).sum()
        dicer = ants.label_overlap_measures( temp, bxt0 )
        if  tempsum > maxol and dicer['MeanOverlap'][1] > 0.5 :
            if verbose:
                print( str(j) + ' dice ' + str( dicer['MeanOverlap'][1] ) )
            maxol = tempsum
            bestlab = temp
    adder = trim_segmentation_by_distance( bxt0, 1, closedilmm )
    bestlab = ants.threshold_image( bestlab + adder, 1, 2 )
    return bestlab



def subdivide_labels( x, verbose = False ):
    """
    quick subdivision of the labels in a label image. can be applied recursively
    to get several levels of subdivision.

    x: input hemisphere label image from label_hemispheres

    verbose: boolean

    Example:

    > x=ants.image_read( ants.get_data("ch2") )
    > seg=ants.threshold_image(x,1,np.math.inf)
    > for n in range(5):
    >    seg=subdivide_labels(seg,verbose=True)
    >    ants.plot(x,seg,axis=2,nslices=21,ncol=7,crop=True)

    """
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
                print( subimg )
                print( nextlab )
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
    """
    quick subdivision of the hemisphere label to go from a 2-label to a 4-label.
    will subdivide along the largest axis in terms of voxels.

    x: input hemisphere label image from label_hemispheres

    verbose: boolean

    """
    notzero=ants.threshold_image( x, 1, 1e9 )
    localshape=ants.crop_image( x, ants.threshold_image( x, 1, 1 ) ).shape
    axtosplit=np.argmax(localshape)
    mid=int(np.round( localshape[axtosplit] /2 ))
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
    """
    quick cropping to a fixed size patch around a center point

    x: input image

    pt: list of physical space coordinates

    domainer:  list defining patch size

    verbose: boolean

    """
    pti = np.round( ants.transform_physical_point_to_index( x, pt ) )
    xdim = x.shape
    for k in range(len(xdim)):
        if pti[k] < 0:
            pti[k]=0
        if pti[k] > (xdim[k]-1):
            pti[k]=(xdim[k]-1)
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

def label_hemispheres( x, template=None, templateLR=None, reg_iterations=[200,50,2,0] ):
    """
    quick somewhat noisy registration solution to hemisphere labeling. typically
    we label left as 1 and right as 2.

    x: input image

    template: MNI space template, should be "croppedMni152" or "biobank"

    templateLR: a segmentation image of template hemispheres

    reg_iterations: reg_iterations for ants.registration

    """

    if template is None or templateLR is None:
        tfn = get_data('T_template0', target_extension='.nii.gz' )
        tfnw = get_data('T_template0_WMP', target_extension='.nii.gz' )
        tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )
        bfn = antspynet.get_antsxnet_data( "croppedMni152" )
        template = ants.image_read( tfn )
        template = ( template * antspynet.brain_extraction( template, 't1' ) ).iMath( "Normalize" )
        templateawmprior = ants.image_read( tfnw )
        templateLR = ants.image_read( tlrfn )

    reg = ants.registration(
        ants.rank_intensity(x),
        ants.rank_intensity(template),
        'antsRegistrationSyNQuickRepro[s]',
        aff_metric='GC',
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=reg_iterations, total_sigma=0.5,
        random_seed = 1 )
    return( ants.apply_transforms( x, templateLR, reg['fwdtransforms'],
        interpolator='nearestNeighbor') )

def deep_tissue_segmentation( x, template=None, registration_map=None,
    atropos_prior=None, sr_model=None ):
    """
    modified slightly more efficient deep atropos that also handles the
    extra CSF issue.  returns segmentation and probability images. see
    the tissues csv available from get_data.

    x: input image, raw

    template: MNI space template, should be "croppedMni152" or "biobank"

    registration_map: pre-existing output from ants.registration

    atropos_prior: prior weight for atropos post-processing; set to None if you
        do not want to use this.  will modify the CSF, GM and WM segmentation to
        better fit the image intensity at the resolution of the input image.

    sr_model: optional (FIXME)

    """

    dapper = antspynet.deep_atropos( x, do_denoising=False )

    myk='segmentation_image'
    if atropos_prior is not None:
        msk = ants.threshold_image( dapper['segmentation_image'], 2, 3 ).iMath("GetLargestComponent",50)
        msk = ants.morphology( msk, "close", 2 )
        mskfull = ants.threshold_image( dapper['segmentation_image'], 1, 6 )
        mskfull = mskfull - msk
        priors = dapper[myk][1:4]
        for k in range( len( priors ) ):
            priors[k]=ants.image_clone( priors[k]*msk )
        aap = ants.atropos( x, msk, i=priors, m='[0.0,1x1x1]',
            c = '[1,0]', priorweight=atropos_prior, verbose=1  )
        dapper['segmentation_image'] = aap['segmentation'] * msk + dapper['segmentation_image'] * mskfull
        dapper['probability_images'][1:4] = aap['probabilityimages']

    return dapper

def deep_brain_parcellation(
    target_image,
    template,
    img6seg = None,
    do_cortical_propagation=False,
    atropos_prior=None,
    verbose=True,
):
    """
    modified slightly more efficient deep dkt that also returns atropos output
    thus providing a complete hierarchical parcellation of t1w.  we run atropos
    here so we dont need to redo registration separately. see
    the lobes and dkt csv available from get_data.

    target_image: input image

    template: MNI space template, should be "croppedMni152" or "biobank"

    img6seg: optional pre-existing brain segmentation - a 6-class image ( ANTs standard )

    do_cortical_propagation: boolean, adds a bit extra time to propagate cortical
        labels explicitly into cortical segmentation

    atropos_prior: prior weight for atropos post-processing; set to None if you
        do not want to use this.  will modify the CSF, GM and WM segmentation to
        better fit the image intensity at the resolution of the input image.

    verbose: boolean


    Returns
    -------
    a dictionary containing:

    - tissue_segmentation : 6 tissue segmentation
    - tissue_probabilities : probability images associated with above
    - dkt_parcellation : tissue agnostic DKT parcellation
    - dkt_lobes : major lobes of the brain
    - dkt_cortex: cortical tissue DKT parcellation (if requested)
    - hemisphere_labels: free to get hemisphere labels
    - wmSNR : white matter signal-to-noise ratio
    - wmcsfSNR : white matter to csf signal-to-noise ratio

    """
    if verbose:
        print("Begin registration")

    rig = ants.registration( template, ants.rank_intensity(target_image),
        "antsRegistrationSyNQuickRepro[a]",
        aff_iterations = (500,200,0,0),
        total_sigma=0.5,        
        random_seed=1 )
    rigi = ants.apply_transforms( template, target_image, rig['fwdtransforms'])

    if verbose:
        print("Begin Atropos tissue segmentation")

    if img6seg is None:
        if verbose:
            print( "do deep_tissue_segmentation")
        mydap = deep_tissue_segmentation( target_image  )
    else:
        if verbose:
            print( "use existing deep_tissue_segmentation")
        mydap = { 'segmentation_image': img6seg, 'probability_images': None }

    if verbose:
        print("End Atropos tissue segmentation")

    if verbose:
        print("Begin DKT")

    dktprepro = True
    dkt = antspynet.desikan_killiany_tourville_labeling(
        target_image,
        do_preprocessing=True,
        return_probability_images=False,
        do_lobar_parcellation = True,
        do_denoising=False
    )

    myhemiL = ants.threshold_image( dkt['lobar_parcellation'], 1, 6 )
    myhemiR = ants.threshold_image( dkt['lobar_parcellation'], 7, 12 )
    myhemi = myhemiL + myhemiR * 2.0
    brainmask = ants.threshold_image( mydap['segmentation_image'], 1, 6 )
    myhemi = ants.iMath( brainmask, 'PropagateLabelsThroughMask', myhemi, 100, 0)

    cortprop = None
    if do_cortical_propagation:
        cortprop = ants.threshold_image( mydap['segmentation_image'], 2, 2 )
        cortlab = dkt['segmentation_image'] * ants.threshold_image( dkt['segmentation_image'], 1000, 5000  )
        cortprop = ants.iMath( cortprop, 'PropagateLabelsThroughMask',
            cortlab, 1, 0)

    wmseg = ants.threshold_image( mydap['segmentation_image'], 3, 3 )
    wmMean = target_image[ wmseg == 1 ].mean()
    wmStd = target_image[ wmseg == 1 ].std()
    csfseg = ants.threshold_image( mydap['segmentation_image'], 1, 1 )
    csfStd = target_image[ csfseg == 1 ].std()
    wmSNR = wmMean/wmStd
    wmcsfSNR = wmMean/csfStd

    return {
        "tissue_segmentation":mydap['segmentation_image'],
        "tissue_probabilities":mydap['probability_images'],
        "dkt_parcellation":dkt['segmentation_image'],
        "dkt_lobes":dkt['lobar_parcellation'],
        "dkt_cortex": cortprop,
        "hemisphere_labels": myhemi,
        "wmSNR": wmSNR,
        "wmcsfSNR": wmcsfSNR, }


def deep_hippo(
    img,
    template,
    number_of_tries = 10,
    tx_type="antsRegistrationSyNQuickRepro[a]",
    sr_model = None,
    verbose=False
):


    super_resolution = None

    avgleft = img * 0
    avgright = img * 0
    for k in range(number_of_tries):
        if verbose:
            print( "try " + str(k))
        rig = ants.registration( template, ants.rank_intensity(img),
            tx_type, random_seed=k,  total_sigma=0.5, verbose=verbose )
        if verbose:
            print( rig )
        rigi = ants.apply_transforms( template, img, rig['fwdtransforms'] )
        if verbose:
            print( "done with warp - do hippmapp3r" )
        hipp = antspynet.hippmapp3r_segmentation( rigi, do_preprocessing=False )
        if verbose:
            print( "done with hippmapp3r - backtransform" )

        if sr_model is not None:
            mysr = super_resolution_segmentation_per_label(
                rigi,  segmentation=hipp, upFactor=[2,2,2],
                sr_model=sr_model, segmentation_numbers=[1,2],
                dilation_amount=0,
                probability_images=None,
                probability_labels=[1,2], max_lab_plus_one=True, verbose=True
                )
            hipp = mysr['super_resolution_segmentation']

        hippr = ants.apply_transforms(
            img,
            hipp,
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='nearestNeighbor',
        )
        avgleft = avgleft + ants.threshold_image( hippr, 2, 2 ).iMath("GetLargestComponent",1) / float(number_of_tries)
        avgright = avgright + ants.threshold_image( hippr, 1, 1 ).iMath("GetLargestComponent",1) / float(number_of_tries)


    avgright = ants.iMath(avgright,"Normalize")  # output: probability image right
    avgleft = ants.iMath(avgleft,"Normalize")    # output: probability image left
    hippright_bin = ants.threshold_image( avgright, 0.5, 2.0 ).iMath("GetLargestComponent",1)
    hippleft_bin = ants.threshold_image( avgleft, 0.5, 2.0 ).iMath("GetLargestComponent",1)
    hipp_bin = hippleft_bin + hippright_bin * 2

    hippleftORlabels  = ants.label_geometry_measures(hippleft_bin, avgleft)
    hippleftORlabels['Description'] = 'left hippocampus'
    hipprightORlabels  = ants.label_geometry_measures(hippright_bin, avgright)
    hipprightORlabels['Description'] = 'right hippocampus'
    # hippleftORlabels=hippleftORlabels.append( hipprightORlabels )
    # FIXME possible fix below
    hippleftORlabels = pd.concat( [hippleftORlabels, hipprightORlabels] , axis=1 )
    hippleftORlabels['Label']=[1,2]
    labels = {
        'segmentation':hipp_bin,
        'description':hippleftORlabels,
        'HLProb':avgleft,
        'HRProb':avgright
        }
    return labels


def dap( x ):
    return( antspynet.deep_atropos( x, do_preprocessing=True, do_denoising=False )['segmentation_image'] )

def label_and_img_to_sr( img, label_img, sr_model, return_intensity=False, target_range=[1,0] ):
    """
    Apply SR to a label image and the associated intensity.

    img : the input intensity image

    label_img : the input label image (segmentation)

    sr_model : a super resolution model

    return_intensity : boolean if True will return both intensity and labels,
        otherwise only the upsampled labels are returned.

    target_range : eg [1,0] or [127.5,-127.5] are most likely

    """
    ulabs = np.unique( label_img.numpy() )
    ulabs.sort()
    ulabs = ulabs[1:len(ulabs)]
    ulabs = ulabs.tolist()
    if return_intensity:
        return super_resolution_segmentation_per_label(
                        img,
                        label_img,
                        [2,2,2],
                        sr_model,
                        segmentation_numbers=ulabs,
                        target_range=target_range,
                        max_lab_plus_one=True  )
    else:
        return super_resolution_segmentation_per_label(
                    img,
                    label_img,
                    [2,2,2],
                    sr_model,
                    segmentation_numbers=ulabs,
                    target_range=target_range,
                    max_lab_plus_one=True  )['super_resolution_segmentation']

def hierarchical_to_sr( t1hier, sr_model, tissue_sr=False, blending=0.5, verbose=False ):
    """
    Apply SR to most output from the hierarchical function

    t1hier : the hierarchical object

    sr_model : a super resolution model

    tissue_sr : boolean, if True will perform SR on the whole image which can
        be very memory intensive; only use if you have plenty of RAM.

    blending : if not None, will blend the SR

    verbose : boolean

    """
    # from  >>> t1h.keys()
    img = t1hier['brain_n4_dnz']
    myvarlist = [ 'mtl', 'cit168lab', 'snseg', 'bf', 'deep_cit168lab' ]
    for myvar in myvarlist:
        if verbose:
            print( myvar )
        t1hier[myvar]=label_and_img_to_sr( img, t1hier[myvar], sr_model )
    if verbose:
        print( 'dkt_cortex' )
    temp = label_and_img_to_sr( img, t1hier['dkt_parc']['dkt_cortex'], sr_model, return_intensity=True )
    tempupimg = temp['super_resolution']
    if blending is not None:
        tempupimg = tempupimg * (1.0 - blending ) + ants.iMath( tempupimg, "Sharpen" ) * blending
    t1hier['dkt_parc']['dkt_cortex']= temp['super_resolution_segmentation']
    if verbose:
        print( 'assemble summaries' )
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
        # fix the doubled labels
        temp = mysr['super_resolution_segmentation']
        for k in [7,8,9,10,11,12] :
            temp[ temp == k ] = temp[ temp == k ] - 6
        t1hier['brain_n4_dnz'] = mysr['super_resolution']
        t1hier['dkt_parc']['tissue_segmentation'] = temp
    else:
        t1hier['brain_n4_dnz'] = tempupimg
        t1hier['dkt_parc']['tissue_segmentation'] = ants.resample_image_to_target(
            t1hier['dkt_parc']['tissue_segmentation'], tempupimg, 'nearestNeighbor')

    if verbose:
        print("unique tissue labels")
        print( np.unique(t1hier['dkt_parc']['tissue_segmentation'].numpy()) )

    tissue = map_segmentation_to_dataframe( "tissues",
        ants.mask_image(
            t1hier['dkt_parc']['tissue_segmentation'],
            t1hier['dkt_parc']['tissue_segmentation'],  [1,2,3,4,5,6] ) )
    t1hier['dataframes']['tissues'] = tissue
    braintissuemask = ants.threshold_image( t1hier['dkt_parc']['tissue_segmentation'], 2, 6 )
#
#    mydcit = deep_cit168( t1hier['brain_n4_dnz'], binary_mask = braintissuemask )
#    t1hier['deep_cit168lab'] = mydcit['segmentation']
#    t1hier['dataframes']["deep_cit168"]=mydcit['description']

#    deep_bf = deep_nbm( t1hier['brain_n4_dnz'] * braintissuemask,
#            get_data("deep_nbm_rank",target_extension='.h5'),
#            csfquantile=None, aged_template=True )
#    t1hier['bf'] = deep_bf['segmentation']
#    t1hier['dataframes']["bf"]=deep_bf['description']
#
    return t1hier


def deep_mtl(t1, sr_model=None, verbose=True):

    """
    Hippocampal/Enthorhinal segmentation using "Deep Flash"

    Perform hippocampal/entorhinal segmentation in T1 images using
    labels from Mike Yassa's lab

    https://faculty.sites.uci.edu/myassa/

    The labeling is as follows:
    Label 0 :  background
    Label 5 :  left aLEC
    Label 6 :  right aLEC
    Label 7 :  left pMEC
    Label 8 :  right pMEC
    Label 9 :  left perirhinal
    Label 10:  right perirhinal
    Label 11:  left parahippocampal
    Label 12:  right parahippocampal
    Label 13:  left DG/CA3
    Label 14:  right DG/CA3
    Label 15:  left CA1
    Label 16:  right CA1
    Label 17:  left subiculum
    Label 18:  right subiculum

    """

    verbose = False

    labels = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    label_descriptions = ['background',
                          'left aLEC',
                          'right aLEC',
                          'left pMEC',
                          'right pMEC',
                          'left perirhinal',
                          'right perirhinal',
                          'left parahippocampal',
                          'right parahippocampal',
                          'left DG/CA3',
                          'right DG/CA3',
                          'left CA1',
                          'right CA1',
                          'left subiculum',
                          'right subiculum'
                          ]

    template = ants.image_read(antspynet.get_antsxnet_data("deepFlashTemplateT1SkullStripped"))
    registration = ants.registration(fixed=template, moving=t1,
        type_of_transform="antsRegistrationSyNQuickRepro[a]", total_sigma=0.5, verbose=verbose)
    template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                               invtransforms=registration['invtransforms'])
    t1_warped = registration['warpedmovout']

    df = antspynet.deep_flash(t1_warped, 
        do_preprocessing=False,
        use_rank_intensity = True,
        verbose=verbose)

    if False:
        if sr_model is not None:
            ulabs = np.unique( df['segmentation_image'].numpy() )
            ulabs.sort()
            ulabs = ulabs[1:len(ulabs)]
            ulabs = ulabs.tolist()
            newprobs = super_resolution_segmentation_per_label(
                t1_warped,
                df['segmentation_image'],
                [2,2,2],
                sr_model,
                ulabs,
                probability_images=df['probability_images'][1:len(df['probability_images'])],
                probability_labels=ulabs,
                max_lab_plus_one=True  )
            for k in range(1, len(df['probability_images']) ):
                df['probability_images'][k] = newprobs['probability_images'][k-1]

    if sr_model is not None:
        newprobs = super_resolution_segmentation_with_probabilities(
            t1_warped, df['probability_images'], sr_model )
        df['probability_images'] = newprobs['sr_probabilities']

    if False:
        probability_images = list()
        for i in range(len(df['probability_images'])):
            probability_image = ants.apply_transforms(fixed=t1,
                                                      moving=df['probability_images'][i],
                                                      transformlist=template_transforms['invtransforms'],
                                                      whichtoinvert=[True],
                                                      interpolator="linear",
                                                      verbose=verbose)
            probability_images.append(probability_image)
        image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
        background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                                np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
        foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
        segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
        segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]
        relabeled_image = ants.image_clone(segmentation_image)
        for i in range(len(labels)):
            relabeled_image[segmentation_image==i] = labels[i]

    relabeled_image = ants.apply_transforms( fixed=t1,
                                          moving=df['segmentation_image'],
                                          transformlist=template_transforms['invtransforms'],
                                          whichtoinvert=[True],
                                          interpolator="nearestNeighbor",
                                          verbose=verbose)
    mtl_description = map_segmentation_to_dataframe( 'mtl_description', relabeled_image )

    deep_mtl_dictionary = {
                          'mtl_description':mtl_description,
                          'mtl_segmentation':relabeled_image
#                          'mtl_probability_images':probability_images
                          }
    return(deep_mtl_dictionary)

def localsyn(
    img,
    template,
    hemiS,
    templateHemi,
    whichHemi,
    padder,
    iterations,
    output_prefix,
    total_sigma=0.5
):
    """
    Perform localized symmetric image registration between a subject hemisphere and a template.

    This function extracts the specified hemisphere from both the subject and template images,
    identifies a tight region around the non-background portion of the template hemisphere, 
    crops both images accordingly, and performs a symmetric normalization (SyN) registration 
    between the cropped template and subject hemisphere.

    Parameters
    ----------
    img : ANTsImage
        The subject's anatomical image (e.g., T1-weighted) to be registered.

    template : ANTsImage
        The template image to which the subject image will be registered.

    hemiS : ANTsImage
        A hemisphere label image corresponding to the subject's space. Each hemisphere should 
        be encoded with a distinct integer label (e.g., left = 1, right = 2).

    templateHemi : ANTsImage
        A hemisphere label image in the template space, analogous to `hemiS`.

    whichHemi : int
        The integer label for the hemisphere to process (e.g., 1 for left, 2 for right).

    padder : int
        Number of morphological dilation steps (`MD`) applied to the cropped mask to ensure
        sufficient context around the hemisphere during cropping.

    iterations : list of str
        Registration iteration schedule passed to `ants.registration()` (e.g., `["40x20x0"]`).

    output_prefix : str
        Prefix to use for naming output files generated by the registration process.

    total_sigma : float, optional
        The total Gaussian smoothing sigma used in the SyN registration (default: 0.5).

    Returns
    -------
    dict
        Dictionary containing output of `ants.registration()`, including the warped image,
        transforms, and potentially intermediate files. Keys include:
        - `'warpedmovout'`: warped subject image
        - `'fwdtransforms'`: forward transformation filenames
        - `'invtransforms'`: inverse transformation filenames
        - ... other ANTsPy registration outputs

    Notes
    -----
    - The function assumes that the subject and template hemisphere label images (`hemiS` and 
      `templateHemi`) use the same labeling scheme and orientation.
    - The `padder` controls how much extra anatomical context is included around the cropped 
      region. Larger values may improve registration robustness but increase computational cost.
    - The registration uses a reproducible variant of SyN (`antsRegistrationSyNQuickRepro[s]`)
      with a fixed random seed (1) for consistent outputs across runs.

    Example
    -------
    >>> syn_result = localsyn(
    ...     img=subject_img,
    ...     template=template_img,
    ...     hemiS=subject_hemi_label,
    ...     templateHemi=template_hemi_label,
    ...     whichHemi=1,
    ...     padder=10,
    ...     iterations=["40x20x0"],
    ...     output_prefix="subj_L_"
    ... )
    >>> warped = syn_result['warpedmovout']
    """
    ihemi = img * ants.threshold_image(hemiS, whichHemi, whichHemi)
    themi = template * ants.threshold_image(templateHemi, whichHemi, whichHemi)
    loquant = np.quantile(themi.numpy(), 0.01) + 1e-6  # background threshold + epsilon
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

def hemi_reg(
    input_image,
    input_image_tissue_segmentation,
    input_image_hemisphere_segmentation,
    input_template,
    input_template_hemisphere_labels,
    output_prefix,
    padding=10,
    labels_to_register = [2,3,4,5],
    total_sigma=0.5,
    is_test=False ):
    """
    hemisphere focused registration that will produce jacobians and figures to
    support data inspection

    input_image: input image

    input_image_tissue_segmentation: segmentation produced in ANTs style ie with
    labels defined by atropos brain segmentation (1 to 6)

    input_image_hemisphere_segmentation: left (1) and right (2) hemisphere
    segmentation

    input_template: template to which we register; prefer a population-specific
    relatively high-resolution template instead of MNI or biobank.

    input_template_hemisphere_labels: a segmentation image of template hemispheres
    with left labeled 1 and right labeled 2

    output_prefix: a path and prefix for registration related outputs

    padding: number of voxels to pad images, needed for diffzero

    labels_to_register: list of integer segmentation labels to use to define
    the tissue types / regions of the brain to register.

    total_sigma: scalar >= 0.0 ; higher means more constrained registration.

    is_test: boolean. this function can be long running by default. this would
    help testing more quickly by running fewer iterations.

    """

    img = ants.rank_intensity( input_image )
    # ionlycerebrum = ants.mask_image( input_image_tissue_segmentation,
    #    input_image_tissue_segmentation, labels_to_register, 1 )
    ionlycerebrum = brain_extraction( input_image )

    tonlycerebrum = brain_extraction( input_template )
    template = ants.rank_intensity( input_template )

    regsegits=[200,200,20]

#    # upsample the template if we are passing SR as input
#    if min(ants.get_spacing(img)) < 0.8 :
#        regsegits=[200,200,200,20]
#        template = ants.resample_image( template, (0.5,0.5,0.5), interp_type = 0 )
#        tonlycerebrum = ants.resample_image_to_target( tonlycerebrum,
#            template,
#            interp_type='nearestNeighbor',
#        )

    if is_test:
        regsegits=[8,0,0]

    input_template_hemisphere_labels = ants.resample_image_to_target(
        input_template_hemisphere_labels,
        template,
        interp_type='nearestNeighbor',
    )

    # now do a hemisphere focused registration
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

    lhjac = ants.create_jacobian_determinant_image(
        synL['warpedmovout'],
        synL['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( lhjac, output_prefix+'left_hemi_jacobian.nii.gz' )

    rhjac = ants.create_jacobian_determinant_image(
        synR['warpedmovout'],
        synR['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( rhjac, output_prefix+'right_hemi_jacobian.nii.gz' )
    return {
        "synL":synL,
        "synLpng":fignameL,
        "synR":synR,
        "synRpng":fignameR,
        "lhjac":lhjac,
        "rhjac":rhjac
        }



def region_reg(
    input_image,
    input_image_tissue_segmentation,
    input_image_region_segmentation,
    input_template,
    input_template_region_segmentation,
    output_prefix,
    padding=10,
    total_sigma=0.5,
    is_test=False ):
    """
    region focused registration that will produce jacobians and figures to
    support data inspection.  region-defining images should be binary.

    input_image: input image

    input_image_tissue_segmentation: segmentation produced in ANTs style ie with
    labels defined by atropos brain segmentation (1 to 6)

    input_image_region_segmentation: a local region to register - binary.

    input_template: template to which we register; prefer a population-specific
    relatively high-resolution template instead of MNI or biobank. brain extracted.

    input_template_region_segmentation: a segmentation image of template regions - binary.

    output_prefix: a path and prefix for registration related outputs

    padding: number of voxels to pad images, needed for diffzero

    total_sigma: scalar >= 0.0 ; higher means more constrained registration.

    is_test: boolean. this function can be long running by default. this would
    help testing more quickly by running fewer iterations.

    """

    img = ants.rank_intensity( input_image )
    ionlycerebrum = brain_extraction( input_image )
    template = ants.rank_intensity( input_template )
    regsegits=[200,200,20]

    # upsample the template if we are passing SR as input
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

    # now do a region focused registration
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

    lhjac = ants.create_jacobian_determinant_image(
        synL['warpedmovout'],
        synL['fwdtransforms'][0],
        do_log=1
        )
    ants.image_write( lhjac, output_prefix+'region_jacobian.nii.gz' )

    return {
        "synL":synL,
        "synLpng":fignameL,
        "lhjac":lhjac,
        'rankimg':img*ionlycerebrum,
        'template':template
        }


def t1_hypointensity( x, xsegmentation, xWMProbability, template, templateWMPrior, wmh_thresh=0.1 ):
    """
    provide measurements that may help decide if a given t1 image is likely
    to have hypointensity.

    x: input image; bias-corrected, brain-extracted and denoised

    xsegmentation: input image hard-segmentation results

    xWMProbability: input image WM probability

    template: template image

    templateWMPrior: template tissue prior

    wmh_thresh: float used to threshold WMH probability and produce summary data

    returns:

        - wmh_summary: summary data frame based on thresholding WMH probability at wmh_thresh
        - wmh_probability_image: probability image denoting WMH probability; higher values indicate
          that WMH is more likely
        - wmh_evidence_of_existence: an integral evidence that indicates the likelihood that the input
            image content supports the presence of white matter hypointensity.
            greater than zero is supportive of WMH.  the higher, the more so.
            less than zero is evidence against.
        - wmh_max_prob: max probability of wmh
        - features: the features driving WMH predictons

    """

    if False: # Need to retrain this model with refined inference code
        return {
            "wmh_summary":None,
            "wmh_probability_image":None,
            "wmh_evidence_of_existence":None,
            "wmh_max_prob":None,
            "features":None }

    mybig = [88,128,128]
    templatesmall = ants.resample_image( template, mybig, use_voxels=True )
    qaff = ants.registration(
        ants.rank_intensity(x),
        ants.rank_intensity(templatesmall), 'antsRegistrationSyNQuickRepro[s]',
        syn_sampling=2,
        syn_metric='CC',
        reg_iterations = [25,15,0,0],total_sigma=0.5,
        aff_metric='GC', random_seed=1 )
    afftx = qaff['fwdtransforms'][1]
    templateWMPrior2x = ants.apply_transforms( x, templateWMPrior, qaff['fwdtransforms'] )
    cerebrum = ants.threshold_image( xsegmentation, 2, 4 )
    realWM = ants.threshold_image( templateWMPrior2x , 0.1, math.inf )
    inimg = ants.rank_intensity( x )
    parcellateWMdnz = ants.kmeans_segmentation( inimg, 2, realWM, mrf=0.3 )['probabilityimages'][0]
    x2template = ants.apply_transforms( templatesmall, inimg, afftx, whichtoinvert=[True] )
    parcellateWMdnz2template = ants.apply_transforms( templatesmall,
      cerebrum * parcellateWMdnz, afftx, whichtoinvert=[True] )
    # features = rank+dnz-image, lprob, wprob, wprior at mybig resolution
    f1 = x2template.numpy()
    f2 = parcellateWMdnz2template.numpy()
    f3 = ants.apply_transforms( templatesmall, xWMProbability, afftx, whichtoinvert=[True] ).numpy()
    f4 = ants.apply_transforms( templatesmall, templateWMPrior, qaff['fwdtransforms'][0] ).numpy()
    myfeatures = np.stack( (f1,f2,f3,f4), axis=3 )
    newshape = np.concatenate( [ [1],np.asarray( myfeatures.shape )] )
    myfeatures = myfeatures.reshape( newshape )

    inshape = [None,None,None,4]
    wmhunet = antspynet.create_unet_model_3d( inshape,
        number_of_outputs = 1,
        number_of_layers = 4,
        mode = 'sigmoid' )

    wmhunet.load_weights( get_data("simwmhseg", target_extension='.h5') )

    pp = wmhunet.predict( myfeatures )

    limg = ants.from_numpy( tf.squeeze( pp[0] ).numpy( ) )
    limg = ants.copy_image_info( templatesmall, limg )
    lesresam = ants.apply_transforms( x, limg, afftx, whichtoinvert=[False] )
    # lesresam = lesresam * cerebrum
    rnmdl = antspynet.create_resnet_model_3d( inshape,
      number_of_outputs = 1,
      layers = (1,2,3),
      residual_block_schedule = (3,4,6,3), squeeze_and_excite = True,
      lowest_resolution = 32, cardinality = 1, mode = "regression" )
    rnmdl.load_weights( get_data("simwmdisc", target_extension='.h5' ) )
    qq = rnmdl.predict( myfeatures )

    lesresamb = ants.threshold_image( lesresam, wmh_thresh, 1.0 )
    lgo=ants.label_geometry_measures( lesresamb, lesresam )
    wmhsummary = pd.read_csv( get_data("wmh_evidence", target_extension='.csv' ) )
    wmhsummary.at[0,'Value']=lgo.at[0,'VolumeInMillimeters']
    wmhsummary.at[1,'Value']=lgo.at[0,'IntegratedIntensity']
    wmhsummary.at[2,'Value']=float(qq)

    return {
        "wmh_summary":wmhsummary,
        "wmh_probability_image":lesresam,
        "wmh_evidence_of_existence":float(qq),
        "wmh_max_prob":lesresam.max(),
        "features":myfeatures }



def deep_nbm( t1,
    nbm_weights,
    binary_mask=None,
    deform=False,
    aged_template=False,
    csfquantile=None,
    reflect=False,
    verbose=False ):

    """
    CH13 and Nucleus basalis of Meynert segmentation and subdivision

    Perform CH13 and NBM segmentation in T1 images using Avants labels.

    t1 : T1-weighted neuroimage antsImage - already brain extracted

    nbm_weights : string weight file for parcellating unet

    binary_mask : will restrict output to this mask

    deform : boolean to correct for image deformation

    aged_template : boolean to control which template to use

    csfquantile: if not None, will try to remove CSF from the image.
        0.25 may be a good value to try.

    verbose: boolean

    The labeling is as follows:

    Label,Description,Side
    1,CH13_left,left
    2,CH13_right,right
    3,NBM_left_ant,left
    4,NBM_left_mid,left
    5,NBM_left_pos,left
    6,NBM_right_ant,right
    7,NBM_right_mid,right
    8,NBM_right_pos,right

    Failure modes will include odd image orientation (in which case you might
    use the registration option).  A more nefarious issue can be a poor extraction
    of the cerebrum in the inferior frontal lobe.  These can be unpredictable
    but if one sees a bad extraction, please check the mask that is output by
    this function to see if it excludes non-cerebral tissue.

    """

    if not aged_template:
        refimg = ants.image_read( get_data( "CIT168_T1w_700um_pad", target_extension='.nii.gz' ))
        refimg = ants.rank_intensity( refimg )
        refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
        refimgseg = ants.image_read( get_data( "CIT168_basal_forebrain", target_extension='.nii.gz' ))
        refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )
    else:
        refimg = ants.image_read( get_data( "CIT168_T1w_700um_pad_adni", target_extension='.nii.gz' ))
        refimg = ants.rank_intensity( refimg )
        refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
        refimgseg = ants.image_read( get_data( "CIT168_basal_forebrain_adni", target_extension='.nii.gz' ))
        refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )

    pt_labels = [1,2,3,4]
    group_labels = [0,1,2,3,4,5,6,7,8]
    reflection_labels = [0,2,1,6,7,8,3,4,5]
    crop_size = [144,96,64]

    def nbmpreprocess( img, pt_labels, group_labels, masker=None, csfquantile=None, returndef=False, reflect=False ):

        if masker is None:
            imgr = ants.rank_intensity( img )
        else:
            imgr = ants.rank_intensity( img, mask = masker )

        if csfquantile is not None and masker is None:
            masker = ants.threshold_image( imgr, np.quantile(imgr[imgr>1e-4], csfquantile ), 1e9 )

        if masker is not None:
            imgr = imgr * masker

        imgrsmall = ants.resample_image( imgr, [1,1,1] )
        if not reflect:
            reg = ants.registration(
                refimgsmall,
                imgrsmall,
                'antsRegistrationSyNQuickRepro[s]',
                reg_iterations = [200,200,20],total_sigma=0.5,
                verbose=False )
        else:
            myref = ants.reflect_image( imgrsmall, axis=0, tx='Translation' )
            reg = ants.registration(
                refimgsmall,
                imgrsmall,
                'antsRegistrationSyNQuickRepro[s]',
                reg_iterations = [200,200,20],
                total_sigma=0.5,
                initial_transform = myref['fwdtransforms'][0],
                verbose=False )
        if not returndef:
            imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'][1], interpolator='linear' )
            imgseg = ants.apply_transforms( refimg, refimgseg, reg['invtransforms'][1], interpolator='nearestNeighbor' )
        else:
            imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'], interpolator='linear' )
            imgseg = ants.image_clone( refimgseg )
        binseg = ants.mask_image( imgseg, imgseg, pt_labels, binarize=True )
        imgseg = ants.mask_image( imgseg, imgseg, group_labels, binarize=False  )
        com = ants.get_center_of_mass( binseg )
        return {
            "img": imgraff,
            "seg": imgseg,
            "imgc": special_crop( imgraff, com, crop_size ),
            "segc": special_crop( imgseg, com, crop_size ),
            "reg" : reg,
            "mask": masker
            }


    nLabels = len( group_labels )
    number_of_outputs = len(group_labels)
    number_of_channels = 1
    ################################################
    unet0 = antspynet.create_unet_model_3d(
         [ None, None, None, number_of_channels ],
         number_of_outputs = 1, # number of landmarks must be known
         number_of_layers = 4, # should optimize this wrt criterion
         number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
         convolution_kernel_size = 3, # maybe should optimize this wrt criterion
         deconvolution_kernel_size = 2,
         pool_size = 2,
         strides = 2,
         dropout_rate = 0.0,
         weight_decay = 0,
         additional_options = "nnUnetActivationStyle",
         mode =  "sigmoid" )

    unet1 = antspynet.create_unet_model_3d(
        [None,None,None,2],
        number_of_outputs=number_of_outputs,
        mode="classification",
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0,
        weight_decay=0,
        additional_options = "nnUnetActivationStyle")

    # concat output to input and pass to 2nd net
    # nextin = tf.concat(  [ unet0.inputs[0], unet0.outputs[0] ], axis=4 )
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    class myConcat(Layer):
        def call(self, x):
            return tf.concat(x, axis=4 )
    nextin = myConcat()( [ unet0.inputs[0], unet0.outputs[0] ] )

    unetonnet = unet1( nextin )
    unet_model = tf.keras.models.Model(
            unet0.inputs,
            [ unetonnet,  unet0.outputs[0] ] )

    unet_model.load_weights( nbm_weights )

    imgprepro = nbmpreprocess( t1, pt_labels, group_labels,
        csfquantile = csfquantile,
        returndef = deform,
        reflect = reflect )

    def map_back( relo, t1, imgprepro, interpolator='linear', deform = False ):
        if not deform:
            relo = ants.apply_transforms( t1, relo,
                imgprepro['reg']['invtransforms'][0], whichtoinvert=[True],
                interpolator=interpolator )
        else:
            relo = ants.apply_transforms( t1, relo,
                imgprepro['reg']['invtransforms'], interpolator=interpolator )
        return relo


    ####################################################
    physspaceBF = imgprepro['imgc']
    if verbose:
        print( "Mean preprocessed cropped image in nbm seg " + str(physspaceBF.mean() ) )

    tfarr1 = tf.cast( physspaceBF.numpy() ,'float32' )
    newshapeBF = list( tfarr1.shape )
    newshapeBF.insert(0,1)
    newshapeBF.insert(4,1)
    tfarr1 = tf.reshape(tfarr1, newshapeBF )
    snpred = unet_model.predict( tfarr1 )
    segpred = snpred[0]
    sigmoidpred = snpred[1]
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
        for i in range(1,len(group_labels)):
            relabeled_image[segmentation_image==(i)] = group_labels[i]
    else:
        for i in range(1,len(group_labels)):
            relabeled_image[segmentation_image==(i)] = reflection_labels[i]

    bfsegdesc = map_segmentation_to_dataframe( 'nbm3CH13', relabeled_image )

    return { 'segmentation':relabeled_image,
        'description':bfsegdesc,
        'mask': imgprepro['mask'],
        'probability_images': probability_images }


def deep_nbm_old( t1, ch13_weights, nbm_weights, registration=True,
    csfquantile = 0.15, binary_mask=None, verbose=False ):

    """
    Nucleus basalis of Meynert segmentation and subdivision

    Perform NBM segmentation in T1 images using Avants labels.

    t1 : T1-weighted neuroimage antsImage - already brain extracted

    ch13_weights : string weight file for ch13

    nbm_weights : string weight file for nbm

    registration : boolean to correct for image orientation and resolution by registration

    csfquantile : float value below 0.5 that tries to trim residual CSF off brain.

    binary_mask : will restrict output to this mask

    The labeling is as follows:

    Label,Description,Side
    1,CH13_left,left
    2,CH13_right,right
    3,NBM_left_ant,left
    4,NBM_left_mid,left
    5,NBM_left_pos,left
    6,NBM_right_ant,right
    7,NBM_right_mid,right
    8,NBM_right_pos,right

    Failure modes will include odd image orientation (in which case you might
    use the registration option).  A more nefarious issue can be a poor extraction
    of the cerebrum in the inferior frontal lobe.  These can be unpredictable
    but if one sees a bad extraction, please check the mask that is output by
    this function to see if it excludes non-cerebral tissue.

    """

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    label_descriptions = ['background',
                          'CH13_left',
                          'CH13_right',
                          'NBM_left_ant',
                          'NBM_left_mid',
                          'NBM_left_pos',
                          'NBM_right_ant',
                          'NBM_right_mid',
                          'NBM_right_pos',
                          ]

    t1use = ants.iMath( t1, "Normalize" )
    if registration:
        nbmtemplate = ants.image_read(get_data("nbm_template", target_extension=".nii.gz"))
        orireg = ants.registration( fixed = nbmtemplate,
            moving = t1use,
            type_of_transform="antsRegistrationSyNQuickRepro[a]", total_sigma=0.5,verbose=False )
        t1use = orireg['warpedmovout']

    template = ants.image_read(get_data("CIT168_T1w_700um_pad_adni", target_extension=".nii.gz"))
    templateSmall = ants.resample_image( template, [2.0,2.0,2.0] )
    registrationsyn = ants.registration(
        fixed=templateSmall,
        moving=ants.iMath(t1use,"Normalize"),
        type_of_transform="antsRegistrationSyNQuickRepro[s]", total_sigma=0.5, verbose=False )

    if verbose:
        print( registrationsyn['fwdtransforms'] )

    image = ants.iMath( t1use, "TruncateIntensity", 0.0001, 0.999 ).iMath("Normalize")
    bfPriorL1 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_1_left", target_extension=".nii.gz"))
    bfPriorR1 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_1_right", target_extension=".nii.gz"))
    bfPriorL2 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_2_left", target_extension=".nii.gz"))
    bfPriorR2 = ants.image_read(get_data("CIT168_basal_forebrain_adni_prob_2_right", target_extension=".nii.gz"))

    patchSize = [ 64, 64, 32 ]
    priorL1tosub = ants.apply_transforms( image, bfPriorL1, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")
    priorR1tosub = ants.apply_transforms( image, bfPriorR1, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")
    priorL2tosub = ants.apply_transforms( image, bfPriorL2, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")
    priorR2tosub = ants.apply_transforms( image, bfPriorR2, registrationsyn['invtransforms'] ).smooth_image( 3 ).iMath("Normalize")

    if binary_mask is None:
        masker = ants.threshold_image(image, np.quantile(image[image>1e-4], csfquantile ), 1e9 )
    else:
        masker = ants.apply_transforms( image, binary_mask,
            orireg['fwdtransforms'], interpolator='nearestNeighbor' )

    ch13point = ants.get_center_of_mass( priorL1tosub + priorR1tosub )

    nchanCH13 = 1
    nclasstosegCH13 = 3 # for ch13
    nchanNBM = 2
    nclasstosegNBM = 4 # for nbm
    actor = 'classification'
    nfilt = 32
    addoptsNBM = "nnUnetActivationStyle"
    unetCH13 = antspynet.create_unet_model_3d(
         [ None, None, None, nchanCH13 ],
         number_of_outputs = nclasstosegCH13, # number of landmarks must be known
         number_of_layers = 4, # should optimize this wrt criterion
         number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
         convolution_kernel_size = 3, # maybe should optimize this wrt criterion
         deconvolution_kernel_size = 2,
         pool_size = 2,
         strides = 2,
         dropout_rate = 0,
         weight_decay = 0,
         mode = 'classification' )
    unetCH13.load_weights( ch13_weights )

    physspace = special_crop( image, ch13point, patchSize)
    ch13array = physspace.numpy()
    newshape = list( ch13array.shape )
    newshape.insert(0,1)
    newshape.append(1)
    ch13pred = unetCH13.predict( tf.reshape( ch13array, newshape ) )
    probability_images = []
    for jj in range(3):
        temp = ants.from_numpy( ch13pred[0,:,:,:,jj] )
        probability_images.append( ants.copy_image_info( physspace, temp ) )
    bint = physspace * 0 + 1
    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], bint )
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], bint),
        np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
    relabeled_image = ants.image_clone(segmentation_image)
    ch13totalback = ants.resample_image_to_target(relabeled_image, image, interp_type='nearestNeighbor') * masker
    if registration:
        ch13totalback = ants.apply_transforms( t1, ch13totalback,
            orireg['invtransforms'][0], whichtoinvert=[True], interpolator='nearestNeighbor' )

    if verbose:
        print("CH13 done")

    maskind = 3
    nlayers =  4 # for unet
    unet1 = antspynet.create_unet_model_3d(
         [ None, None, None, 2 ],
          number_of_outputs = 1, # number of landmarks must be known
           number_of_layers = 4, # should optimize this wrt criterion
           number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
           convolution_kernel_size = 3, # maybe should optimize this wrt criterion
           deconvolution_kernel_size = 2,
           pool_size = 2,
           strides = 2,
           dropout_rate = 0.0,
           weight_decay = 0,
           additional_options = "nnUnetActivationStyle",
           mode = "sigmoid" )
    maskinput = tf.keras.layers.Input( [ None, None,  None, 1 ] )
    posteriorMask1 = tf.keras.layers.multiply(
      [ unet1.outputs[0] , maskinput ], name='maskTimesPosteriors1'  )
    unet = tf.keras.models.Model( [ unet1.inputs[0], maskinput ], posteriorMask1 )

    unet2 = antspynet.create_unet_model_3d(
         [ None, None, None, 2 ],
          number_of_outputs = nclasstosegNBM, # number of landmarks must be known
           number_of_layers = 4, # should optimize this wrt criterion
           number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
           convolution_kernel_size = 3, # maybe should optimize this wrt criterion
           deconvolution_kernel_size = 2,
           pool_size = 2,
           strides = 2,
           dropout_rate = 0.0,
           weight_decay = 0,
           additional_options = "nnUnetActivationStyle",
           mode =  "classification" )

    temp = tf.split( unet1.inputs[0], 2, axis=4 )
    temp[1] = unet.outputs[0]
    newmult = tf.concat( temp, axis=4 )
    unetonnet = unet2( newmult )
    unetNBM = tf.keras.models.Model(
        unet.inputs,
        [ unetonnet,  unet.outputs[0] ] )
    unetNBM.load_weights( nbm_weights )

    # do each side separately
    bfseg = t1 * 0.0
    for nbmnum in [0,1]:
        if nbmnum == 0:
            nbmpoint = ants.get_center_of_mass( priorL2tosub )
            nbmprior = special_crop( priorL2tosub, nbmpoint, patchSize).numpy() # prior
            labels=[3,4,5]
        if nbmnum == 1:
            nbmpoint = ants.get_center_of_mass( priorR2tosub )
            nbmprior = special_crop( priorR2tosub, nbmpoint, patchSize).numpy() # prior
            labels=[6,7,8]
        physspaceNBM = special_crop( image, nbmpoint, patchSize) # image
        nbmmask = special_crop( masker, nbmpoint, patchSize).numpy() # mask
        tfarr1 = tf.stack( [physspaceNBM.numpy(),nbmprior], axis=3  )
        newshapeNBM = list( tfarr1.shape )
        newshapeNBM.insert(0,1)
        tfarr1 = tf.reshape(tfarr1, newshapeNBM )
        tfarr2 = tf.reshape( nbmmask, newshape )
        nbmpred = unetNBM.predict( ( tfarr1, tfarr2  ) )
        segpred = nbmpred[0]
        sigmoidpred = nbmpred[1]
        nbmpred1_image = ants.from_numpy( sigmoidpred[0,:,:,:,0] )
        nbmpred1_image = ants.copy_image_info( physspaceNBM, nbmpred1_image )
        bint = ants.threshold_image( nbmpred1_image, 0.5, 1.0 ).iMath("GetLargestComponent",1)
        probability_images = []
        for jj in range(3):
            temp = ants.from_numpy( segpred[0,:,:,:,jj+1] )
            probability_images.append( ants.copy_image_info( physspaceNBM, temp ) )
        image_matrix = ants.image_list_to_matrix(probability_images, bint)
        segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1)
        segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
        relabeled_image = ants.image_clone(segmentation_image)
        for i in range(len(labels)):
            relabeled_image[segmentation_image==(i+1)] = labels[i]
        relabeled_image = ants.resample_image_to_target(relabeled_image, image, interp_type='nearestNeighbor')
        if registration:
            relabeled_image = ants.apply_transforms( t1, relabeled_image,
                    orireg['invtransforms'][0], whichtoinvert=[True],
                    interpolator='nearestNeighbor' )
        if verbose:
            print("NBM" + str( nbmnum ) )
        bfseg = bfseg + relabeled_image
    bfseg = ch13totalback + bfseg * ants.threshold_image( ch13totalback, 0, 0 )
    bfsegdesc = map_segmentation_to_dataframe( 'nbm3CH13', bfseg )

    if registration:
        masker = ants.apply_transforms( t1, masker,
            orireg['invtransforms'][0], whichtoinvert=[True],
            interpolator='nearestNeighbor' )

    return { 'segmentation':bfseg, 'description':bfsegdesc, 'mask': masker }





def deep_cit168( t1, binary_mask = None,
    syn_type='antsRegistrationSyNQuickRepro[s]',
    priors = None, verbose = False):

    """
    CIT168 atlas segmentation with a parcellation unet.

    Perform CIT168 segmentation in T1 images using Pauli atlas (CIT168) labels.

    t1 : T1-weighted neuroimage antsImage - already brain extracted.  image should
    be normalized 0 to 1 and with a nicely uniform histogram (no major outliers).
    we do a little work internally to help this but no guarantees it will handle
    all possible confounding issues.

    binary_mask : will restrict output to this mask

    syn_type : the type of registration used for generating priors; usually
       either SyN or antsRegistrationSyNQuickRepro[s] for repeatable results

    priors : the user can provide their own priors through this argument; for
       example, the user may run this function twice, with the output of the first
       giving input to the second run.

    verbose: boolean

    Failure modes will primarily occur around red nucleus and caudate nucleus.
    For the latter, one might consider masking by the ventricular CSF, in particular
    near the anterior and inferior portion of the caudate in subjects with
    large ventricles.  Low quality images with high atropy are also likely outside
    of the current range of the trained models. Iterating the model may help.

    """
    def tfsubset( x, indices ):
        with tf.device('/CPU:0'):
            outlist = []
            for k in indices:
                outlist.append( x[:,:,:,int(k)] )
            return tf.stack( outlist, axis=3 )

    def tfsubsetbatch( x, indices ):
        with tf.device('/CPU:0'):
            outlist2 = []
            for j in range( len( x ) ):
                outlist = []
                for k in indices:
                    if len( x[j].shape ) == 5:
                        outlist.append( x[j][k,:,:,:,:] )
                    if len( x[j].shape ) == 4:
                        outlist.append( x[j][k,:,:,:] )
                outlist2.append( tf.stack( outlist, axis=0 ) )
        return outlist2


    registration = True
    cit168seg = t1 * 0
    myprior = ants.image_read(get_data("det_atlas_25_pad_LR_adni", target_extension=".nii.gz"))
    nbmtemplate = ants.image_read( get_data( "CIT168_T1w_700um_pad_adni", target_extension=".nii.gz" ) )
    nbmtemplate = ants.resample_image( nbmtemplate, [0.5,0.5,0.5] )
    templateSmall = ants.resample_image( nbmtemplate, [2.0,2.0,2.0] )
    orireg = ants.registration(
                    fixed = templateSmall,
                    moving = ants.iMath( t1, "Normalize" ),
                    type_of_transform=syn_type, total_sigma=0.5, verbose=False )
    image = ants.apply_transforms( nbmtemplate, ants.iMath( t1, "Normalize" ),
        orireg['fwdtransforms'][1] )
    image = ants.iMath( image, "TruncateIntensity",0.001,0.999).iMath("Normalize")
    patchSize = [ 160,160,112 ]
    if priors is None:
        priortosub = ants.apply_transforms( image, myprior,
            orireg['invtransforms'][1], interpolator='nearestNeighbor' )
    else:
        if verbose:
            print("using priors")
        priortosub = ants.apply_transforms( image, priors,
            orireg['fwdtransforms'][1], interpolator='nearestNeighbor' )
    bmask = ants.threshold_image( priortosub, 1, 999 )
    # this identifies the cropping location - assumes a good registration
    pt = list( ants.get_center_of_mass( bmask ) )
    pt[1] = pt[1] + 10.0  # same as we did in training

    physspaceCIT = special_crop( image, pt, patchSize) # image

    if binary_mask is not None:
        binary_mask_use = ants.apply_transforms( nbmtemplate, binary_mask,
            orireg['fwdtransforms'][1] )
        binary_mask_use = special_crop( binary_mask_use, pt, patchSize)

    for sn in [True,False]:
        if sn:
            group_labels = [0,7,8,9,23,24,25,33,34]
            newfn=get_data( "deepCIT168_sn", target_extension=".h5" )
        else:
            group_labels = [0,1,2,5,6,17,18,21,22]
            newfn=get_data( "deepCIT168", target_extension=".h5" )

        number_of_outputs = len(group_labels)
        number_of_channels = len(group_labels)

        unet0 = antspynet.create_unet_model_3d(
                 [ None, None, None, number_of_channels ],
                 number_of_outputs = 1, # number of landmarks must be known
                 number_of_layers = 4, # should optimize this wrt criterion
                 number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
                 convolution_kernel_size = 3, # maybe should optimize this wrt criterion
                 deconvolution_kernel_size = 2,
                 pool_size = 2,
                 strides = 2,
                 dropout_rate = 0.0,
                 weight_decay = 0,
                 additional_options = "nnUnetActivationStyle",
                 mode =  "sigmoid" )

        unet1 = antspynet.create_unet_model_3d(
            [None,None,None,2],
            number_of_outputs=number_of_outputs,
            mode="classification",
            number_of_filters=(32, 64, 96, 128, 256),
            convolution_kernel_size=(3, 3, 3),
            deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0,
            weight_decay=0,
            additional_options = "nnUnetActivationStyle")

#        temp = tf.split( unet0.inputs[0], 9, axis=4 )
        import tensorflow as tf
        from tensorflow.keras.layers import Layer
        class mySplit(Layer):
            def call(self, x):
                return tf.split(x, 9, axis=4 )
        temp = mySplit()( unet0.inputs[0] )

        temp[1] = unet0.outputs[0]

        class myConcat(Layer):
            def call(self, x):
                return tf.concat(x, axis=4 )
#        newmult = tf.concat( temp[0:2], axis=4 )
        newmult = myConcat()( temp[0:2] )
        unetonnet = unet1( newmult )
        unet_model = tf.keras.models.Model(
                unet0.inputs,
                [ unetonnet,  unet0.outputs[0] ] )
        unet_model.load_weights( newfn )
        ###################
        nbmprior = special_crop( priortosub, pt, patchSize).numpy() # prior
        imgnp = tf.reshape( physspaceCIT.numpy(), [160, 160, 112,1])
        nbmprior = tf.one_hot( nbmprior, 35 )
        nbmprior = tfsubset( nbmprior, group_labels[1:len(group_labels)] )
        imgnp = tf.reshape( tf.concat( [imgnp,nbmprior], axis=3), [1,160, 160, 112,9])

        nbmpred = unet_model.predict( imgnp )
        segpred = nbmpred[0]
        sigmoidpred = nbmpred[1]

        def map_back_cit( relo, t1, orireg, interpolator='linear' ):
            relo = ants.apply_transforms( t1, relo,
                    orireg['invtransforms'][0], whichtoinvert=[True],
                    interpolator=interpolator )
            return relo

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


def preprocess_intensity( x, brain_extraction,
    intensity_truncation_quantiles=[1e-4, 0.999],
    rescale_intensities=True  ):
    """
    Default intensity processing for a brain-extracted T1-weighted image.

    Arguments
    ---------

    x : T1-weighted neuroimage antsImage after brain extraction applied

    brain_extraction : T1-weighted neuroimage brain extraction / segmentation

    intensity_truncation_quantiles: parameters passed to TruncateIntensity; the
    first value truncates values below this quantile; the second truncates
    values above this quantile.

    rescale_intensities: boolean passed to n4

    Returns
    -------
    processed image
    """
    brain_extraction = ants.resample_image_to_target( brain_extraction, x, interp_type='nearestNeighbor' )
    img = x * brain_extraction
    # print( "PP input : " + str( img.mean() ) )
    img = ants.iMath( img, "TruncateIntensity", intensity_truncation_quantiles[0], intensity_truncation_quantiles[1] ).iMath( "Normalize" )
    # print( "PP Trunc : " + str( img.mean() ) )
    # img = ants.denoise_image( img, brain_extraction, noise_model='Gaussian')
    # print( "PP DNZ : " + str( img.mean() ) )
    img = ants.n4_bias_field_correction( img, mask=brain_extraction, rescale_intensities=rescale_intensities, ).iMath("Normalize")
    # print( "PP N4 : " + str( img.mean() ) )
    return img


def hierarchical( x, output_prefix, labels_to_register=[2,3,4,5],
    imgbxt=None, img6seg=None, cit168 = False, is_test=False,
    atropos_prior=None,
    sr_model = None,
    verbose=True ):
    """
    Default processing for a T1-weighted image.  See README.

    Arguments
    ---------
    x : T1-weighted neuroimage antsImage

    output_prefix: string directory and prefix

    labels_to_register: list of integer segmentation labels (of 1 to 6 as defined
    by atropos: csf, gm, wm, dgm, brainstem, cerebellum) to define
    the tissue types / regions of the brain to register.  set to None to
    skip registration which will be faster but omit some results.

    imgbxt : pre-existing brain extraction - a binary image - will disable some processing

    img6seg: pre-existing brain segmentation - a 6-class image ( ANTs standard )

    cit168 : boolean returns labels from CIT168 atlas with high-resolution registration
        otherwise, low-resolution regitration is used.

    is_test: boolean ( parameters to run more quickly but with low quality )

    atropos_prior: prior weight for atropos post-processing; set to None if you
        do not want to use this.  will modify the CSF, GM and WM segmentation to
        better fit the image intensity at the resolution of the input image.

    sr_model: optional will trigger sr-based upsampling in some functions.

    verbose: boolean

    Returns
    -------
    dataframes and associated derived data

        - brain_n4_dnz : extracted brain denoised and bias corrected
        - brain_extraction : brain mask
        - rbp:  random basis projection results
        - left_right : left righ hemisphere segmentation
        - dkt_parc : dictionary object containing segmentation labels
        - registration : dictionary object containing registration results
        - hippLR : dictionary object containing hippocampus results
        - medial_temporal_lobe : dictionary object containing deep_flash (medial temporal lobe parcellation) results
        - white_matter_hypointensity : dictionary object containing WMH results
        - wm_tractsL  : white matter tracts, left
        - wm_tractsR  : white matter tracts, right
        - dataframes : summary data frames

    """
    if x.dimension != 3:
        raise ValueError('hierarchical: input image should be 3-dimensional')

    if verbose:
        print("Read")
    tfn = get_data('T_template0', target_extension='.nii.gz' )
    tfnw = get_data('T_template0_WMP', target_extension='.nii.gz' )
    tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )
    bfn = antspynet.get_antsxnet_data( "croppedMni152" )

    ##### read images and do simple bxt ops
    templatea = ants.image_read( tfn )
    if verbose:
        print("bxt")
    templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
    templateawmprior = ants.image_read( tfnw )
    templatealr = ants.image_read( tlrfn )
    templateb = ants.image_read( bfn )
    templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
    if imgbxt is None:
        probablySR = False
        # brain_extraction( ants.iMath( x, "Normalize" ) )
        imgbxt =  antspynet.brain_extraction( ants.iMath( x, "Normalize" ), modality="t1threetissue")['segmentation_image'].threshold_image(1,1)
        img = preprocess_intensity( ants.iMath( x, "Normalize" ), imgbxt )
    else:
        probablySR = True
        img = ants.iMath( x, "Normalize" )

    if verbose:
        print("QC -- mean of img : "+str(img.mean()))
        print(".  -- mean of bxt : "+str(imgbxt.mean()))

    # this is an unbiased method for identifying predictors that can be used to
    # rank / sort data into clusters, some of which may be associated
    # with outlierness or low-quality data
    templatesmall = ants.resample_image( templateb, (91,109,91), use_voxels=True )
    myqc = inspect_raw_t1( img, output_prefix=output_prefix, option='brain' )

    if verbose:
        print("qc results")
        print( myqc )

    ##### intensity modifications
    img = ants.iMath( img, "Normalize" )

    if verbose:
        print("parcellation -- mean of img " + str(img.mean()))

    ##### hierarchical labeling
    myparc = deep_brain_parcellation( x, templateb,
        img6seg = img6seg,
        atropos_prior = atropos_prior,
        do_cortical_propagation = not is_test, verbose=verbose )


    cit168lab = None
    cit168reg = None
    cit168lab_desc = None
    cit168adni = get_data( "CIT168_T1w_700um_pad_adni",target_extension='.nii.gz')
    cit168adni = ants.image_read( cit168adni ).iMath("Normalize")
    cit168labT = get_data( "det_atlas_25_pad_LR_adni", target_extension='.nii.gz' )
    cit168labT = ants.image_read( cit168labT )
    cit168labStem = get_data( "CIT168_T1w_700um_pad_adni_brainstem", target_extension='.nii.gz' )
    cit168labStem = ants.image_read( cit168labStem )

    if verbose:
        print("cit168-- mean of input image : " + str(img.mean()))
        print("cit168-- mean of input imgbxt : " + str(imgbxt.mean()))
        print(".     -- mean of input segmentation : " + str(myparc['tissue_segmentation'].mean()))

    cit168reg = region_reg(
            input_image = img,
            input_image_tissue_segmentation=myparc['tissue_segmentation'],
            input_image_region_segmentation=imgbxt,
            input_template=cit168adni,
            input_template_region_segmentation=ants.threshold_image( cit168adni, 0.15, 1 ),
            output_prefix=output_prefix + "_CIT168RRSYN",
            padding=10,
            total_sigma=0.5,
            is_test=not cit168 )['synL']
    cit168lab = ants.apply_transforms( img, cit168labT,
                    cit168reg['invtransforms'], interpolator = 'nearestNeighbor' )
    cit168lab_desc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', cit168lab ).dropna(axis=0)

    # optional - quick look at result
    # bxt_png = output_prefix + "_brain_extraction_dnz_n4_view.png"
    # ants.plot(img * 255.0,axis=2,ncol=8,nslices=24, crop=True, black_bg=False,
    #    filename = bxt_png )

    if verbose:
        print("hemi -- mean of input image : " + str(img.mean()))

    # assuming data is reasonable quality, we should proceed with the rest ...
    mylr = label_hemispheres( img, templatea, templatealr )

    ##### accumulate data into data frames
    hemi = map_segmentation_to_dataframe( "hemisphere", myparc['hemisphere_labels'] )
    tissue = map_segmentation_to_dataframe( "tissues", myparc['tissue_segmentation'] )
    dktl = map_segmentation_to_dataframe( "lobes", myparc['dkt_lobes'] )
    dktp = map_segmentation_to_dataframe( "dkt", myparc['dkt_parcellation'] )
    dktc = None
    if not is_test:
        dktc = map_segmentation_to_dataframe( "dkt", myparc['dkt_cortex'] )

    tissue_seg_png = output_prefix + "_seg.png"
    ants.plot( img*255, myparc['tissue_segmentation'], axis=2, nslices=21, ncol=7,
        alpha=0.6, filename=tissue_seg_png,
        crop=True, black_bg=False )

    myhypo = None
#    if verbose:
#        print("WMH")
    ##### below here are more exploratory nice to have outputs
#    myhypo = t1_hypointensity(
#        img,
#        myparc['tissue_segmentation'], # segmentation
#        myparc['tissue_probabilities'][3], # wm posteriors
#        templatea,
#        templateawmprior )

    if verbose:
        print("icv")

    myicv = icv( x )

    braintissuemask =  ants.threshold_image( myparc['tissue_segmentation'], 2, 6 )
    if verbose:
        print( myicv )
        print("NBM : mean of input image " + str( (img * braintissuemask).mean() ) )
    ##### deep_nbm basal forebrain parcellation
    deep_bf = deep_nbm( img * braintissuemask,
        get_data("deep_nbm_rank",target_extension='.h5'),
        csfquantile=None, aged_template=True, verbose=verbose )

    if is_test:
        mydataframes = {
                "icv": myicv,
                "rbp": myqc['brain'],
                "tissues":tissue,
                "dktlobes":dktl,
                "dktregions":dktp,
                "dktcortex":dktc,
                "bf":deep_bf['description'],
                }

        outputs = {
                "brain_n4_dnz": img,
                "brain_extraction": imgbxt,
                "left_right": mylr,
                "dkt_parc": myparc,
                "bf":deep_bf['segmentation'],
                "dataframes": mydataframes
            }

        return outputs


    ##### traditional deformable registration as a high-resolution complement to above
    wm_tractsL = None
    wm_tractsR = None
    wmtdfL = None
    wmtdfR = None
    reg = None
    if labels_to_register is not None and not is_test:
        reg = hemi_reg(
            input_image = img,
            input_image_tissue_segmentation = myparc['tissue_segmentation'],
            input_image_hemisphere_segmentation = mylr,
            input_template=templatea,
            input_template_hemisphere_labels=templatealr,
            output_prefix = output_prefix + "_SYN",
            labels_to_register = labels_to_register,
            is_test=is_test )
        if verbose:
            print("wm tracts")
        ##### how to use the hemi-reg output to generate any roi value from a template roi
        wm_tracts = ants.image_read( get_data( "wm_major_tracts", target_extension='.nii.gz' ) )
        wm_tractsL = ants.apply_transforms( img, wm_tracts, reg['synL']['invtransforms'],
          interpolator='nearestNeighbor' ) * ants.threshold_image( mylr, 1, 1  )
        wm_tractsR = ants.apply_transforms( img, wm_tracts, reg['synR']['invtransforms'],
          interpolator='nearestNeighbor' ) * ants.threshold_image( mylr, 2, 2  )
        wmtdfL = map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsL )
        wmtdfR = map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsR )

    if verbose:
        print("hippocampus")

    ##### specialized labeling for hippocampus
    do_hippmapper = False
    if do_hippmapper:
        ntries = 10
        if is_test:
            ntries = 1
        hippLR = deep_hippo( img=img, template=templateb, number_of_tries=ntries,
            tx_type='Affine', sr_model = sr_model )

    if verbose:
        print("medial temporal lobe")

    ##### deep_flash medial temporal lobe parcellation
    deep_flash = deep_mtl(img, sr_model = sr_model )

    if verbose:
        print("deep CIT168")
    ##### deep CIT168 segmentation - relatively fast
    deep_cit = deep_cit168( img,  priors = cit168lab,
        binary_mask = braintissuemask )

    if verbose:
        print( "SN-specific segmentation" )
#  input_image_region_segmentation, input_template, input_template_region_segmentation, output_prefix, padding=10, labels_to_register=[2, 3, 4, 5], total_sigma=0.5, is_test=False)

    if not is_test:
        tbinseg = ants.mask_image( cit168labT, cit168labT, [7,9,23,25,33,34], binarize=True)
        tbinseg = ants.morphology( tbinseg, "dilate", 14 )
        ibinseg = ants.apply_transforms( img, tbinseg, cit168reg['invtransforms'],
            interpolator='nearestNeighbor')
        snreg = region_reg( img, myparc['tissue_segmentation'], ibinseg,
            cit168adni, tbinseg, output_prefix=output_prefix + "_SNREG",
            padding = 4, is_test=False )['synL']
        tbinseg = ants.mask_image( cit168labT, cit168labT, [7,9,23,25,33,34], binarize=False)
        snseg = ants.apply_transforms( img, tbinseg,
            snreg['invtransforms'], interpolator = 'nearestNeighbor' )
        snseg = snseg * ants.threshold_image( myparc['tissue_segmentation'], 2, 6 )
        snseg_desc = map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', snseg ).dropna(axis=0)
    else :
        snseg = None
        snseg_desc = None

    if verbose:
        print( "brainstem and cerebellar segmentation" )
    # midbrain/brainstem
    brainstemseg = ants.apply_transforms( img, cit168labStem,
                cit168reg['invtransforms'], interpolator = 'nearestNeighbor' )
    brainstemseg = brainstemseg * braintissuemask
    brainstem_desc = map_segmentation_to_dataframe( 'CIT168_T1w_700um_pad_adni_brainstem', brainstemseg )
    brainstem_desc = brainstem_desc.loc[:, ~brainstem_desc.columns.str.contains('^Side')]


    # cerebellum
    cereb = antspynet.cerebellum_morphology( ants.iMath( x, "Normalize" ), compute_thickness_image=False, verbose=False, do_preprocessing=True )
    # refinement with cerebellum estimate (comment out since it's not needed for this image).
    maskc = ants.threshold_image(cereb['cerebellum_probability_image'], 0.5, 1, 1, 0)
    cereb = antspynet.cerebellum_morphology( ants.iMath( x, "Normalize" ), cerebellum_mask=maskc, compute_thickness_image=False, verbose=False, do_preprocessing=True )
    cereb_desc = map_segmentation_to_dataframe( 'cerebellum', cereb['parcellation_segmentation_image'] ).dropna(axis=0)

    if verbose:
        print( "antspyt1w.hierarchical complete" )
    mydataframes = {
        "rbp": myqc['brain'],
        "icv": myicv,
        "hemispheres":hemi,
        "tissues":tissue,
        "dktlobes":dktl,
        "dktregions":dktp,
        "dktcortex":dktc,
        "wmtracts_left":wmtdfL,
        "wmtracts_right":wmtdfR,
#        "wmh":myhypo['wmh_summary'],
        "mtl":deep_flash['mtl_description'],
        "bf":deep_bf['description'],
        "cit168":cit168lab_desc,
        "deep_cit168":deep_cit['description'],
        "snseg":snseg_desc,
#        "hippLR":hippLR['description'],
        "brainstem": brainstem_desc,
        "cerebellum": cereb_desc
        }

    outputs = {
        "brain_n4_dnz": img,
        "brain_n4_dnz_png": myqc['brain_image'],
        "brain_extraction": imgbxt,
        "tissue_seg_png": tissue_seg_png,
        "left_right": mylr,
        "dkt_parc": myparc,
        "registration":reg,
#        "hippLR":hippLR['segmentation'],
#        "white_matter_hypointensity":myhypo['wmh_probability_image'],
        "wm_tractsL":wm_tractsL,
        "wm_tractsR":wm_tractsR,
        "mtl":deep_flash['mtl_segmentation'],
        "bf":deep_bf['segmentation'],
        "deep_cit168lab":  deep_cit['segmentation'],
        "cit168lab":  cit168lab,
        "cit168reg":  cit168reg,
        "snseg":snseg,
        "snreg":snreg,
        "brainstem": brainstemseg,
        "cerebellum":cereb['parcellation_segmentation_image'],
        "dataframes": mydataframes
    }

    return outputs


def trim_segmentation_by_distance( segmentation, which_label, distance ):
    """
    trim a segmentation by the distance provided by the user. computes a distance
    transform from the segmentation - treated as binary - and trims the target
    label by that distance.

    Arguments
    ---------
    segmentation : ants image segmentation

    which_label : the label to trim

    distance : float distance value

    Returns
    -------
    trimmed_segmentation

    Example
    -------
    >>> import ants
    >>> img = ants.image_read( ants.get_data( 'r16' ) )
    >>> seg = ants.threshold_image( img, "Otsu", 3 )
    >>> tseg = antspyt1w.trim_segmentation_by_distance( seg, 1, 10 )
    """
    bseg = ants.threshold_image( segmentation, 1, segmentation.max() )
    dist = ants.iMath( bseg, "MaurerDistance" ) * (-1.0)
    disttrim = ants.threshold_image( dist, distance, dist.max() )
    tarseg = ants.threshold_image( segmentation, which_label, which_label ) * disttrim
    segmentationtrim = segmentation.clone()
    segmentationtrim[ segmentation == which_label ] = 0
    return segmentationtrim + tarseg * which_label



def zoom_syn( target_image, template, template_segmentations,
    initial_registration,
    dilation = 4,
    regIterations = [25] ):
    """
    zoomed in syn - a hierarchical registration applied to a hierarchical segmentation

    Initial registration is followed up by a refined and focused high-resolution registration.
    This is performed on the cropped image where the cropping region is determined
    by the first segmentation in the template_segmentations list.  Segmentations
    after the first one are assumed to exist as sub-regions of the first.  All
    segmentations are assumed to be binary.

    Arguments
    ---------
    target_image : ants image at original resolution

    template : ants image template to be mapped to the target image

    template_segmentations : list of binary segmentation images

    dilation : morphological dilation amount applied to the first segmentation and used for cropping

    regIterations : parameter passed to ants.registration

    Returns
    -------
    dictionary
        containing segmentation and registration results in addition to cropping results

    Example
    -------
    >>> import ants
    >>> ireg = ants.registration( target_image, template, "antsRegistrationSyNQuickRepro[s]" )
    >>> xxx = antspyt1w.zoom_syn(  orb,  template, level2segs, ireg )
    """
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
    return{
          'segmentations': orlist,
          'registration': synnerlow,
          'croppedimage': croplow,
          'croppingmask': cropper
          }


def read_hierarchical( output_prefix ):
    """
    standardized reading of output for hierarchical function

    Arguments
    ---------

    output_prefix : string path including directory and file prefix that was
        be applied to all output, both csv and images.

    Returns
    -------
    hierarchical_object : output of antspyt1w.hierarchical
        objects will be None if they cannot be found or if reading/writing
        is not yet implemented for them

    """

    mydataframes = {
            "rbp": None,
            "hemispheres":None,
            "tissues":None,
            "dktlobes":None,
            "dktregions":None,
            "dktcortex":None,
            "wmtracts_left":None,
            "wmtracts_right":None,
    #        "wmh":myhypo['wmh_summary'],
            "mtl":None,
            "bf":None,
            "cit168":None,
            "deep_cit168":None,
            "snseg":None,
#            "hippLR":None,
            "cerebellum":None,
            "brainstem":None
            }

    dkt_parc = {
        "tissue_segmentation":None,
        "tissue_probabilities":None,
        "dkt_parcellation":None,
        "dkt_lobes":None,
        "dkt_cortex": None,
        "hemisphere_labels": None,
        "wmSNR": None,
        "wmcsfSNR": None, }

    hierarchical_object = {
            "brain_n4_dnz": None,
            "brain_n4_dnz_png": None,
            "brain_extraction": None,
            "tissue_seg_png": None,
            "left_right": None,
            "dkt_parc": dkt_parc,
            "registration":None,
#            "hippLR":None,
    #        "white_matter_hypointensity":None,
            "wm_tractsL":None,
            "wm_tractsR":None,
            "mtl":None,
            "bf":None,
            "deep_cit168lab":  None,
            "cit168lab":  None,
            "cit168reg":  None,
            "snseg":None,
            "snreg":None,
            "cerebellum":None,
            "brainstem":None,
            "dataframes": mydataframes
        }

    for myvar in hierarchical_object['dataframes'].keys():
        if hierarchical_object['dataframes'][myvar] is None and exists( output_prefix + myvar + ".csv"):
            hierarchical_object['dataframes'][myvar] = pd.read_csv(output_prefix + myvar + ".csv")

    myvarlist = hierarchical_object.keys()
    for myvar in myvarlist:
        if hierarchical_object[myvar] is None and exists( output_prefix + myvar + '.nii.gz' ):
            hierarchical_object[myvar] = ants.image_read( output_prefix + myvar + '.nii.gz' )

    myvarlist = [
        'tissue_segmentation',
        'dkt_parcellation',
        'dkt_lobes',
        'dkt_cortex',
        'hemisphere_labels' ]
    for myvar in myvarlist:
        if hierarchical_object['dkt_parc'][myvar] is None and exists( output_prefix + myvar + '.nii.gz' ):
            hierarchical_object['dkt_parc'][myvar] = ants.image_read( output_prefix + myvar + '.nii.gz' )


    return hierarchical_object




def write_hierarchical( hierarchical_object, output_prefix, verbose=False ):
    """
    standardized writing of output for hierarchical function

    Arguments
    ---------
    hierarchical_object : output of antspyt1w.hierarchical

    output_prefix : string path including directory and file prefix that will
        be applied to all output, both csv and images.

    verbose: boolean

    Returns
    -------
    None

    """

    # write extant dataframes
    for myvar in hierarchical_object['dataframes'].keys():
        if hierarchical_object['dataframes'][myvar] is not None:
            if verbose:
                print( myvar )
            hierarchical_object['dataframes'][myvar].dropna(axis=0).to_csv(output_prefix + myvar + ".csv")

    myvarlist = hierarchical_object.keys()
    r16img = ants.image_read( ants.get_data( "r16" ))
    for myvar in myvarlist:
        if hierarchical_object[myvar] is not None and type(hierarchical_object[myvar]) == type( r16img ):
            if verbose:
                print( output_prefix + myvar )
            ants.image_write( hierarchical_object[myvar], output_prefix + myvar + '.nii.gz' )

    myvarlist = [
        'tissue_segmentation',
        'dkt_parcellation',
        'dkt_lobes',
        'dkt_cortex',
        'hemisphere_labels' ]
    for myvar in myvarlist:
        if hierarchical_object['dkt_parc'][myvar] is not None:
            if verbose:
                print( output_prefix + myvar )
            ants.image_write( hierarchical_object['dkt_parc'][myvar], output_prefix + myvar + '.nii.gz' )

    return




def merge_hierarchical_csvs_to_wide_format( hierarchical_dataframes, col_names = None , identifier=None, identifier_name='u_hier_id', verbose=False ):
    """
    standardized merging of output for dataframes produced by hierarchical function.

    Arguments
    ---------
    hierarchical_dataframes : output of antspyt1w.hierarchical

    identifier : unique subject identifier e.g. subject_001

    identifier_name : string name for the unique identifer column e.g. subject_id

    Returns
    -------
    data frame in wide format

    """
    if identifier is None:
        identifier='A'
    wide_df = pd.DataFrame( )
    icvkey='icv'
    if icvkey in hierarchical_dataframes.keys():
        temp = hierarchical_dataframes[icvkey].copy()
        temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
        temp.insert(loc=0, column=identifier_name, value=identifier)
        temp = temp.set_index(identifier_name)
        wide_df = wide_df.join(temp,how='outer')
    for myvar in hierarchical_dataframes.keys():
        if verbose:
            print( myvar )
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
    # handle wmh
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




def super_resolution_segmentation_per_label(
    imgIn,
    segmentation, # segmentation label image
    upFactor,
    sr_model,
    segmentation_numbers,
    dilation_amount = 0,
    probability_images=None, # probability list
    probability_labels=None, # the segmentation ids for the probability image,
    max_lab_plus_one=True,
    target_range=[1,0],
    poly_order = 'hist',
    verbose = False,
):
    """
    Apply a two-channel super resolution model to an image and probability pair.

    Arguments
    ---------
    imgIn : ANTsImage
        image to be upsampled

    segmentation : ANTsImage
        segmentation probability, n-ary or binary image

    upFactor : list
        the upsampling factors associated with the super-resolution model

    sr_model : tensorflow model or String
        for computing super-resolution; String denotes an interpolation type

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    dilation_amount : integer
        amount to pad around each segmentation defining region over which to
        computer the super-resolution in each label

    probability_images : list of ANTsImages
        list of probability images

    probability_labels : integer list
        providing the integer values associated with each probability image

    max_lab_plus_one : boolean
        add background label

    target_range : lower and upper limit of intensity expected by network [1,0] if trained by siq

    poly_order : integer or 'hist' or None

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `super_resolution` : ANTsImage
            super_resolution image

        `super_resolution_segmentation` : ANTsImage
            super_resolution_segmentation image

        `segmentation_geometry` : list of data frame types
            segmentation geometry for each label

        `probability_images` : list of ANTsImage
            segmentation probability maps


    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    """
    import re
    if type( sr_model ) == type(""):
        if re.search( 'h5', sr_model ) is not None:
            if verbose:
                print("load model")
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
        if verbose:
            print( "SR-per-label:" + str( locallab ) )
        binseg = ants.threshold_image( segmentationUse, locallab, locallab )
        sizethresh = 2
        if ( binseg == 1 ).sum() < sizethresh :
            warnings.warn( "SR-per-label:" + str( locallab ) + ' is small' )
        if ( binseg == 1 ).sum() >= sizethresh :
            minprob="NA"
            maxprob="NA"
            if probability_images is not None:
                whichprob = probability_labels.index(locallab)
                probimg = probability_images[whichprob].resample_image_to_target( binseg )
                minprob = min( probimg[ binseg >= 0.5 ] )
                maxprob = max( probimg[ binseg >= 0.5 ] )
            if verbose:
                print( "SR-per-label:" + str( locallab ) + " min/max-prob: " + str(minprob)+ " / " + str(maxprob)  )
            binsegdil = ants.iMath( ants.threshold_image( segmentationUse, locallab, locallab ), "MD", dilation_amount )
            binsegdil2input = ants.resample_image_to_target( binsegdil, imgIn, interp_type='nearestNeighbor'  )
            imgc = ants.crop_image( ants.iMath(imgIn,"Normalize"), binsegdil2input )
            imgc = imgc * target_range[0] - target_range[1] # for SR
            imgchCore = ants.crop_image( binseg, binsegdil )
            if probability_images is not None:
                imgchCore = ants.crop_image( probimg, binsegdil )
#            imgch = ants.iMath( imgchCore, "Normalize" ) * target_range[0] - target_range[1] # for SR
            imgch = imgchCore * target_range[0] - target_range[1] # for SR
            if type( sr_model ) == type(""): # this is just for testing
                binsegup = ants.resample_image_to_target( binseg, imgup, interp_type='linear' )
                problist.append( binsegup )
            else:
                if verbose:
                    print(imgc)
                    print(imgch)
                myarr = np.stack( [imgc.numpy(),imgch.numpy()],axis=3 )
                newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
                myarr = myarr.reshape( newshape )
                if verbose:
                    print("calling prediction function")
                    print("arr range " + str(myarr.max() ) + " " + str(myarr.min() ) )
                    print( myarr.shape )
                pred = sr_model.predict( myarr )
                predshape = np.asarray(pred).shape
                predshapelen = len( predshape )
                if predshape[ predshapelen - 1] == 2:
                    pred = tf.split( pred, 2, axis=4 )
                if verbose:
                    print("predict done")
                imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
                if verbose:
                    print( imgc )
                    print( imgsr )
                imgsr = ants.copy_image_info( imgc, imgsr )
                ants.set_spacing( imgsr,  newspc )
                imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
                imgsrh = ants.copy_image_info( imgc, imgsrh )
                ants.set_spacing( imgsrh,  newspc )
                if poly_order is not None:
                    if poly_order == 'hist':
                        if verbose:
                            print("hist match intensity")
                        imgsrh = ants.histogram_match_image( imgsrh, imgchCore )
                        imgsr = ants.histogram_match_image( imgsr, imgc )
                    else:
                        if verbose:
                            print("poly match intensity")
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


    if True: # we dont generally know which of these is background/foreground so .....
        segmat = ants.images_to_matrix(problist, tarmask)
        finalsegvec = segmat.argmax(axis=0)
        finalsegvec2 = finalsegvec.copy()
        for i in range(len(problist)):
            segnum = segmentation_numbers_use[i]
            finalsegvec2[finalsegvec == i] = segnum
        outimg = ants.make_image(tarmask, finalsegvec2)
        outimg = ants.mask_image( outimg, outimg, segmentation_numbers )
        seggeom = ants.label_geometry_measures( outimg )
    else:
        image_matrix = ants.image_list_to_matrix(problist[0:(len(problist)-1)], tarmask )
        background_foreground_matrix = np.stack([ants.image_list_to_matrix([problist[len(problist)-1]], tarmask),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
        foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
        segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
        segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), tarmask)[0]
        outimg = ants.image_clone(segmentation_image)
        for i in range(len(problist)):
            outimg[segmentation_image==i] = segmentation_numbers_use[i]
        outimg = ants.mask_image( outimg, outimg, segmentation_numbers )
        seggeom = ants.label_geometry_measures( outimg )

    return {
        "super_resolution": imgsrfull2,
        "super_resolution_segmentation": outimg,
        "segmentation_geometry": seggeom,
        "probability_images": problist
        }




def super_resolution_segmentation_with_probabilities(
    img,
    initial_probabilities,
    sr_model,
    target_range=[1,0],
    verbose = False
):
    """
    Simultaneous super-resolution and probabilistic segmentation.

    this function will perform localized super-resolution analysis on the
    underlying image content to produce a SR version of both the local region
    intentsity and segmentation probability.  the number of probability images
    determines the number of outputs for the intensity and probabilitiy list
    that are returned.  NOTE: should consider inverse logit transform option.

    Arguments
    ---------
    img : input intensity image

    initial_probabilities: original resolution probability images

    sr_model : the super resolution model - should have 2 channel input

    target_range : lower and upper limit of intensity expected by network [1,0] if trained by siq

    verbose : boolean

    Returns
    -------
    list of ANTsImages

    """

    # SR Part
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
            if verbose:
                print(k)
                print(imgc)
            imgcrescale = ants.iMath( imgc, "Normalize" ) * target_range[0] - target_range[1] # for SR
            imgchrescale = imgch * target_range[0] - target_range[1]
            myarr = np.stack( [ imgcrescale.numpy(), imgchrescale.numpy() ],axis=3 )
            newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
            myarr = myarr.reshape( newshape )
            pred = sr_model.predict( myarr )
            imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
            imgsr = ants.copy_image_info( imgc, imgsr )
            newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
            ants.set_spacing( imgsr,  newspc )
            if verbose:
                print(imgsr)
            imgsr = ants.regression_match_image( imgsr, ants.resample_image_to_target(imgc,imgsr) )
            imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
            imgsrh = ants.copy_image_info( imgsr, imgsrh )
            tempup = ants.resample_image_to_target( tempm, imgsr )
            srimglist.append( imgsr )
            # NOTE: get rid of pixellated junk/artifacts - acts like a prior
            srproblist.append( imgsrh * tempup )

    if verbose:
        print("done srwithprob")

    labels = {
        'sr_intensities':srimglist,
        'sr_probabilities':srproblist,
    }
    return labels

def kelly_kapowski_thickness( x, labels,
    label_description='dkt', iterations=45, max_thickness=6.0, mydap=None, verbose=False ):
    """
    Apply a two-channel super resolution model to an image and probability pair.

    Arguments
    ---------
    x : ANTsImage
        raw t1 image

    labels : ANTsImage
        cortical parcellation

    label_description : pandas data frame

    iterations : integer
        number of iterations ( probably not to be changed except for testing )

    max_thickness : in mm to consider default is 6.0

    verbose : boolean

    Returns
    -------

    dictionary w/ following key/value pairs:
        `thickness_image` : ANTsImage

        `thickness_dataframe` : dataframe with mean thickness values

    Example
    -------
    >>> import ants
    >>> thk = antspyt1w.kelly_kapowski_thickness( x )
    """
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
    return {
        'thickness_image' : kkthk,
        'thickness_dataframe' : kkdf_wide
    }


def minimal_sr_preprocessing( x, imgbxt=None ):
    """
    x : input t1 image (raw)

    imgbxt : optional existing brain extraction

    outputs: preprocessedT1, hemisphereLabels
    """
    if x.dimension != 3:
        raise ValueError('hierarchical: input image should be 3-dimensional')

    tfn = get_data('T_template0', target_extension='.nii.gz' )
    tlrfn = get_data('T_template0_LR', target_extension='.nii.gz' )

    ##### read images and do simple bxt ops
    templatea = ants.image_read( tfn )
    templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
    templatealr = ants.image_read( tlrfn )
    if imgbxt is None:
        imgbxt = brain_extraction( ants.iMath( x, "Normalize" ) )
    img = preprocess_intensity( ants.iMath( x, "Normalize" ), imgbxt )
    img = ants.iMath( img, "Normalize" )
    mylr = label_hemispheres( img, templatea, templatealr )
    return img, mylr * imgbxt
