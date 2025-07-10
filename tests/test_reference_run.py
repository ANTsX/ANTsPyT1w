import sys, os
import unittest
mynt="48"
os.environ["TF_NUM_INTEROP_THREADS"] = mynt
os.environ["TF_NUM_INTRAOP_THREADS"] = mynt
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = mynt
import tempfile
import shutil
import antspyt1w
import antspynet
import ants
import math
import re
import pandas as pd
# just test that things loaded ok
if os.getenv('CI') == 'true' and os.getenv('CIRCLECI') == 'true':
    def test_simple():
        assert os.getenv('CI') == 'true' and os.getenv('CIRCLECI') == 'true'
    def test_download():
        assert antspyt1w.get_data() == None
    def test_img():
        fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
        img = ants.image_read( fn )
        assert img is not None
    def test_inspect():
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            tempfn=temp_dir+'/apt1wtest'
            testhier = antspyt1w.inspect_raw_t1( img, tempfn )
            assert math.fabs(testhier['brain']['resnetGrade'].iloc[0]-1.56) < 0.1
else:
    def test_download():
        assert antspyt1w.get_data() == None
    def test_img():
        fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
        img = ants.image_read( fn )
        assert img is not None
    def test_inspect():
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            tempfn=temp_dir+'/apt1wtest'
            testhier = antspyt1w.inspect_raw_t1( img, tempfn )
            assert math.fabs(testhier['brain']['resnetGrade'].iloc[0]-1.56) < 0.1
    def test_nbm():
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            tempfn=temp_dir+'/apt1wtest'
            braintissuemask = antspyt1w.brain_extraction( img  )
            print("dbf")
            deep_bf = antspyt1w.deep_nbm( img * braintissuemask,
                antspyt1w.get_data("deep_nbm_rank",target_extension='.h5'),
                csfquantile=None, aged_template=True, verbose=True )
            return deep_bf
    def test_deepcit():
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            tempfn=temp_dir+'/apt1wtest'
            braintissuemask = antspyt1w.brain_extraction( img  )
            print("cit")
            deep_bf = antspyt1w.deep_cit168( img * braintissuemask, verbose=True )
            return deep_bf
    def test_hier():
        with tempfile.TemporaryDirectory() as temp_dir:
            fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
            img = ants.image_read( fn )
            tempfn=temp_dir+'/apt1wtest'
            testhier = antspyt1w.hierarchical( img, output_prefix=tempfn, labels_to_register=None, imgbxt=None, cit168=False, is_test=True, verbose=True)
            mmdfw = antspyt1w.merge_hierarchical_csvs_to_wide_format( testhier['dataframes'] )
            print( mmdfw )
            return testhier
            # assert math.fabs(testhier['brain_image']['resnetGrade']-1.56) < 0.1


if False:
    import ants
    import antspyt1w
    fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
    img = ants.image_read( fn )
    tempfn='/tmp/apt1wtestA'
    testhier0 = antspyt1w.hierarchical( img, output_prefix=tempfn, labels_to_register=None, imgbxt=None, cit168=False, is_test=True, verbose=True)
    tempfn='/tmp/apt1wtestB'
    testhier1 = antspyt1w.hierarchical( img, output_prefix=tempfn, labels_to_register=None, imgbxt=None, cit168=False, is_test=True, verbose=True)
