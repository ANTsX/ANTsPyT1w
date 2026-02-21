# ANTsPyT1w

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/stnava/ANTsPyT1w/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/stnava/ANTsPyT1w/tree/main)

## reference processing for t1-weighted neuroimages (human)

the outputs of these processes can be used for data inspection/cleaning/triage
as well for interrogating neuroscientific hypotheses.

this package also keeps track of the latest preferred algorithm variations for
production environments.

install by calling (within the source directory):

```
python setup.py install
```

or install via `pip install antspyt1w`

Function-specific documentation is [here](https://antsx.github.io/ANTsPyT1w/antspyt1w/get_data.html).

# what this will do

- provide example data

- brain extraction

- denoising

- n4 bias correction

- brain parcellation into tissues, hemispheres, lobes and regions

- hippocampus specific segmentation

- t1 hypointensity segmentation and classification *exploratory*

- deformable registration with robust and repeatable parameters

- registration-based labeling of major white matter tracts

- helpers that organize and annotate segmentation variables into data frames

- hypothalamus segmentation *FIXME/TODO*


the two most time-consuming processes are hippocampus-specific segentation
(because it uses augmentation) and registration.  both take 10-20 minutes
depending on your available computational resources and the data.  both
could be made computationally cheaper at the cost of accuracy/reliability.

# first time setup

```python
import antspyt1w
antspyt1w.get_data()
```

NOTE: `get_data` has a `force_download` option to make sure the latest
package data is installed.

# example processing

```python
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import antspyt1w
import antspynet
import ants

##### get example data + reference templates
# NOTE:  PPMI-3803-20120814-MRI_T1-I340756 is a good example of our naming style
# Study-SubjectID-Date-Modality-UniqueID
# where Modality could also be measurement or something else
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
img = ants.image_read( fn )

# generalized default processing
myresults = antspyt1w.hierarchical( img, output_prefix = '/tmp/XXX' )

##### organize summary data into data frames - user should pivot these to columns
# and attach to unique IDs when accumulating for large-scale studies
# see below for how to easily pivot into wide format
# https://stackoverflow.com/questions/28337117/how-to-pivot-a-dataframe-in-pandas


```

An example "full study" (at small scale) is illustrated in `~/.antspyt1w/run_dlbs.py`
which demonstrates/comments on:
- how to aggregate dataframes
- how to pivot to wide format
- how to join with a demographic/metadata file
- visualizing basic outcomes.

## ssl error 

if you get an odd certificate error when calling `force_download`, try:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## to publish a release

before doing this - make sure you have a recent run of `pip-compile pyproject.toml`

```
rm -r -f build/ antspyt1w.egg-info/ dist/
python3 -m  build .
python3 -m pip install --upgrade twine
python3 -m twine upload --repository antspyt1w dist/*
```

## to publish docs

```bash
pdoc antspyt1w -o docs
git add docs/ && git commit -m "DOC: update docs" && git push
# Settings → Pages → Source = main, Folder = /docs
```

