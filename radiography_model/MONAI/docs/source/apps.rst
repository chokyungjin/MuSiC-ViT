:github_url: https://github.com/Project-MONAI/MONAI

.. _apps:

Applications
============
.. currentmodule:: monai.apps

`Datasets`
----------

.. autoclass:: MedNISTDataset
    :members:

.. autoclass:: DecathlonDataset
    :members:

.. autoclass:: CrossValidation
    :members:

`Utilities`
-----------

.. autofunction:: check_hash

.. autofunction:: download_url

.. autofunction:: extractall

.. autofunction:: download_and_extract

`Deepgrow`
----------

.. automodule:: monai.apps.deepgrow.dataset
.. autofunction:: create_dataset

.. automodule:: monai.apps.deepgrow.interaction
.. autoclass:: Interaction
    :members:

.. automodule:: monai.apps.deepgrow.transforms
.. autoclass:: AddInitialSeedPointd
    :members:
.. autoclass:: AddGuidanceSignald
    :members:
.. autoclass:: AddRandomGuidanced
    :members:
.. autoclass:: AddGuidanceFromPointsd
    :members:
.. autoclass:: SpatialCropForegroundd
    :members:
.. autoclass:: SpatialCropGuidanced
    :members:
.. autoclass:: RestoreLabeld
    :members:
.. autoclass:: ResizeGuidanced
    :members:
.. autoclass:: FindDiscrepancyRegionsd
    :members:
.. autoclass:: FindAllValidSlicesd
    :members:
.. autoclass:: Fetch2DSliced
    :members:

`Pathology`
-----------

.. automodule:: monai.apps.pathology.datasets
.. autoclass:: PatchWSIDataset
    :members:
.. autoclass:: SmartCachePatchWSIDataset
    :members:

.. automodule:: monai.apps.pathology.utils
.. autoclass:: PathologyProbNMS
    :members:
