# Changelog
All notable changes to MONAI are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.4.0] - 2020-12-15
### Added
* Overview document for [feature highlights in v0.4.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Torchscript support for the net modules
* New networks and layers:
  * Discrete Gaussian kernels
  * Hilbert transform and envelope detection
  * Swish and mish activation
  * Acti-norm-dropout block
  * Upsampling layer
  * Autoencoder, Variational autoencoder
  * FCNet
* Support of initialisation from pretrained weights for densenet, senet, multichannel AHNet
* Layer-wise learning rate API
* New model metrics and event handlers based on occlusion sensitivity, confusion matrix, surface distance
* CAM/GradCAM/GradCAM++
* File format-agnostic image loader APIs with Nibabel, ITK readers
* Enhancements for dataset partition, cross-validation APIs
* New data APIs:
  * LMDB-based caching dataset
  * Cache-N-transforms dataset
  * Iterable dataset
  * Patch dataset
* Weekly PyPI release
* Fully compatible with PyTorch 1.7
* CI/CD enhancements:
  * Skipping, speed up, fail fast, timed, quick tests
  * Distributed training tests
  * Performance profiling utilities
* New tutorials and demos:
  * Autoencoder, VAE tutorial
  * Cross-validation demo
  * Model interpretability tutorial
  * COVID-19 Lung CT segmentation challenge open-source baseline
  * Threadbuffer demo
  * Dataset partitioning tutorial
  * Layer-wise learning rate demo
  * [MONAI Bootcamp 2020](https://github.com/Project-MONAI/MONAIBootcamp2020)

### Changed
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:20.10-py3` from `nvcr.io/nvidia/pytorch:20.08-py3`

#### Backwards Incompatible Changes
* `monai.apps.CVDecathlonDataset` is extended to a generic `monai.apps.CrossValidation` with an `dataset_cls` option
* Cache dataset now requires a `monai.transforms.Compose` instance as the transform argument
* Model checkpoint file name extensions changed from `.pth` to `.pt`
* Readers' `get_spatial_shape` returns a numpy array instead of list
* Decoupled postprocessing steps such as `sigmoid`, `to_onehot_y`, `mutually_exclusive`, `logit_thresh` from metrics and event handlers,
the postprocessing steps should be used before calling the metrics methods
* `ConfusionMatrixMetric` and `DiceMetric` computation now returns an additional `not_nans` flag to indicate valid results
* `UpSample` optional `mode` now supports `"deconv"`, `"nontrainable"`, `"pixelshuffle"`; `interp_mode` is only used when `mode` is `"nontrainable"`
* `SegResNet` optional `upsample_mode` now supports `"deconv"`, `"nontrainable"`, `"pixelshuffle"`
* `monai.transforms.Compose` class inherits `monai.transforms.Transform`
* In `Rotate`, `Rotated`, `RandRotate`, `RandRotated`  transforms, the `angle` related parameters are interpreted as angles in radians instead of degrees.
* `SplitChannel` and `SplitChanneld` moved from `transforms.post` to `transforms.utility`

### Removed
* Support of PyTorch 1.4

### Fixed
* Enhanced loss functions for stability and flexibility
* Sliding window inference memory and device issues
* Revised transforms:
  * Normalize intensity datatype and normalizer types
  * Padding modes for zoom
  * Crop returns coordinates
  * Select items transform
  * Weighted patch sampling
  * Option to keep aspect ratio for zoom
* Various CI/CD issues

## [0.3.0] - 2020-10-02
### Added
* Overview document for [feature highlights in v0.3.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Automatic mixed precision support
* Multi-node, multi-GPU data parallel model training support
* 3 new evaluation metric functions
* 11 new network layers and blocks
* 6 new network architectures
* 14 new transforms, including an I/O adaptor
* Cross validation module for `DecathlonDataset`
* Smart Cache module in dataset
* `monai.optimizers` module
* `monai.csrc` module
* Experimental feature of ImageReader using ITK, Nibabel, Numpy, Pillow (PIL Fork)
* Experimental feature of differentiable image resampling in C++/CUDA
* Ensemble evaluator module
* GAN trainer module
* Initial cross-platform CI environment for C++/CUDA code
* Code style enforcement now includes isort and clang-format
* Progress bar with tqdm

### Changed
* Now fully compatible with PyTorch 1.6
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:20.08-py3` from `nvcr.io/nvidia/pytorch:20.03-py3`
* Code contributions now require signing off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org/)
* Major work in type hinting finished
* Remote datasets migrated to [Open Data on AWS](https://registry.opendata.aws/)
* Optionally depend on PyTorch-Ignite v0.4.2 instead of v0.3.0
* Optionally depend on torchvision, ITK
* Enhanced CI tests with 8 new testing environments
### Removed
* `MONAI/examples` folder (relocated into [`Project-MONAI/tutorials`](https://github.com/Project-MONAI/tutorials))
* `MONAI/research` folder (relocated to [`Project-MONAI/research-contributions`](https://github.com/Project-MONAI/research-contributions))
### Fixed
* `dense_patch_slices` incorrect indexing
* Data type issue in `GeneralizedWassersteinDiceLoss`
* `ZipDataset` return value inconsistencies
* `sliding_window_inference` indexing and `device` issues
* importing monai modules may cause namespace pollution
* Random data splits issue in `DecathlonDataset`
* Issue of randomising a `Compose` transform
* Various issues in function type hints
* Typos in docstring and documentation
* `PersistentDataset` issue with existing file folder
* Filename issue in the output writers

## [0.2.0] - 2020-07-02
### Added
* Overview document for [feature highlights in v0.2.0](https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md)
* Type hints and static type analysis support
* `MONAI/research` folder
* `monai.engine.workflow` APIs for supervised training
* `monai.inferers` APIs for validation and inference
* 7 new tutorials and examples
* 3 new loss functions
* 4 new event handlers
* 8 new layers, blocks, and networks
* 12 new transforms, including post-processing transforms
* `monai.apps.datasets` APIs, including `MedNISTDataset` and `DecathlonDataset`
* Persistent caching, `ZipDataset`, and `ArrayDataset` in `monai.data`
* Cross-platform CI tests supporting multiple Python versions
* Optional import mechanism
* Experimental features for third-party transforms integration
### Changed
> For more details please visit [the project wiki](https://github.com/Project-MONAI/MONAI/wiki/Notable-changes-between-0.1.0-and-0.2.0)
* Core modules now require numpy >= 1.17
* Categorized `monai.transforms` modules into crop and pad, intensity, IO, post-processing, spatial, and utility.
* Most transforms are now implemented with PyTorch native APIs
* Code style enforcement and automated formatting workflows now use autopep8 and black
* Base Docker image upgraded to `nvcr.io/nvidia/pytorch:20.03-py3` from `nvcr.io/nvidia/pytorch:19.10-py3`
* Enhanced local testing tools
* Documentation website domain changed to https://docs.monai.io
### Removed
* Support of Python < 3.6
* Automatic installation of optional dependencies including pytorch-ignite, nibabel, tensorboard, pillow, scipy, scikit-image
### Fixed
* Various issues in type and argument names consistency
* Various issues in docstring and documentation site
* Various issues in unit and integration tests
* Various issues in examples and notebooks

## [0.1.0] - 2020-04-17
### Added
* Public alpha source code release under the Apache 2.0 license ([highlights](https://github.com/Project-MONAI/MONAI/blob/0.1.0/docs/source/highlights.md))
* Various tutorials and examples
  - Medical image classification and segmentation workflows
  - Spacing/orientation-aware preprocessing with CPU/GPU and caching
  - Flexible workflows with PyTorch Ignite and Lightning
* Various GitHub Actions
  - CI/CD pipelines via self-hosted runners
  - Documentation publishing via readthedocs.org
  - PyPI package publishing
* Contributing guidelines
* A project logo and badges

[highlights]: https://github.com/Project-MONAI/MONAI/blob/master/docs/source/highlights.md

[Unreleased]: https://github.com/Project-MONAI/MONAI/compare/0.4.0...HEAD
[0.4.0]: https://github.com/Project-MONAI/MONAI/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/Project-MONAI/MONAI/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/Project-MONAI/MONAI/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/Project-MONAI/MONAI/commits/0.1.0
