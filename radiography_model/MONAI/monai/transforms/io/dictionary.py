# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for IO functions
defined in :py:class:`monai.transforms.io.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Optional, Union

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage
from monai.transforms.transform import MapTransform
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode

__all__ = [
    "LoadImaged",
    "LoadImageD",
    "LoadImageDict",
    "SaveImaged",
    "SaveImageD",
    "SaveImageDict",
]


class LoadImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    must load image and metadata together. If loading a list of files in one key,
    stack them together and add a new dimension as the first dimension, and use the
    meta data of the first image to represent the stacked result. Note that the affine
    transform of all the stacked images should be same. The output metadata field will
    be created as ``key_{meta_key_postfix}``.

    It can automatically choose readers based on the supported suffixes and in below order:
    - User specified reader at runtime when call this loader.
    - Registered readers from the latest to the first in list.
    - Default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
    (npz, npy -> NumpyReader), (others -> ITKReader).

    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
        image_only: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            reader: register reader to load image file and meta data, if None, still can register readers
                at runtime or use the default readers. If a string of reader name provided, will construct
                a reader object with the `*args` and `**kwargs` parameters, supported reader name: "NibabelReader",
                "PILReader", "ITKReader", "NumpyReader".
            dtype: if not None convert the loaded image data to this data type.
            meta_key_postfix: use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.
            image_only: if True return dictionary containing just only the image volumes, otherwise return
                dictionary containing image data array and header dict per input key.
            allow_missing_keys: don't raise exception if key is missing.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.
        """
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key in self.key_iterator(d):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                if not isinstance(data, np.ndarray):
                    raise ValueError("loader must return a numpy array (because image_only=True was used).")
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                key_to_add = f"{key}_{self.meta_key_postfix}"
                if key_to_add in d and not self.overwriting:
                    raise KeyError(f"Meta data with key {key_to_add} already exists and overwriting=False.")
                d[key_to_add] = data[1]
        return d


class SaveImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SaveImage`.

    NB: image should include channel dimension: [B],C,H,W,[D].

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        meta_key_postfix: `key_{postfix}` was used to store the metadata in `LoadImaged`.
            So need the key to extract metadata to save images, default is `meta_dict`.
            The meta data is a dictionary object, if no corresponding metadata, set to `None`.
            For example, for data with key `image`, the metadata by default is in `image_meta_dict`.
        output_dir: output image directory.
        output_postfix: a string appended to all output file names, default to `trans`.
        output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
        resample: whether to resample before saving the data array.
            if saving PNG format image, based on the `spatial_shape` from metadata.
            if saving NIfTI format image, based on the `original_affine` from metadata.
        mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

            - NIfTI files {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

            - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files
                This option is ignored.

        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
            it's used for PNG format only.
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
            it's used for NIfTI format only.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
            it's used for NIfTI format only.
        save_batch: whether the import image is a batch data, default to `False`.
            usually pre-transforms run for channel first data, while post-transforms run for batch data.
        allow_missing_keys: don't raise exception if key is missing.
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
            image will always be saved as (H,W,D,C).
            it's used for NIfTI format only.

    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = "meta_dict",
        output_dir: str = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        save_batch: bool = False,
        allow_missing_keys: bool = False,
        squeeze_end_dims: bool = True,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.meta_key_postfix = meta_key_postfix
        self._saver = SaveImage(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            resample=resample,
            mode=mode,
            padding_mode=padding_mode,
            scale=scale,
            dtype=dtype,
            output_dtype=output_dtype,
            save_batch=save_batch,
            squeeze_end_dims=squeeze_end_dims,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            meta_data = d[f"{key}_{self.meta_key_postfix}"] if self.meta_key_postfix is not None else None
            self._saver(img=d[key], meta_data=meta_data)
        return d


LoadImageD = LoadImageDict = LoadImaged
SaveImageD = SaveImageDict = SaveImaged
