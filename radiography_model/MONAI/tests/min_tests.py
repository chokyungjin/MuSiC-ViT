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

import glob
import os
import sys
import unittest


def run_testsuit():
    """
    Load test cases by excluding those need external dependencies.
    The loaded cases should work with "requirements-min.txt"::

        # in the monai repo folder:
        pip install -r requirements-min.txt
        QUICKTEST=true python -m tests.min_tests

    :return: a test suite
    """
    exclude_cases = [  # these cases use external dependencies
        "test_ahnet",
        "test_arraydataset",
        "test_cachedataset",
        "test_cachedataset_parallel",
        "test_dataset",
        "test_detect_envelope",
        "test_iterable_dataset",
        "test_ensemble_evaluator",
        "test_handler_checkpoint_loader",
        "test_handler_checkpoint_saver",
        "test_handler_classification_saver",
        "test_handler_lr_scheduler",
        "test_handler_confusion_matrix",
        "test_handler_confusion_matrix_dist",
        "test_handler_hausdorff_distance",
        "test_handler_mean_dice",
        "test_handler_rocauc",
        "test_handler_rocauc_dist",
        "test_handler_segmentation_saver",
        "test_handler_smartcache",
        "test_handler_stats",
        "test_handler_surface_distance",
        "test_handler_tb_image",
        "test_handler_tb_stats",
        "test_handler_validation",
        "test_hausdorff_distance",
        "test_header_correct",
        "test_hilbert_transform",
        "test_img2tensorboard",
        "test_integration_segmentation_3d",
        "test_integration_sliding_window",
        "test_integration_unet_2d",
        "test_integration_workflows",
        "test_integration_workflows_gan",
        "test_keep_largest_connected_component",
        "test_keep_largest_connected_componentd",
        "test_lltm",
        "test_lmdbdataset",
        "test_load_image",
        "test_load_imaged",
        "test_load_spacing_orientation",
        "test_mednistdataset",
        "test_image_dataset",
        "test_nifti_header_revise",
        "test_nifti_rw",
        "test_nifti_saver",
        "test_orientation",
        "test_orientationd",
        "test_parallel_execution",
        "test_persistentdataset",
        "test_cachentransdataset",
        "test_pil_reader",
        "test_plot_2d_or_3d_image",
        "test_png_rw",
        "test_png_saver",
        "test_rand_rotate",
        "test_rand_rotated",
        "test_rand_zoom",
        "test_rand_zoomd",
        "test_resize",
        "test_resized",
        "test_rotate",
        "test_rotated",
        "test_smartcachedataset",
        "test_spacing",
        "test_spacingd",
        "test_surface_distance",
        "test_zoom",
        "test_zoom_affine",
        "test_zoomd",
        "test_occlusion_sensitivity",
        "test_torchvision",
        "test_torchvisiond",
        "test_handler_metrics_saver",
        "test_handler_metrics_saver_dist",
        "test_evenly_divisible_all_gather_dist",
        "test_handler_classification_saver_dist",
        "test_deepgrow_transforms",
        "test_deepgrow_interaction",
        "test_deepgrow_dataset",
        "test_save_image",
        "test_save_imaged",
        "test_ensure_channel_first",
        "test_ensure_channel_firstd",
    ]
    assert sorted(exclude_cases) == sorted(set(exclude_cases)), f"Duplicated items in {exclude_cases}"

    files = glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))

    cases = []
    for case in files:
        test_module = os.path.basename(case)[:-3]
        if test_module in exclude_cases:
            exclude_cases.remove(test_module)
            print(f"skipping tests.{test_module}.")
        else:
            cases.append(f"tests.{test_module}")
    assert not exclude_cases, f"items in exclude_cases not used: {exclude_cases}."
    test_suite = unittest.TestLoader().loadTestsFromNames(cases)
    return test_suite


if __name__ == "__main__":

    # testing import submodules
    from monai.utils.module import load_submodules

    _, err_mod = load_submodules(sys.modules["monai"], True)
    if err_mod:
        print(err_mod)
        # expecting that only engines and handlers are not imported
        assert sorted(err_mod) == ["monai.engines", "monai.handlers"]

    # testing all modules
    test_runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = test_runner.run(run_testsuit())
    sys.exit(int(not result.wasSuccessful()))
