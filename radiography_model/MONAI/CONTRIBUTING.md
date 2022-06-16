- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  * [Preparing pull requests](#preparing-pull-requests)
    1. [Checking the coding style](#checking-the-coding-style)
    1. [Unit testing](#unit-testing)
    1. [Building the documentation](#building-the-documentation)
    1. [Automatic code formatting](#automatic-code-formatting)
    1. [Signing your work](#signing-your-work)
    1. [Utility functions](#utility-functions)
  * [Submitting pull requests](#submitting-pull-requests)
- [The code reviewing process (for the maintainers)](#the-code-reviewing-process)
  * [Reviewing pull requests](#reviewing-pull-requests)
- [Admin tasks (for the maintainers)](#admin-tasks)
  * [Releasing a new version](#release-a-new-version)

## Introduction


This documentation is intended for individuals and institutions interested in contributing to MONAI. MONAI is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

### Communicate with us

We are happy to talk with you about your needs for MONAI and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point.

### Does it belong in PyTorch instead of MONAI?

MONAI is part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/), and mainly based on the PyTorch and Numpy libraries. These libraries implement what we consider to be best practice for general scientific computing and deep learning functionality. MONAI builds on these with a strong focus on medical applications. As such, it is a good idea to consider whether your functionality is medical-application specific or not. General deep learning functionality may be better off in PyTorch; you can find their contribution guidelines [here](https://pytorch.org/docs/stable/community/contribution_guide.html).

## The contribution process

_Pull request early_

We encourage you to create pull requests early. It helps us track the contributions under development, whether they are ready to be merged or not. Change your pull request's title to begin with `[WIP]` until it is ready for formal review.

Please note that, as per PyTorch, MONAI uses American English spelling. This means classes and variables should be: normali**z**e, visuali**z**e, colo~~u~~r, etc.

### Preparing pull requests
To ensure the code quality, MONAI relies on several linting tools ([flake8 and its plugins](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), [isort](https://github.com/timothycrosley/isort)),
static type analysis tools ([mypy](https://github.com/python/mypy), [pytype](https://github.com/google/pytype)), as well as a set of unit/integration tests.

This section highlights all the necessary preparation steps required before sending a pull request.
To collaborate efficiently, please read through this section and follow them.

* [Checking the coding style](#checking-the-coding-style)
* [Unit testing](#unit-testing)
* [Building documentation](#building-the-documentation)
* [Signing your work](#signing-your-work)

#### Checking the coding style
Coding style is checked and enforced by flake8, black, and isort, using [a flake8 configuration](./setup.cfg) similar to [PyTorch's](https://github.com/pytorch/pytorch/blob/master/.flake8).
Before submitting a pull request, we recommend that all linting should pass, by running the following command locally:

```bash
# optionally update the dependencies and dev tools
python -m pip install -U pip
python -m pip install -U -r requirements-dev.txt

# run the linting and type checking tools
./runtests.sh --codeformat

# try to fix the coding style errors automatically
./runtests.sh --autofix
```

License information: all source code files should start with this paragraph:
```
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

```

##### Exporting modules

If you intend for any variables/functions/classes to be available outside of the file with the edited functionality, then:

- Create or append to the `__all__` variable (in the file in which functionality has been added), and
- Add to the `__init__.py` file.

#### Unit testing
MONAI tests are located under `tests/`.

- The unit test's file name follows `test_[module_name].py`.
- The integration test's file name follows `test_integration_[workflow_name].py`.

A bash script (`runtests.sh`) is provided to run all tests locally.
Please run ``./runtests.sh -h`` to see all options.

To run a particular test, for example `tests/test_dice_loss.py`:
```
python -m tests.test_dice_loss
```

Before submitting a pull request, we recommend that all linting and unit tests
should pass, by running the following command locally:

```bash
./runtests.sh -f -u --net --coverage
```
or (for new features that would not break existing functionality):

```bash
./runtests.sh --quick --unittests
```

It is recommended that the new test `test_[module_name].py` is constructed by using only
python 3.6+ build-in functions, `torch`, `numpy`, `coverage` (for reporting code coverages) and `parameterized` (for organising test cases) packages.
If it requires any other external packages, please make sure:
- the packages are listed in [`requirements-dev.txt`](requirements-dev.txt)
- the new test `test_[module_name].py` is added to the `exclude_cases` in [`./tests/min_tests.py`](./tests/min_tests.py) so that
the minimal CI runner will not execute it.

_If it's not tested, it's broken_

All new functionality should be accompanied by an appropriate set of tests.
MONAI functionality has plenty of unit tests from which you can draw inspiration,
and you can reach out to us if you are unsure of how to proceed with testing.

MONAI's code coverage report is available at [CodeCov](https://codecov.io/gh/Project-MONAI/MONAI).

#### Building the documentation
MONAI's documentation is located at `docs/`.

```bash
# install the doc-related dependencies
pip install --upgrade pip
pip install -r docs/requirements.txt

# build the docs
cd docs/
make html
```
The above commands build html documentation, they are used to automatically generate [https://docs.monai.io](https://docs.monai.io).

Before submitting a pull request, it is recommended to:
- edit the relevant `.rst` files in [`docs/source`](./docs/source) accordingly.
- build html documentation locally
- check the auto-generated documentation (by browsing `./docs/build/html/index.html` with a web browser)
- type `make clean` in `docs/` folder to remove the current build files.

Please type `make help` in `docs/` folder for all supported format options.

#### Automatic code formatting
MONAI provides support of automatic Python code formatting via [a customised GitHub action](https://github.com/Project-MONAI/monai-code-formatter).
This makes the project's Python coding style consistent and reduces maintenance burdens.
Commenting a pull request with `/black` triggers the formatting action based on [`psf/Black`](https://github.com/psf/black) (this is implemented with [`slash command dispatch`](https://github.com/marketplace/actions/slash-command-dispatch)).

Steps for the formatting process:
- After submitting a pull request or push to an existing pull request,
make a comment to the pull request to trigger the formatting action.
The first line of the comment must be `/black` so that it will be interpreted by [the comment parser](https://github.com/marketplace/actions/slash-command-dispatch#how-are-comments-parsed-for-slash-commands).
- [Auto] The GitHub action tries to format all Python files (using [`psf/Black`](https://github.com/psf/black)) in the branch and makes a commit under the name "MONAI bot" if there's code change. The actual formatting action is deployed at [project-monai/monai-code-formatter](https://github.com/Project-MONAI/monai-code-formatter).
- [Auto] After the formatting commit, the GitHub action adds an emoji to the comment that triggered the process.
- Repeat the above steps if necessary.

#### Signing your work
MONAI enforces the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) on all pull requests.
All commit messages should contain the `Signed-off-by` line with an email address. The [GitHub DCO app](https://github.com/apps/dco) is deployed on MONAI. The pull request's status will be `failed` if commits do not contain a valid `Signed-off-by` line.

Git has a `-s` (or `--signoff`) command-line option to append this automatically to your commit message:
```bash
git commit -s -m 'a new commit'
```
The commit message will be:
```
    a new commit

    Signed-off-by: Your Name <yourname@example.org>
```

Full text of the DCO:
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

#### Utility functions
MONAI provides a set of generic utility functions and frequently used routines.
These are located in [``monai/utils``](./monai/utils/) and in the module folders such as [``networks/utils.py``](./monai/networks/).
Users are encouraged to use these common routines to improve code readability and reduce the code maintenance burdens.

Notably,
- ``monai.module.export`` decorator can make the module name shorter when importing,
for example, ``import monai.transforms.Spacing`` is the equivalent of ``monai.transforms.spatial.array.Spacing`` if
``class Spacing`` defined in file `monai/transforms/spatial/array.py` is decorated with ``@export("monai.transforms")``.

For string definition, [f-string](https://www.python.org/dev/peps/pep-0498/) is recommended to use over `%-print` and `format-print` from python 3.6. So please try to use `f-string` if you need to define any string object.


### Submitting pull requests
All code changes to the master branch must be done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).
1. Create a new ticket or take a known ticket from [the issue list][monai issue list].
1. Check if there's already a branch dedicated to the task.
1. If the task has not been taken, [create a new branch in your fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
of the codebase named `[ticket_id]-[task_name]`.
For example, branch name `19-ci-pipeline-setup` corresponds to [issue #19](https://github.com/Project-MONAI/MONAI/issues/19).
Ideally, the new branch should be based on the latest `master` branch.
1. Make changes to the branch ([use detailed commit messages if possible](https://chris.beams.io/posts/git-commit/)).
1. Make sure that new tests cover the changes and the changed codebase [passes all tests locally](#unit-testing).
1. [Create a new pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) from the task branch to the master branch, with detailed descriptions of the purpose of this pull request.
1. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed.
1. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
1. If there are conflicts between the pull request branch and the master branch, pull the changes from the master and resolve the conflicts locally.
1. Reviewer and contributor may have discussions back and forth until all comments addressed.
1. Wait for the pull request to be merged.

## The code reviewing process


### Reviewing pull requests
All code review comments should be specific, constructive, and actionable.
1. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed before reviewing (contact the branch owner if needed).
1. Read carefully the descriptions of the pull request and the files changed, write comments if needed.
1. Make in-line comments to specific code segments, [request for changes](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews) if needed.
1. Review any further code changes until all comments addressed by the contributors.
1. Comment to trigger `/black` and/or `/integration-test` for optional auto code formatting and [integration tests](.github/workflows/integration.yml).
1. Merge the pull request to the master branch.
1. Close the corresponding task ticket on [the issue list][monai issue list].

[github ci]: https://github.com/Project-MONAI/MONAI/actions
[monai issue list]: https://github.com/Project-MONAI/MONAI/issues


## Admin tasks

### Release a new version
- Prepare [a release note](https://github.com/Project-MONAI/MONAI/releases).
- Checkout a new branch `releases/[version number]` locally from the master branch and push to the codebase.
- Create a tag, for example `git tag -a 0.1a -m "version 0.1a"`.
- Push the tag to the codebase, for example `git push origin 0.1a`.
  This step will trigger package building and testing.
  The resultant packages are automatically uploaded to
  [TestPyPI](https://test.pypi.org/project/monai/).  The packages are also available for downloading as
  repository's artifacts (e.g. the file at https://github.com/Project-MONAI/MONAI/actions/runs/66570977).
- Check the release test at [TestPyPI](https://test.pypi.org/project/monai/), download the artifacts when the CI finishes.
- Upload the packages to [PyPI](https://pypi.org/project/monai/).
  This could be done manually by ``twine upload dist/*``, given the artifacts are unzipped to the folder ``dist/``.
- Publish the release note.

Note that the release should be tagged with a [PEP440](https://www.python.org/dev/peps/pep-0440/) compliant
[semantic versioning](https://semver.org/spec/v2.0.0.html) number.

If any error occurs during the release process, first checkout a new branch from the master, make PRs to the master
to fix the bugs via the regular contribution procedure.
Then rollback the release branch and tag:
 - remove any artifacts (website UI) and tag (`git tag -d` and `git push origin -d`).
 - reset the `releases/[version number]` branch to the latest master:
 ```bash
git checkout master
git pull origin master
git checkout releases/[version number]
git reset --hard master
```
Finally, repeat the tagging and TestPyPI uploading process.
