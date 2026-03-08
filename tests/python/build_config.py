# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import unittest

from build_helpers import resolve_cuda_build_enabled
from tests.python.common import DEFAULT_PLY_PATH


class TestBuildConfig(unittest.TestCase):
    def test_resolve_cuda_build_enabled(self):
        self.assertFalse(
            resolve_cuda_build_enabled("darwin", False, False, "12.8", "/usr/local/cuda")
        )
        self.assertFalse(
            resolve_cuda_build_enabled("linux", True, False, "12.8", "/usr/local/cuda")
        )
        self.assertFalse(resolve_cuda_build_enabled("linux", False, False, None, None))
        self.assertTrue(
            resolve_cuda_build_enabled("linux", False, True, "12.8", "/usr/local/cuda")
        )

    def test_local_point_cloud_fixture_exists(self):
        self.assertTrue(DEFAULT_PLY_PATH.is_file(), DEFAULT_PLY_PATH)
