"""Public authoring and library imports compose in fresh Python processes."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

import graphion
from graphion.dsp.blocks import FrameDelay, Gain
from graphion.profile import numpy_frame as profile


class PublicImportTests(unittest.TestCase):
    def test_root_and_profile_export_identical_authoring_objects(self) -> None:
        self.assertEqual(set(graphion.__all__), set(profile.__all__))
        for name in graphion.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(graphion, name), getattr(profile, name))

    def test_library_blocks_compose_with_both_authoring_imports(self) -> None:
        class DelayedGain(graphion.Block):
            x: profile.Array = profile.In(graphion.C, graphion.N)
            y: graphion.Array = graphion.Out(profile.C, profile.N)

            def build(self) -> None:
                self.gain = self.child("gain", Gain(gain=2.0))
                self.delay = self.child("delay", FrameDelay())
                self.connect(self, "x", self.gain, "x")
                self.connect(self.gain, "y", self.delay, "x")
                self.connect(self.delay, "y", self, "y")

            def process(self) -> None:
                self.gain.run()
                self.delay.run()

        model = DelayedGain()
        model.setup(profile.Ctx(rate=48_000, n=4, channels=1))
        source = np.arange(4.0).reshape(1, 4)
        model.x = source
        first = model.run()
        np.testing.assert_array_equal(first, np.zeros_like(source))
        self.assertIs(model.run(), first)
        np.testing.assert_array_equal(first, source * 2)
        self.assertFalse(first.flags.writeable)
        self.assertIsInstance(model.delay, profile.Block)
        self.assertEqual(model.describe()["profile"], "python.numpy.frame")
        model.reset()
        np.testing.assert_array_equal(model.run(), np.zeros_like(source))

    def test_fresh_import_orders_and_standard_library_profiler(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for entry in ("graphion", "graphion.profile.numpy_frame", "graphion.dsp.blocks"):
            for profiler_first in (False, True):
                with self.subTest(entry=entry, profiler_first=profiler_first):
                    imports = [f"import {entry}", "import cProfile"]
                    if profiler_first:
                        imports.reverse()
                    code = (
                        "\n".join(imports)
                        + """
from graphion import Block, Ctx
from graphion.profile import numpy_frame
from graphion.dsp.blocks import Gain
import numpy as np
assert Block is numpy_frame.Block
gain = Gain(gain=2.0)
gain.setup(Ctx(rate=1, n=2, channels=1))
gain.x = np.ones((1, 2))
np.testing.assert_array_equal(cProfile.Profile().runcall(gain.run), [[2.0, 2.0]])
assert gain.describe()["profile"] == "python.numpy.frame"
"""
                    )
                    result = subprocess.run(
                        [sys.executable, "-c", code],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
