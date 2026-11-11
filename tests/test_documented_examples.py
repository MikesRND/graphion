import re
import unittest
from pathlib import Path

import numpy as np

from graphion import Ctx, SetupError
from tests.test_declarations import module_with

DOCS = Path(__file__).resolve().parents[1] / "docs"


def code(path, heading=None):
    text = (DOCS / path).read_text()
    if heading:
        text = text.split(heading + "\n", 1)[1]
    return re.search(r"```python\n(.*?)```", text, re.S)[1]


def reference(heading):
    return code("reference/examples.md", "## Basic authoring and use") + "\n" + code("reference/examples.md", heading)


class DocumentedExampleTests(unittest.TestCase):
    def test_quickstart_and_retained_snapshot(self):
        with module_with(code("quickstart.md")) as example:
            np.testing.assert_array_equal(example.result, [[1.0, 2.0, 3.0, 4.0]])
            self.assertFalse(example.result.flags.writeable)
            example.amp.gain = 2
            example.amp.run()
            np.testing.assert_array_equal(example.result, [[4.0, 8.0, 12.0, 16.0]])
            example.amp.reset()
            np.testing.assert_array_equal(example.saved, [[1.0, 2.0, 3.0, 4.0]])

    def test_guide_managing_data(self):
        source = code("quickstart.md") + "\n" + code("guide/managing-data.md")
        with module_with(source) as example:
            np.testing.assert_array_equal(example.result, [[2.0, 2.0, 2.0, 2.0]])
            np.testing.assert_array_equal(example.snapshot, [[1.0, 1.0, 1.0, 1.0]])
            self.assertIs(example.result, example.amp.run())

    def test_reference_host_access(self):
        source = reference("### A stateful filter") + "\n" + code("reference/examples.md", "### A connected composite")
        source += "\nstream = [np.ones((2, 4096), dtype=np.complex64)]\nreceived = []\nsink = lambda value: received.append(value.copy())\n"
        source += code("reference/examples.md", "### Host access")
        with module_with(source) as example:
            self.assertTrue(np.shares_memory(example.latest_input, example.stream[-1]))
            self.assertFalse(example.latest_input.flags.writeable)
            self.assertIs(example.latest_output, example.fx.y)
            self.assertIs(example.intermediate, example.fx.a.y)
            np.testing.assert_array_equal(example.snapshot, example.received[-1])
            self.assertFalse(np.shares_memory(example.snapshot, example.latest_output))

    def test_reference_gain(self):
        with module_with(code("reference/examples.md", "## Basic authoring and use")) as example:
            np.testing.assert_array_equal(example.saved, 0.5)
            np.testing.assert_array_equal(example.y, 1)
            self.assertEqual(example.y.dtype, np.complex64)
            self.assertIs(example.y, example.amp.run())

    def test_guide_and_reference_frame_delay(self):
        for source in (code("guide/writing-blocks.md"), reference("### A frame delay")):
            with self.subTest(source=source[:30]), module_with(source) as example:
                delay = example.FrameDelay()
                delay.setup(Ctx(rate=1, n=4, channels=1))
                source_array = np.full((1, 4), 2.0)
                delay.x = source_array
                np.testing.assert_array_equal(delay.run(), 0)
                source_array.fill(7)
                np.testing.assert_array_equal(delay.run(), 2)
                held = delay.held
                delay.reset()
                self.assertIs(delay.held, held)
                np.testing.assert_array_equal(delay.run(), 0)

    def test_reference_one_pole_lowpass(self):
        with module_with(reference("### A stateful filter")) as example:
            block = example.OnePoleLowpass(pole=0.5)
            block.setup(Ctx(rate=1, n=4, channels=1))
            block.x = np.ones((1, 4))
            np.testing.assert_allclose(block.run(), [[0.5, 0.75, 0.875, 0.9375]])
            block.reset()
            np.testing.assert_allclose(block.run(), [[0.5, 0.75, 0.875, 0.9375]])

    def test_guide_two_gains(self):
        source = code("quickstart.md") + "\n" + code("guide/composing-models.md")
        with module_with(source) as example:
            block = example.TwoGains()
            block.setup(Ctx(rate=8, n=4, channels=1))
            frame = np.array([[2.0, 4.0, 6.0, 8.0]])
            block.x = frame
            result = block.run()
            np.testing.assert_array_equal(result, [[1.0, 2.0, 3.0, 4.0]])
            self.assertIs(block.a.x, block.x)
            self.assertIs(result, block.y)
            self.assertIs(result, block.b.run())
            self.assertFalse(result.flags.writeable)

            replacement = np.array([[4.0, 8.0, 12.0, 16.0]])
            previous_input = block.x
            block.x = replacement
            self.assertIsNot(block.x, previous_input)
            self.assertIs(block.a.x, block.x)
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, [[2.0, 4.0, 6.0, 8.0]])
            block.reset()
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, [[2.0, 4.0, 6.0, 8.0]])

    def test_guide_build_walkthrough(self):
        source = code("quickstart.md") + "\n" + code("guide/composing-models.md")
        source += "\n" + code("guide/composing-models.md", "## Build, inspect, then set up")
        with module_with(source) as example:
            diagram = example.diagram
            self.assertEqual(diagram["topology"], "complete")
            self.assertEqual(diagram["schema_version"], 2)
            for block in diagram["blocks"]:
                self.assertIsNone(block["context"])
                self.assertIsNone(block["storage"])
            fields = {field["name"]: field for field in diagram["blocks"][0]["declarations"]}
            self.assertEqual(
                fields["x"]["shape"],
                [{"kind": "symbol", "name": "C"}, {"kind": "symbol", "name": "N"}],
            )
            np.testing.assert_array_equal(example.result, [[1.0, 2.0, 3.0, 4.0]])
            automatic = example.auto.describe()
            self.assertEqual(automatic["topology"], "complete")
            self.assertTrue(all(block["storage"] is not None for block in automatic["blocks"]))

    def test_reference_chain(self):
        source = reference("### A stateful filter") + "\n" + code("reference/examples.md", "### A connected composite")
        with module_with(source) as example:
            block = example.Chain(first_pole=0.5, second_pole=0.5)
            block.setup(Ctx(rate=8, n=4, channels=1))
            block.x = np.ones((1, 4))
            result = block.run()
            np.testing.assert_array_equal(result, [[0.25, 0.5, 0.6875, 0.8125]])
            self.assertIs(block.a.x, block.x)
            self.assertIs(result, block.y)
            self.assertTrue(np.shares_memory(result, block.b.y))
            self.assertFalse(result.flags.writeable)
            first_memory, second_memory = block.a.memory, block.b.memory

            block.x = np.zeros((1, 4))
            self.assertIs(block.a.x, block.x)
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, [[0.640625, 0.4375, 0.27734375, 0.16796875]])
            block.reset()
            self.assertIs(block.a.memory, first_memory)
            self.assertIs(block.b.memory, second_memory)
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, 0)
            block.x = np.ones((1, 4))
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, [[0.25, 0.5, 0.6875, 0.8125]])

    def test_reference_feedback(self):
        with module_with(reference("### Feedback with explicit State")) as example:
            block = example.Feedback()
            block.setup(Ctx(rate=1, n=4, channels=1))
            impulse = np.ones((1, 4))
            block.x = impulse
            np.testing.assert_array_equal(block.run(), 1)
            impulse.fill(0)
            for weight in (0.5, 0.25, 0.125):
                np.testing.assert_array_equal(block.run(), weight)
            self.assertFalse(np.shares_memory(block.y, block.previous))

    def test_reference_split_feedback(self):
        source = reference("### A frame delay") + "\n" + code("reference/examples.md", "## Feedback across blocks")
        with module_with(source) as example:
            block = example.FeedbackLoop()
            block.setup(Ctx(rate=1, n=4, channels=1))
            frame = np.ones((1, 4))
            block.x = frame
            for _ in range(2):
                frame.fill(1)
                np.testing.assert_array_equal(block.run(), 1)
                frame.fill(0)
                for value in (0.5, 0.25, 0.125):
                    np.testing.assert_array_equal(block.run(), value)
                block.reset()

    def test_reference_downsample(self):
        with module_with(reference("### Rate conversion and a child context")) as example:
            for factor in (0, -1, True, 1.5, 3):
                with self.subTest(factor=factor), self.assertRaises(SetupError):
                    example.Downsample(factor=factor).setup(Ctx(rate=8, n=8, channels=1))
            block = example.Downsample(factor=2)
            block.setup(Ctx(rate=8, n=8, channels=1))
            block.x = np.arange(8, dtype=np.float64).reshape(1, 8)
            np.testing.assert_array_equal(block.run(), [[0, 2, 4, 6]])

    def test_reference_reduced_rate(self):
        with module_with(reference("### Rate conversion and a child context")) as example:
            block = example.ReducedRate(factor=2)
            ctx = Ctx(rate=8, n=8, channels=2, dtype=np.float32)
            block.setup(ctx)
            self.assertIs(block.down.ctx, ctx)
            self.assertEqual(block.post.ctx.n, 4)
            self.assertEqual(block.post.ctx.rate, 4)
            self.assertEqual(block.post.ctx.channels, 2)
            self.assertEqual(block.post.ctx.dtype, np.dtype(np.float32))
            frame = np.arange(16, dtype=np.float32).reshape(2, 8)
            block.x = frame
            result = block.run()
            np.testing.assert_array_equal(result, [[0, 1, 2, 3], [4, 5, 6, 7]])
            self.assertIs(block.down.x, block.x)
            self.assertIs(result, block.y)
            self.assertIs(result, block.post.run())
            self.assertFalse(result.flags.writeable)

            block.x = np.full((2, 8), 6, dtype=np.float32)
            self.assertIs(block.down.x, block.x)
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, 3)
            block.reset()
            self.assertIs(block.run(), result)
            np.testing.assert_array_equal(result, 3)

    def test_reference_reduced_rate_preserves_fraction_arithmetic(self):
        from fractions import Fraction

        with module_with(reference("### Rate conversion and a child context")) as example:
            for rate, expected in (
                (Fraction(48_000_000, 1001), Fraction(12_000_000, 1001)),
                (48_000_000, 12_000_000.0),
            ):
                with self.subTest(rate_type=type(rate).__name__):
                    block = example.ReducedRate(factor=4)
                    ctx = Ctx(rate=rate, n=8, channels=2)
                    block.setup(ctx)
                    self.assertIs(block.down.ctx, ctx)
                    self.assertEqual(ctx.rate, rate)
                    self.assertIs(type(ctx.rate), type(rate))
                    self.assertIs(type(block.post.ctx.rate), type(expected))
                    self.assertEqual(block.post.ctx.rate, expected)
                    self.assertEqual(block.post.ctx.n, 2)
                    frame = np.arange(16, dtype=np.float64).reshape(2, 8)
                    block.x = frame
                    result = block.run()
                    np.testing.assert_array_equal(result, [[0, 2], [4, 6]])
                    self.assertIs(result, block.y)
                    self.assertFalse(result.flags.writeable)


if __name__ == "__main__":
    unittest.main()
