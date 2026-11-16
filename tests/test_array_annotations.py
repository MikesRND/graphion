"""Runtime compatibility of the two supported Array annotations."""

from typing import get_args, get_origin

import numpy as np
import pytest
from numpy.typing import NDArray

from graphion import Array, Block, Ctx, DeclarationError, In, Out, Param, Sample, State


@pytest.mark.parametrize("declaration", [In, Out, State])
@pytest.mark.parametrize("sample_dtype", [np.uint8, np.float32, np.complex128])
@pytest.mark.parametrize("selector", [Sample, np.int16])
def test_array_spellings_preserve_dtype_and_description(declaration, sample_dtype, selector):
    kwargs = {"default": 0} if declaration is State else {}
    expected = np.dtype(sample_dtype if selector is Sample else selector)
    snapshots = []
    for annotation in (Array, NDArray[np.generic]):
        model = type(
            "Model",
            (Block,),
            {"__annotations__": {"value": annotation}, "value": declaration(4, dtype=selector, **kwargs)},
        )()
        before = model.describe()
        model.setup(Ctx(rate=1, n=4, channels=1, dtype=sample_dtype))
        if declaration is In:
            model.value = np.ones(4, dtype=expected)
        assert model.value.dtype == expected
        assert model.schema["value"].dtype is selector
        snapshots.append((before, model.describe()))
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("spelling", [Array, NDArray[np.generic]])
def test_port_annotation_matching_never_compares_author_objects(spelling):
    class EqualityForbidden:
        def __eq__(self, other):
            raise AssertionError("validation compared annotation argument")

        def __repr__(self):
            raise AssertionError("validation represented annotation argument")

    args = get_args(spelling)
    unsupported = get_origin(spelling)[(*args[:-1], EqualityForbidden())]
    with pytest.raises(DeclarationError, match="ports and array State require the Array annotation"):
        type("Invalid", (Block,), {"__annotations__": {"value": unsupported}, "value": Out(4)})


def test_description_does_not_compare_legacy_alias_arguments():
    class EqualityForbidden:
        def __eq__(self, other):
            raise AssertionError("description compared annotation argument")

        def __repr__(self):
            raise AssertionError("description represented annotation argument")

    spelling = NDArray[np.generic]
    origin, args = get_origin(spelling), get_args(spelling)
    model = type(
        "Model",
        (Block,),
        {
            "__annotations__": {
                "equivalent": origin[args],
                "unusual": origin[(*args[:-1], EqualityForbidden())],
            },
            "equivalent": Param(default=np.ones(2)),
            "unusual": Param(default=np.ones(2)),
        },
    )()
    fields = {field["name"]: field for field in model.describe()["blocks"][0]["declarations"]}
    assert fields["equivalent"]["value_type"] == {"kind": "array"}
    assert fields["unusual"]["value_type"] == {"kind": "opaque", "type": "types.GenericAlias"}
