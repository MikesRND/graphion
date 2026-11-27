"""The wire form of an observation: a JSON header, then every frame's raw bytes.

    uint32 little-endian   length of the header, padding included
    header                 UTF-8 JSON, padded with spaces
    payload                each array's bytes, little-endian, C order

The header's ``arrays`` maps an array name to ``[offset, nbytes]`` within the
payload, and each frame's ``key`` names its array. The payload starts on an
8-byte boundary and so does every array, so a client can view the bytes as
typed arrays without copying them.
"""

import json
import struct

import numpy as np

CONTENT_TYPE = "application/vnd.graphion.observation"


def _pad(length):
    return -length % 8


def parts(observation):
    """Return one observation as a list of buffers to send in order.

    The frames are views of the snapshot's own arrays, so sending them copies
    nothing in this process however many clients are given the same snapshot.
    """
    chunks, index, offset = [], {}, 0
    for name, array in observation.arrays.items():
        index[name] = [offset, array.nbytes]
        chunks.append(array.view(np.uint8).data)
        if _pad(array.nbytes):
            chunks.append(b"\0" * _pad(array.nbytes))
        offset += array.nbytes + _pad(array.nbytes)
    head = json.dumps({**observation.header, "arrays": index}, allow_nan=False, default=str).encode()
    head += b" " * _pad(4 + len(head))
    return [struct.pack("<I", len(head)) + head, *chunks]


def encode(observation):
    """Return the bytes of one observation."""
    return b"".join(parts(observation))


def decode(body):
    """Return (header, arrays by name) from the bytes of one observation."""
    (length,) = struct.unpack_from("<I", body)
    header = json.loads(body[4 : 4 + length])
    start = 4 + length
    arrays = {}
    for probe in header["frames"].values():
        name = probe.get("key")
        if name is None or name in arrays:
            continue
        offset, nbytes = header["arrays"][name]
        arrays[name] = np.frombuffer(body, dtype=probe["dtype"], count=probe["count"], offset=start + offset)
        assert arrays[name].nbytes == nbytes
    return header, arrays
