"""Context derivation for continuous, non-overlapping, equal-duration streams."""

from fractions import Fraction

from graphion.profile.numpy_frame import Block, Ctx, SetupError


def stream_ctx(block: Block, port_name: str) -> Ctx:
    """Derive a stream context from a locally configured participating port.

    The first ``stream_ports`` entry is the reference described by ``block.ctx``.
    Equal frame duration is an author contract, not something this helper proves.
    Literal port dtypes take precedence over the context's Sample dtype.
    """
    ports: tuple[str, ...] = getattr(block, "stream_ports", ())
    if port_name not in ports:
        raise SetupError(f"{port_name}: outside the stream-context convention")
    reference = block.port_spec(ports[0])
    spec = block.port_spec(port_name)
    ctx = block.ctx
    if reference.shape != (ctx.channels, ctx.n):
        raise SetupError("reference stream shape must match (ctx.channels, ctx.n)")
    if len(spec.shape) != 2 or min(spec.shape) <= 0:
        raise SetupError("stream ports require positive (channels, samples) shapes")
    channels, n = spec.shape
    rate = ctx.rate if n == ctx.n else ctx.rate * Fraction(n, ctx.n)
    return ctx.replace(channels=channels, n=n, dtype=spec.dtype, rate=rate)
