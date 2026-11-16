"""Drive a Matplotlib animation's timer callbacks without a GUI event loop."""


def run_to_completion(animation, figure, *, limit=1000):
    """Start the animation with a draw, then fire its timer until no callback remains.

    This follows the timer contract: a callback returning a false value is
    removed. It returns the number of timer firings that still had a callback.
    """
    figure.canvas.draw()
    timer = animation.event_source
    firings = 0
    while timer.callbacks and firings < limit:
        firings += 1
        for func, args, kwargs in list(timer.callbacks):
            if not func(*args, **kwargs):
                timer.callbacks.remove((func, args, kwargs))
    return firings
