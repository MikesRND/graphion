from atom.api import Atom
import numpy as np


class Tensor(Atom):

    # Shape of the tensor
    shape = property(lambda self: self._get_shape())

    # Shape of the tensor
    dtype = property(lambda self: self._get_dtype())

    # Shape of the tensor
    value = property(lambda self: self._memory)

    # Private tensor storage
    _memory: np.ndarray

    def __init__(self, init: np.ndarray):
        """
        Initialize the tensor

        Parameters
        ----------
        init: np.ndarray
            Initial value of the tensor.

        """
        self._memory = init

    def _get_shape(self):
        return self._memory.shape

    def _get_dtype(self):
        return self._memory.dtype


class Block(Atom):
    """Callable interface of a processing block"""

    # Instance name
    name: str

    # Initialization status
    isInitialized = property(lambda self: self._initialized)

    # Private storage
    _initialized: bool = False

    def initialize(self):
        """Initialize the block"""
        if not self._initialized:
            self._setup()
        self._initialized = True

    def run(self):
        """Run the block"""
        if not self._initialized:
            raise RuntimeError("Block must be initialized before running")
        return self._run()

    def terminate(self):
        """Terminate the block"""
        if self.isInitialized:
            self._terminate()
            self._initialized = False

    # Subclass implementation methods

    def _setup(self):
        pass

    def _run(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _terminate(self):
        pass


if __name__ == "__main__":

    a1 = np.random.randn(2, 3)
    print(a1)
    print(a1.shape)
    print(a1.dtype)
    t1 = Tensor(a1)
    print(t1.shape)
    b1 = Block()
    print(b1)
