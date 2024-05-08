from dataclasses import dataclass

@dataclass(slots=True)
class AttnConf:
    hidDim: int = 768
    nHead: int = 12
    headDim: int = None
    nKVHead: int = None
    nQPerKvHead: int = None
    scale: float = None

    def __post_init__(self):
        if self.headDim is None:
            self.headDim = self.hidDim // self.nHead
        if self.nKVHead is not None:
            assert self.nHead % self.nKVHead == 0
            self.nQPerKvHead = self.nHead // self.nKVHead
        self.scale = self.headDim ** -0.5