from dataclasses import dataclass

@dataclass(slots=True)
class AttnBlocksConf:
    hidDim: int
    nHead: int
    # layerNum: int
    headDim: int = None
    nKVHead: int = None
    nQPerKvHead: int = None
    scale: float = None

    def __post_init__(self):
        if self.hidDim <= 0:
            raise ValueError(f'Invalid hidDim: {self.hidDim}')
        if self.nHead <= 0:
            raise ValueError(f'Invalid nHead: {self.nHead}')
        # if self.layerNum <= 0:
        #     raise ValueError(f'Invalid layerNum: {self.layerNum}')
        if self.headDim is None:
            self.headDim = self.hidDim // self.nHead
        if self.nKVHead is not None:
            if self.nKVHead <= 0:
                raise ValueError(f'Invalid nKVHead: {self.nKVHead}')
            if self.nHead % self.nKVHead != 0:
                raise ValueError(f"nKVHead can't divide nHead: {self.nHead}, {self.nKVHead}")
            if self.nHead != self.nKVHead:
                self.nQPerKvHead = self.nHead // self.nKVHead
        else:
            self.nKVHead = self.nHead
        self.scale = self.headDim ** -0.5
