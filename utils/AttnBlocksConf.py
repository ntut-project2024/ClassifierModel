from dataclasses import dataclass

@dataclass(slots=True)
class AttnBlocksConf:
    hidDim: int
    nHead: int
    layerNum: int

    def __post_init__(self):
        if self.hidDim <= 0:
            raise ValueError(f'Invalid hidDim: {self.hidDim}')
        if self.nHead <= 0:
            raise ValueError(f'Invalid nHead: {self.nHead}')
        if self.layerNum <= 0:
            raise ValueError(f'Invalid layerNum: {self.layerNum}')