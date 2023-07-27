import collections
import typing

class TrainConfig(typing.NamedTuple):

    input_size: int
    batch_size: int
    hidden: int
    lr: float
    dropout: float
    logname: str
    outputLength: int
    inputLength: int
    subName: str
    debug: bool = False
    dataAug: bool = False
    prob0: float = 0.1
    prob1: float = 0.1
    prob2: float = 0.6
    prob3: float = 0.6


    @classmethod
    def from_dict(cls,dikt):
        input_size = dikt['input_size']
        batch_size = dikt['batch_size']
        hidden = dikt['hidden']
        lr = dikt['lr']
        dropout = dikt['dropout']
        logname = dikt["logname"]
        outputLength = dikt["outputLength"]
        inputLength= dikt["inputLength"]
        subName = dikt['subName']
        dataAug = dikt['dataAug']
        prob0 = dikt['prob0']
        prob1 = dikt['prob1']
        prob2 = dikt['prob2']
        prob3 = dikt['prob3']

        return cls(input_size,batch_size,hidden,lr,dropout,
                   logname,outputLength,inputLength,subName,dataAug=dataAug, prob0=prob0, prob1=prob1, prob2=prob2, prob3=prob3)


Basic = collections.namedtuple("NILM", ["model", "model_opt"])
