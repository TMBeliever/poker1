from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Standing:
    agent_id:str; hands:int; net_bb:float; bb100:float|None; rank:int|None=None

def bb100(net_bb:float,hands:int)->float|None:
    return None if hands<=0 else (net_bb/hands)*100.0

def rank_standings(rows:list[Standing], tie_round:dict[str,float]|None=None)->list[Standing]:
    tie_round=tie_round or {}
    rows=sorted(rows,key=lambda x:(-(x.bb100 if x.bb100 is not None else float('-inf')),-x.hands,-tie_round.get(x.agent_id,float('-inf')),x.agent_id))
    for i,x in enumerate(rows,1): x.rank=i
    return rows

def percentile_rank(rank:int,n:int)->float:
    return 1.0-(rank-1)/max(1,n-1)
