from __future__ import annotations
import os, time, json
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import requests

def load_env(env_path: str | Path = '.env') -> None:
    candidate_paths = [
        Path(env_path),
        Path(__file__).resolve().parent.parent.parent / '.env',
        Path(__file__).resolve().parent.parent.parent.parent / 'agentpoker_final 2' / '.env',
    ]
    for p in candidate_paths:
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

load_env()

class APIError(RuntimeError):
    def __init__(self,status:int,code:str,message:str,retry_after:float|None=None):
        super().__init__(f"HTTP {status} {code}: {message}"); self.status=status; self.code=code; self.message=message; self.retry_after=retry_after

@dataclass
class Config:
    base_url: str = os.getenv('AGENTPOKER_APP','https://poker.bang.sohu.com').rstrip('/')
    key: str|None = os.getenv('AGENTPOKER_KEY')
    timeout: float = float(os.getenv('AGENTPOKER_TIMEOUT','10'))
    competition_id: str|None = os.getenv('AGENTPOKER_COMPETITION_ID')

class AgentPokerClient:
    def __init__(self,cfg:Config|None=None):
        self.cfg=cfg or Config(); self.s=requests.Session()
        # Bypass the OS/env proxy entirely -- this client only ever talks to the
        # configured Agent Poker API host, and a stale system proxy (e.g. a VPN
        # tool pointing at a dead local port) should never be able to break it.
        self.s.trust_env = False
        self.s.proxies = {}
        self.agent_id=None
    def _headers(self):
        h={'Content-Type':'application/json','Accept':'application/json'}
        if self.cfg.key: h['Authorization']=f'Bearer {self.cfg.key}'
        return h
    def _request(self,method,path,body=None,auth=False,accept=None):
        headers=self._headers() if auth else {'Accept': accept or 'application/json','Content-Type':'application/json'}
        if accept: headers['Accept']=accept
        try:r=self.s.request(method,self.cfg.base_url+path,json=body,headers=headers,timeout=self.cfg.timeout)
        except requests.RequestException as e: raise
        if r.status_code>=400:
            try:data=r.json(); err=data.get('error',{})
            except Exception:data={}; err={}
            default_code = 'temporarily_unavailable' if r.status_code in (502, 503, 504) else f'http_{r.status_code}'
            raise APIError(r.status_code,err.get('code', default_code),err.get('message',r.text[:200]),float(r.headers.get('Retry-After','0') or 0) or None)
        if not r.content: return {}
        try:return r.json()
        except Exception: raise APIError(r.status_code,'invalid_response','non-JSON success response')
    def discover(self,status='active'):
        return self._request('GET',f'/api/competitions?status={status}')
    def join(self,cid): return self._request('POST','/api/competitions/join',{'competitionId':cid},True)
    def observe(self,cid,tid=None):
        body={'competitionId':cid};
        if tid: body['tableId']=tid
        return self._request('POST','/api/competitions/observe',body,True)
    def action(self,body): return self._request('POST','/api/competitions/action',body,True)
    def leave(self,body): return self._request('POST','/api/competitions/leave',body,True)
    def standings(self,cid): return self._request('GET',f'/api/competitions/{cid}/standings')
    def hands(self,cid,**params):
        q='&'.join(f'{k}={requests.utils.quote(str(v))}' for k,v in params.items() if v is not None)
        return self._request('GET',f'/api/competitions/{cid}/hands'+(('?'+q) if q else ''))
    def exchange_code(self, code: str):
        return self._request('POST', '/api/cli/exchange', {'code': code})
    def get_strategy(self,agent_id): return self._request('GET',f'/api/agents/{agent_id}/strategy',auth=True)
    def put_strategy(self,agent_id,text): return self._request('PUT',f'/api/agents/{agent_id}/strategy',{'strategy':text},True)


def retry_same(fn, *, max_attempts=None, retry_codes=('temporarily_unavailable', 'service_unavailable', 'gateway_timeout'), backoff=1.0):
    n=0
    while True:
        try:
            return fn()
        except requests.RequestException as e:
            n += 1
            if max_attempts and n >= max_attempts:
                raise
            time.sleep(min(30.0, backoff * (2 ** (n - 1))))
        except APIError as e:
            n += 1
            if e.code not in retry_codes and e.status not in (502, 503, 504):
                raise
            if max_attempts and n >= max_attempts:
                raise
            time.sleep(e.retry_after if e.retry_after is not None else min(30.0, backoff * (2 ** (n - 1))))

