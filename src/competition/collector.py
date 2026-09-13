from __future__ import annotations
from pathlib import Path
import json, re, time
SECRET=re.compile(r'sk_[A-Za-z0-9._~-]+')

def redact(obj):
    if isinstance(obj,dict): return {k:('[REDACTED]' if k.lower()=='authorization' else redact(v)) for k,v in obj.items()}
    if isinstance(obj,list): return [redact(x) for x in obj]
    if isinstance(obj,str): return SECRET.sub('[REDACTED]',obj)
    return obj

class JSONLCollector:
    def __init__(self,path='data/raw/events.jsonl'):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def write(self,event):
        row={'ts':time.time(),**redact(event)}
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')

class ReplayBuilder:
    def __init__(self,raw_path): self.raw=Path(raw_path)
    def hands(self):
        seen={}
        for line in self.raw.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            e=json.loads(line); o=e.get('data',{}); table=o.get('table') if isinstance(o,dict) else None
            hand=(table or {}).get('hand') if table else None
            if not hand or not hand.get('id'): continue
            seen[hand['id']]=o
        return list(seen.values())
    def export(self,out='data/processed/hands.jsonl'):
        p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
        hs=self.hands()
        with p.open('w',encoding='utf-8') as f:
            for h in hs: f.write(json.dumps(h,ensure_ascii=False,separators=(',',':'))+'\n')
        return len(hs)
