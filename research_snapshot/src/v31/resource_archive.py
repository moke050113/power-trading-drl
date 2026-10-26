"""Append-only structured records with explicit releases, revisions and SHA audit."""
from pathlib import Path
import json
from .schema import InputRef,canonical,digest,instant,sha
from .resource_schema import ResourceRecord


class ResourceArchive:
    def __init__(self):self._records={};self._audit=()

    @property
    def audit(self):return tuple(json.loads(v) for v in self._audit)

    @property
    def content_sha256(self):return digest(sorted((k,v.content_sha256) for k,v in self._records.items()))

    def append(self,record):self.append_many((record,))

    def append_many(self,records):
        candidate=dict(self._records);events=[]
        for r in tuple(records):
            if type(r) is not ResourceRecord or r.record_id in candidate:raise ValueError('duplicate/invalid resource record')
            oldrows=[x for x in candidate.values() if x.series_id==r.series_id]
            if not oldrows:
                if r.revision!=0 or r.supersedes is not None:raise ValueError('first release must be revision zero')
            else:
                old=max(oldrows,key=lambda x:x.revision)
                if (r.revision!=old.revision+1 or r.supersedes!=old.record_id or r.day!=old.day or r.source!=old.source or
                    instant(r.published_at)<instant(old.published_at) or instant(r.available_at)<instant(old.available_at)):
                    raise ValueError('invalid/backdated resource revision')
                if old.record_type=='history_day':
                    if r.record_type!='history_day':raise ValueError('history cannot become current state')
                else:
                    if r.record_type=='history_day':raise ValueError('current execution cannot become historical demand label')
                    a,b=old.tasks[0],r.tasks[0]
                    if (a.task_id,a.resource,a.pool_id,a.origin)!=(b.task_id,b.resource,b.pool_id,b.origin):raise ValueError('task identity changed')
                    if old.record_type in ('arrived','state'):
                        if r.record_type!='state' or b!=a or instant(r.state.observed_at)<instant(old.state.observed_at):
                            raise ValueError('confirmed demand/time cannot be rewritten')
                        if r.state.remaining_grid_mwh>old.state.remaining_grid_mwh:
                            raise ValueError('completed demand cannot be restored')
            # Same task or historical day cannot be hidden in a second live series.
            for x in candidate.values():
                if x.series_id==r.series_id:continue
                if r.record_type==x.record_type=='history_day' and r.day==x.day and r.source==x.source:
                    raise ValueError('duplicate daily label coverage')
                if r.record_type!='history_day' and x.record_type!='history_day' and r.tasks[0].task_id==x.tasks[0].task_id:
                    raise ValueError('task lifecycle must share one series')
            candidate[r.record_id]=r
            events.append(canonical({'operation':'resource_append','at':r.available_at,'content_sha256':r.content_sha256}))
        self._records=candidate;self._audit+=tuple(events)

    def snapshot(self,as_of,*,source=None):
        at=instant(as_of);selected={}
        for r in self._records.values():
            if instant(r.available_at)<=at and (source is None or r.source==source):
                old=selected.get(r.series_id)
                if old is None or r.revision>old.revision:selected[r.series_id]=r
        rows=tuple(sorted(selected.values(),key=lambda r:(r.available_at,r.series_id,r.revision)))
        self._audit+=(canonical({'operation':'resource_replay','at':at.isoformat(),'content_sha256':[r.content_sha256 for r in rows]}),)
        return tuple(ResourceRecord.from_dict(r.to_dict()) for r in rows)

    def resolve(self,ref,as_of):
        if type(ref) is not InputRef:raise ValueError('InputRef required')
        r=self._records.get(ref.record_id)
        if r is None or r.content_sha256!=ref.sha256 or instant(r.available_at)>instant(as_of):raise ValueError('unavailable/mismatched resource lineage')
        return ResourceRecord.from_dict(r.to_dict())

    def save(self,path):
        payload={'schema_version':1,'records':[{'record':r.to_dict(),'sha256':r.content_sha256} for r in sorted(self._records.values(),key=lambda r:(r.series_id,r.revision))], 'audit':self.audit}
        with Path(path).open('x',encoding='utf-8') as f:f.write(canonical({'payload':payload,'sha256':digest(payload)})+'\n')
        return digest(payload)

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'));p=data['payload']
        if data['sha256']!=digest(p) or p['schema_version']!=1:raise ValueError('resource envelope corrupt')
        records=[]
        for row in p['records']:
            r=ResourceRecord.from_dict(row['record'])
            if r.content_sha256!=row['sha256']:raise ValueError('resource content SHA mismatch')
            records.append(r)
        obj=cls();obj.append_many(records)
        for e in p['audit']:
            if set(e)!={'operation','at','content_sha256'}:raise ValueError('invalid audit keys')
            if e['operation'] not in ('resource_append','resource_replay'):raise ValueError('invalid resource audit')
            instant(e['at'])
            hashes=(e['content_sha256'],) if isinstance(e['content_sha256'],str) else e['content_sha256']
            if not isinstance(hashes,(list,tuple)):raise ValueError('invalid audit hash payload')
            for h in hashes:sha(h)
        obj._audit=tuple(canonical(e) for e in p['audit']);return obj
