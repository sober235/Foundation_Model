"""Bounded-cache, role-isolated, stateless observations for qualified training."""
from collections import OrderedDict

import numpy as np
import torch

from anatobind.aur.ssl.unified_data import (load_unified_ssl_image,make_unified_ssl_crop,collate_unified_ssl)
from anatobind.aur.unified_data import load_unified_volume,make_unified_crop,collate_unified
from anatobind.aur.unified_run import PatientBalancedSampler,validate_run_config


class QualifiedViews:
    def __init__(self,rows,stage,cfg,*,augment=True,validation=False,cache_volumes=2):
        self.rows,self.stage,self.cfg=list(rows),stage,validate_run_config(cfg)
        if stage not in ('I','II','III') or not self.rows or type(cache_volumes) is not int or cache_volumes<0:
            raise ValueError('invalid qualified observation configuration')
        self.augment,self.validation=bool(augment),bool(validation)
        role=('ssl_validation' if validation else 'train_ssl') if stage=='I' else ('development_validation' if validation else 'train_supervised')
        split='val' if validation else 'train'
        for row in self.rows:
            if row.get('role')!=role or row.get('split')!=split:
                raise ValueError('observation role conflicts with requested partition')
            if stage=='I' and any(k in row for k in ('anatomy','lesion','u_values','a_ignore_values','a_supervised','u_supervised','r_supervised')):
                raise ValueError('SSL role must be image-only')
        self.seed=self.cfg['seed']+(100000 if validation else 0)
        self.sampler=PatientBalancedSampler(self.rows,self.seed,self.cfg['hcp_cap'])
        groups=sorted({(r['source'],str(r['patient'])) for r in self.rows})
        self.patient_codes={g:i for i,g in enumerate(groups)}
        self.cache_volumes,self.cache=cache_volumes,OrderedDict()

    def _volume(self,index):
        if index in self.cache:
            value=self.cache.pop(index)
            self.cache[index]=value
            return value
        row=self.rows[index]
        value=load_unified_ssl_image(row) if self.stage=='I' else load_unified_volume(row)
        if self.cache_volumes:
            self.cache[index]=value
            while len(self.cache)>self.cache_volumes:
                self.cache.popitem(last=False)
        return value

    def __call__(self,mode,draw):
        indices=self.sampler.indices(mode,draw,self.cfg['batch_size'])
        observations=[]
        for position,index in enumerate(indices):
            vol=self._volume(index)
            rng=np.random.default_rng([self.seed,int(mode=='3d'),draw,position])
            if self.stage=='I':
                item=make_unified_ssl_crop(vol,rng,mode,self.cfg['crops'][mode],do_rotate=self.augment)
                group=(self.rows[index]['source'],str(self.rows[index]['patient']))
                item['patient']=torch.tensor(self.patient_codes[group],dtype=torch.long)
            else:
                item=make_unified_crop(vol,rng,mode,self.cfg['crops'][mode],do_augment=self.augment,
                    lesion_centred=bool(rng.random()<.5),n_events=self.cfg['model']['n_events'],
                    planar_relation_policy=self.cfg['planar_relation_policy'] if mode=='2d' else None)
            observations.append([item])
        batch=(collate_unified_ssl if self.stage=='I' else collate_unified)(observations)
        batch['source_labels']=[self.rows[i]['source'] for i in indices]
        return batch
