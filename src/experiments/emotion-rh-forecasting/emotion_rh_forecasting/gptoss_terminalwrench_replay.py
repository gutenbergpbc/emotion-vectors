"""Pinned MXFP4 GPT-OSS emotion projections with sentence-end causal replay."""
from collections import defaultdict
import hashlib
import importlib.metadata
import io
import time
import numpy as np
from shared.io import atomic,digest,require,sha
from .terminalwrench_replay import sentence_cuts,chunk_ranges

class GPTOSSReplay:
    def __init__(self,bundle,spec):
        import torch,kernels
        from transformers import AutoModelForCausalLM,AutoTokenizer,Mxfp4Config
        self.torch,self.bundle,self.spec=torch,bundle,spec
        original=kernels.get_kernel
        def pinned(repo_id,revision=None,**kwargs):
            require(repo_id=='kernels-community/triton_kernels','Unexpected kernel repository')
            return original(repo_id,revision=spec['kernel_revision'],**kwargs)
        kernels.get_kernel=pinned
        try:
            self.model=AutoModelForCausalLM.from_pretrained(bundle['model']['name'],revision=bundle['model']['revision'],torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='eager',quantization_config=Mxfp4Config(dequantize=False)).eval()
        finally:kernels.get_kernel=original
        self.tokenizer=AutoTokenizer.from_pretrained(bundle['model']['name'],revision=bundle['model']['revision'])
        require(self.model.is_quantized and len(self.model.model.layers)==36 and bundle['width']==2880,'Wrong GPT-OSS architecture')
        self.groups=defaultdict(list);self.captured={};self.handles=[]
        for j,p in enumerate(bundle['probes']):self.groups[p['layer']].append(j)
        self.vectors={l:torch.tensor(np.array([bundle['probes'][j]['vector'] for j in js]).T,dtype=torch.float32,device='cuda') for l,js in self.groups.items()}
        for l in self.groups:
            def hook(module,args,output,layer=l):
                h=output[0] if isinstance(output,tuple) else output
                self.captured[layer]=(h[0].float()@self.vectors[layer]).cpu().numpy()
            self.handles.append(self.model.model.layers[l].register_forward_hook(hook))
        self.runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','kernels','triton','numpy']}
        self.runtime.update(gpu=torch.cuda.get_device_name(),attention='eager',quantization='original MXFP4',validation='Independent replay using the same causal chunk schedule; no full-forward numerical equivalence claim.')

    def run(self,ids,cuts):
        torch=self.torch;past=None;parts=[]
        with torch.inference_mode():
            for lo,hi in chunk_ranges(len(ids),cuts,self.spec['chunk_tokens']):
                self.captured.clear()
                result=self.model.model(input_ids=torch.tensor([ids[lo:hi]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True)
                past=result.past_key_values
                values=np.empty((hi-lo,12),dtype=np.float32)
                for layer,js in self.groups.items():values[:,js]=self.captured[layer]
                require(np.isfinite(values).all(),'Nonfinite emotion scores');parts.append(values)
        del result,past
        return np.concatenate(parts)

    def score(self,job,arrays,output,smoke=False):
        started=time.monotonic();ids,offsets,kind=arrays['token_ids'],arrays['offsets'],arrays['kind']
        encoded=self.tokenizer(job['rendered'],add_special_tokens=False,return_offsets_mapping=True)
        require(encoded['input_ids']==ids.tolist(),'Tokenizer IDs changed')
        require(encoded['offset_mapping']==[tuple(x) for x in offsets.tolist()],'Tokenizer offsets changed')
        require(hashlib.sha256(job['rendered'].encode()).hexdigest()==job['rendered_sha256'],'Rendering changed')
        cuts=sentence_cuts(offsets,job['source_boundaries']);count=max(cuts)
        require(count<=self.spec['max_tokens'],'Context limit exceeded')
        ids,offsets,kind=ids[:count],offsets[:count],kind[:count]
        scores=self.run(ids,cuts);checks=[]
        for name,boundary in [('future_removed_at_first_sentence',cuts[0])]+([('independent_causal_schedule_repeat',cuts[len(cuts)//2])] if smoke else []):
            prefix=self.run(ids[:boundary],[c for c in cuts if c<=boundary])
            rel=float(np.linalg.norm(prefix-scores[:boundary])/max(np.linalg.norm(prefix),1e-12))
            require(rel<1e-6,f'Causal prefix mismatch: {rel}')
            checks.append(dict(check=name,tokens=boundary,relative_l2=rel,max_abs_difference=float(np.max(np.abs(prefix-scores[:boundary])))))
        selected=np.flatnonzero(np.isin(kind,[1,2]));buf=io.BytesIO()
        np.savez_compressed(buf,scores=scores[selected],offsets=offsets[selected],token_ids=ids[selected],native_token_index=selected,response_char_start=np.array(0),kind=kind[selected])
        blob=buf.getvalue();path=output/(job['id']+'.npz');path.write_bytes(blob)
        receipt=dict(schema_version=1,id=job['id'],replay_identity=self.spec['identity'],probe_identity=self.spec['probe_identity'],input_sha256=job['input_sha256'],source_sha256=job['source_sha256'],rendered_sha256=job['rendered_sha256'],sha256=sha(path),prefix_tokens=count,assistant_replay_tokens=len(selected),sentence_boundary_count=len(cuts),sentence_cuts_sha256=digest(cuts),tokenizer_match=True,causal_checks=checks,runtime=self.runtime,seconds=time.monotonic()-started)
        atomic(path.with_suffix('.receipt.json'),receipt)
        return blob,receipt
