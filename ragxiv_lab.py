"""RAGxiv: QASPER research-paper RAG experiments (paper-scoped evaluation)."""
from __future__ import annotations
import json, math, re, tarfile, time, urllib.request, os, hashlib, itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path.cwd()
ART = ROOT / 'artifacts'
URL = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz'
EMBED_NAME = 'BAAI/bge-small-en-v1.5'


def norm(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def load_papers(split='dev', limit=12):
    """Official JSON, not obsolete HuggingFace dataset-script loader."""
    assert split in ('train','dev'), 'Do not tune on QASPER test.'
    ART.mkdir(exist_ok=True, parents=True)
    target = ART / f'qasper-{split}-v0.3.json'
    if not target.exists():
        archive = ART / 'qasper-train-dev-v0.3.tgz'
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive, 'r:gz') as tf:
            member = next((m for m in tf.getmembers() if m.isfile() and Path(m.name).name == target.name), None)
            if member is None: raise FileNotFoundError(f'{target.name} not found in official archive')
            with tf.extractfile(member) as inp, target.open('wb') as out:
                import shutil
                shutil.copyfileobj(inp,out)
    data = json.loads(target.read_text(encoding='utf-8'))
    return dict(list(data.items())[:limit]) if limit else data


def paragraphs(paper):
    """Returns stable IDs tied to original evidence units."""
    result=[]
    abstract=paper.get('abstract') or ''
    if isinstance(abstract,list): abstract=' '.join(map(str,abstract))
    if norm(abstract):result.append(('abstract', 'Abstract', norm(abstract)))
    ft=paper.get('full_text') or []
    # QASPER's official JSON uses a list of section dictionaries. Some converted
    # variants use a column-oriented dict; accept either representation.
    if isinstance(ft, list):
        sections=ft
    elif isinstance(ft, dict):
        names=ft.get('section_name',[])
        contents=ft.get('paragraphs',[])
        sections=[{'section_name':names[i] if i<len(names) else f'Section {i+1}',
                   'paragraphs':ps} for i,ps in enumerate(contents)]
    else:
        raise TypeError(f'Unexpected full_text format: {type(ft)}')
    for si, section in enumerate(sections):
        if not isinstance(section,dict): continue
        name=section.get('section_name') or f'Section {si+1}'
        parts=section.get('paragraphs') or []
        if isinstance(parts,str): parts=[parts]
        for pi,part in enumerate(parts):
            if isinstance(part,str) and norm(part):
                result.append((f's{si}_p{pi}',str(name),norm(part)))
    return result


def annotations(paper_id,paper):
    qas=paper.get('qas') or {}
    if isinstance(qas,list): rows=qas
    else:
        qs=qas.get('question',[]); ids=qas.get('question_id',[]); ans=qas.get('answers',[])
        rows=[{'question':q, 'question_id':ids[i] if i<len(ids) else f'{paper_id}_{i}',
               'answers':ans[i] if i<len(ans) else []} for i,q in enumerate(qs)]
    out=[]
    for qi,qa in enumerate(rows):
        aa=qa.get('answers') or []
        if isinstance(aa,dict):
            aa=[{'answer': item} for item in aa.get('answer',[])]
        ev=[]; refs=[]; unanswerable=False
        for a in aa:
            d=a.get('answer',a) if isinstance(a,dict) else {}
            if not isinstance(d,dict):continue
            ev.extend(d.get('evidence') or [])
            if d.get('unanswerable'): unanswerable=True
            if d.get('free_form_answer'):refs.append(norm(d['free_form_answer']))
            if d.get('yes_no') is not None:refs.append('yes' if d['yes_no'] else 'no')
            if d.get('extractive_spans'):refs.append(norm(' '.join(d['extractive_spans'])))
        out.append({'paper_id':paper_id,'question_id':str(qa.get('question_id',f'{paper_id}_{qi}')),
                    'question':norm(qa.get('question')),'evidence':list(dict.fromkeys(norm(e) for e in ev if norm(e))),
                    'references':list(dict.fromkeys(refs)), 'unanswerable':unanswerable})
    return out


def gold_ids(paper,qa):
    # QASPER annotators provide the literal evidence paragraphs, sometimes unavailable
    # for figures/tables. Exact normalized matching avoids fabricated gold labels.
    lookup=defaultdict(set)
    for pid,_,txt in paragraphs(paper):lookup[norm(txt).casefold()].add(pid)
    mapped=set()
    for txt in qa['evidence']: mapped.update(lookup.get(norm(txt).casefold(),set()))
    return mapped


@dataclass(frozen=True)
class Config:
    chunking: str='paragraph'  # paragraph, fixed, recursive, semantic, sentence_window, parent, auto_merging
    retrieval: str='dense'     # dense, bm25, hybrid
    rerank: bool=False
    transformation: str='none' # none, rewrite, multi_query, hyde
    top_k: int=5
    candidate_k: int=15
    chunk_size: int=256
    overlap: int=32
    window_size: int=2
    merge_threshold: float=.5
    rrf_k: int=60
    embed_model: str=EMBED_NAME
    generator_model: str='qwen2.5:3b'
    semantic_buffer: int=1
    semantic_percentile: int=90
    parent_chunk_size: int=512
    child_chunk_size: int=128
    bm25_k1: float=1.5
    bm25_b: float=0.75
    dense_top_k: int=15
    sparse_top_k: int=15
    rerank_top_n: int=5
    reranker_model: str='cross-encoder/ms-marco-MiniLM-L-6-v2'
    query_count: int=3
    hyde_count: int=1


def build_nodes(paper_id,paper,cfg,embed):
    from llama_index.core import Document
    from llama_index.core.schema import TextNode
    from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, SentenceWindowNodeParser
    source=[]; parents={}; children={}
    docs=[]
    for pid,section,txt in paragraphs(paper):
        metadata={'paper_id':paper_id, 'source_pid':pid,'section':section,'title':paper.get('title','')}
        docs.append(Document(text=txt, id_=f'{paper_id}::{pid}', metadata=metadata,
                             excluded_embed_metadata_keys=list(metadata), excluded_llm_metadata_keys=list(metadata)))
        parents[pid]=txt
    if cfg.chunking=='paragraph':
        nodes=[TextNode(id_=d.id_,text=d.text,metadata=d.metadata,
                        excluded_embed_metadata_keys=list(d.metadata),excluded_llm_metadata_keys=list(d.metadata)) for d in docs]
    elif cfg.chunking=='fixed':
        # TokenTextSplitter splits using token budget (not character count).
        from llama_index.core.node_parser import TokenTextSplitter
        parser=TokenTextSplitter(chunk_size=cfg.chunk_size,chunk_overlap=cfg.overlap)
        nodes=parser.get_nodes_from_documents(docs)
    elif cfg.chunking=='recursive':
        parser=SentenceSplitter(chunk_size=cfg.chunk_size,chunk_overlap=cfg.overlap)
        nodes=parser.get_nodes_from_documents(docs)
    elif cfg.chunking=='semantic':
        parser=SemanticSplitterNodeParser.from_defaults(embed_model=embed,buffer_size=cfg.semantic_buffer,breakpoint_percentile_threshold=cfg.semantic_percentile)
        nodes=parser.get_nodes_from_documents(docs)
    elif cfg.chunking=='sentence_window':
        parser=SentenceWindowNodeParser.from_defaults(window_size=cfg.window_size,window_metadata_key='window',original_text_metadata_key='original_text')
        nodes=parser.get_nodes_from_documents(docs)
    elif cfg.chunking in ('parent','auto_merging'):
        # Parent representation is source paragraph; children use configurable token window.
        # The parent size parameter controls eligible parent splitting when the source
        # paragraph exceeds that size. Ground truth remains original paragraph IDs.
        parent_parser=SentenceSplitter(chunk_size=cfg.parent_chunk_size,chunk_overlap=0)
        parent_nodes=parent_parser.get_nodes_from_documents(docs)
        child_parser=SentenceSplitter(chunk_size=cfg.child_chunk_size,chunk_overlap=min(cfg.overlap,cfg.child_chunk_size-1))
        nodes=[]
        for parent in parent_nodes:
            source_pid=parent.metadata['source_pid']
            parent_key=parent.node_id
            parents[parent_key]=parent.text
            for child in child_parser.get_nodes_from_documents([Document(text=parent.text,metadata={**parent.metadata,'parent_key':parent_key},id_=parent_key)]):
                child.metadata['source_pid']=source_pid
                child.metadata['parent_key']=parent_key
                nodes.append(child)
    else: raise ValueError(cfg.chunking)
    for n in nodes:
        pid=n.metadata.get('source_pid')
        children.setdefault(n.metadata.get('parent_key',pid),[]).append(n.node_id)
    return nodes,parents,children


def get_embedder(name=EMBED_NAME):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name=name)


class SearchLab:
    def __init__(self,pid,paper,cfg:Config, embed=None):
        from llama_index.core import VectorStoreIndex
        self.pid=pid;self.paper=paper;self.cfg=cfg
        self.embed=embed or get_embedder(cfg.embed_model)
        self.nodes,self.parents,self.children=build_nodes(pid,paper,cfg,self.embed)
        self.by_id={n.node_id:n for n in self.nodes}
        self.index=None;self.bm25=None;self.reranker=None
        if cfg.retrieval in ('dense','hybrid'):
            self.index=VectorStoreIndex(self.nodes,embed_model=self.embed,show_progress=False)
        if cfg.retrieval in ('bm25','hybrid'):
            from llama_index.retrievers.bm25 import BM25Retriever
            self.bm25=BM25Retriever.from_defaults(nodes=self.nodes,similarity_top_k=cfg.sparse_top_k)
        if cfg.rerank:
            from sentence_transformers import CrossEncoder
            self.reranker=CrossEncoder(cfg.reranker_model)

    def _query_variants(self,q):
        cfg=self.cfg
        if cfg.transformation=='none':return [q]
        prompt={
            'rewrite':f'Rewrite this academic research question for retrieving paper passages. Preserve meaning. Output only one query.\nQuestion: {q}',
            'multi_query':f'Write {cfg.query_count} diverse search queries for the research question, one per line, no numbering.\nQuestion: {q}',
            'hyde':f'Write a short hypothetical scientific-paper passage that could answer this question. Do not claim to have read the paper.\nQuestion: {q}'
        }[cfg.transformation]
        result=ollama_chat(prompt,cfg.generator_model)
        if cfg.transformation=='multi_query':
            # LLM output can contain numbered lines, empty bullets, or placeholders.
            # Keep the original question first and deduplicate cleaned variants.
            cleaned=[]
            for line in result.splitlines():
                candidate=re.sub(r'^\s*(?:[-*]\s*|\d+[.)]\s*)+', '', line).strip().strip('"`')
                if candidate and candidate.casefold() not in {'none', 'null', 'n/a'}:
                    cleaned.append(candidate)
            return list(dict.fromkeys([q]+cleaned[:cfg.query_count]))
        return [result.strip()] if result.strip() else [q]

    def search(self,question, top_k=None):
        from llama_index.core.schema import NodeWithScore
        k=top_k or self.cfg.top_k
        variants=self._query_variants(question)
        rankings=[]
        for variant_index, q in enumerate(variants):
            # Some versions of the BM25/dense stack raise TypeError when a
            # generated query produces an unusable internal score (None * float).
            # That query must not abort the entire paper/configuration. Always
            # require the original question to work, and explicitly report any
            # failed *additional* query, rather than silently inventing scores.
            if not isinstance(q, str) or not q.strip():
                continue
            sources=[]
            if self.index is not None:
                sources.append(('dense', lambda q=q: self.index.as_retriever(
                    similarity_top_k=self.cfg.dense_top_k).retrieve(q)))
            if self.bm25 is not None:
                sources.append(('bm25', lambda q=q: self.bm25.retrieve(q)))
            for source_name, retrieve in sources:
                try:
                    ranking=retrieve()
                except TypeError as exc:
                    if variant_index == 0 or 'NoneType' not in str(exc):
                        raise RuntimeError(
                            f'{source_name} retrieval failed for '
                            f'{"original" if variant_index == 0 else "generated"} query {q!r}'
                        ) from exc
                    import warnings
                    warnings.warn(
                        f'Skipping malformed multi-query {variant_index} ({source_name}): '
                        f'{exc}; question={q[:100]!r}', RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if ranking:
                    rankings.append(ranking)
        if not rankings:
            raise RuntimeError(f'No successful retrieval results for question: {question[:120]!r}')
        if len(rankings)==1:
            candidates=rankings[0]
        else:
            # RRF rank fusion: mathematically comparable between BM25 and dense.
            scores=defaultdict(float);records={}
            for ranking in rankings:
                for i,result in enumerate(ranking,1):
                    nodeid=result.node.node_id
                    scores[nodeid]+=1/(self.cfg.rrf_k+i)
                    records[nodeid]=result.node
            candidates=[NodeWithScore(node=records[n],score=s) for n,s in sorted(scores.items(),key=lambda a:-a[1])]
        # Expansion after retrieval, preserving the source paragraph for evaluation.
        if self.cfg.chunking in ('parent','auto_merging'):
            retrieved_counts=Counter(x.node.metadata.get('parent_key',x.node.metadata.get('source_pid')) for x in candidates[:self.cfg.candidate_k])
            expanded=[];done=set()
            for x in candidates:
                pid=x.node.metadata.get('parent_key',x.node.metadata.get('source_pid'))
                can_expand=(self.cfg.chunking=='parent' or retrieved_counts[pid]/max(1,len(self.children.get(pid,[])))>=self.cfg.merge_threshold)
                if can_expand and pid in self.parents:
                    if pid in done:continue
                    done.add(pid)
                    from llama_index.core.schema import TextNode
                    n=TextNode(id_=f'{self.pid}::{pid}::expanded',text=self.parents[pid],metadata={**x.node.metadata,'expanded':True})
                    expanded.append(NodeWithScore(node=n,score=x.score))
                else:expanded.append(x)
            candidates=expanded
        if self.cfg.rerank and candidates:
            pairs=[(question,x.node.metadata.get('window',x.node.text) if self.cfg.chunking=='sentence_window' else x.node.text) for x in candidates]
            scores=self.reranker.predict(pairs)
            candidates=[v for _,v in sorted(zip(scores,candidates),key=lambda t:-float(t[0]))][:self.cfg.rerank_top_n]
        return candidates[:k]


def passage(hit,cfg):
    return hit.node.metadata.get('window',hit.node.text) if cfg.chunking=='sentence_window' else hit.node.text


def retrieval_metrics(hits,gold,k):
    # Paragraph-level de-duplication: duplicates from chunking should not inflate recall.
    ranked=[]
    for h in hits[:k]:
        sid=h.node.metadata.get('source_pid')
        if sid not in ranked:ranked.append(sid)
    rel=[int(x in gold) for x in ranked]
    found=sum(rel)
    precision=found/k
    recall=found/len(gold) if gold else float('nan')
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(rel))
    ideal=sum(1/math.log2(i+2) for i in range(min(len(gold),k)))
    ap=sum(sum(rel[:i+1])/(i+1) for i,v in enumerate(rel) if v)/min(len(gold),k) if gold else float('nan')
    rr=next((1/(i+1) for i,v in enumerate(rel) if v),0)
    return {'precision_at_k':precision,'recall_at_k':recall,'hit_at_k':float(found>0),'mrr_at_k':rr,
            'map_at_k':ap,'ndcg_at_k':dcg/ideal if ideal else float('nan'),
            'retrieved_unique_paragraphs':len(ranked)}


def evaluate_retrieval(papers,cfg:Config,max_questions=50):
    import pandas as pd
    t0=time.perf_counter();rows=[];mapping={'questions_total':0,'questions_mappable':0,'questions_unmapped':0,'unanswerable':0}
    embed=get_embedder(cfg.embed_model)
    for pid,paper in papers.items():
        qas=annotations(pid,paper)
        chosen=[]
        for q in qas:
            if len(rows)+len(chosen)>=max_questions:break
            mapping['questions_total']+=1
            if q['unanswerable']:mapping['unanswerable']+=1
            gold=gold_ids(paper,q)
            if not gold: mapping['questions_unmapped']+=1;continue
            mapping['questions_mappable']+=1
            chosen.append((q,gold))
        if not chosen:continue
        lab=SearchLab(pid,paper,cfg,embed)
        for q,gold in chosen:
            start=time.perf_counter()
            hits=lab.search(q['question'])
            d=retrieval_metrics(hits,gold,cfg.top_k)
            rows.append({'paper_id':pid,'paper_title':paper.get('title',''),'question_id':q['question_id'],
                         'question':q['question'],'latency_s':time.perf_counter()-start,
                         'gold_count':len(gold),'retrieved_ids':[x.node.metadata.get('source_pid') for x in hits],
                         'gold_ids':sorted(gold),'contexts':[passage(x,cfg) for x in hits],
                         'references':q['references'],**d})
        if len(rows)>=max_questions:break
    mapping['runtime_s']=round(time.perf_counter()-t0,2)
    return pd.DataFrame(rows),mapping


def summary(df,cfg,mapping):
    metrics=['precision_at_k','recall_at_k','hit_at_k','mrr_at_k','map_at_k','ndcg_at_k','latency_s']
    return {**asdict(cfg),**{m:float(df[m].mean()) if len(df) else None for m in metrics},
            'evaluated_questions':len(df),**mapping}


def save_run(df,cfg,mapping,run_name):
    import pandas as pd
    ART.mkdir(exist_ok=True,parents=True)
    folder=ART/'runs';folder.mkdir(exist_ok=True)
    df.to_json(folder/f'{run_name}.json',orient='records',indent=2)
    (folder/f'{run_name}_summary.json').write_text(json.dumps(summary(df,cfg,mapping),indent=2))
    return folder/f'{run_name}_summary.json'


def ollama_chat(prompt,model='qwen2.5:3b',url='http://localhost:11434',json_mode=False):
    import requests
    r=requests.post(url+'/api/generate',json={'model':model,'prompt':prompt,'stream':False, **({'format':'json', 'options':{'temperature':0}} if json_mode else {})},timeout=180)
    r.raise_for_status()
    return r.json()['response']


def generate_answer(question, hits, cfg, max_chars=14000):
    context='\n\n'.join(f'[{i+1}] {passage(hit,cfg)}' for i,hit in enumerate(hits))[:max_chars]
    prompt=f'''Answer the research question using ONLY the evidence below. If insufficient, say you cannot determine it. Cite source numbers in square brackets.\nQUESTION: {question}\nEVIDENCE:\n{context}\nANSWER:'''
    return ollama_chat(prompt,cfg.generator_model)



def judge_scores(question, answer, contexts, model='qwen2.5:3b', max_attempts=3):
    """Local Ollama LLM-as-judge; return NaN for failed scores, never zero.

    These scores are exploratory proxies, not the same as RAGAS metrics.
    """
    import math
    prompt = f'''Evaluate this RAG response. Return a JSON object ONLY with numeric values 0 to 1:
{{"groundedness":0.0,"context_relevance":0.0,"answer_relevance":0.0}}
Groundedness: are the answer's factual claims supported by the retrieved context?
Context relevance: are the supplied passages relevant to the question?
Answer relevance: does the response address the question?
Question: {question}\nAnswer: {answer}\nContexts: {json.dumps(contexts,ensure_ascii=False)[:11000]}'''
    required = ('groundedness', 'context_relevance', 'answer_relevance')
    for attempt in range(1, max_attempts+1):
        try:
            txt = ollama_chat(prompt, model, json_mode=True)
            found = re.search(r'\{.*\}', txt, re.S)
            vals = json.loads(found.group(0)) if found else json.loads(txt)
            scores = {k: float(vals[k]) for k in required}
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
                raise ValueError('Judge scores must be finite values in [0,1]')
            return scores
        except Exception as exc:
            print(f'Local judge attempt {attempt}/{max_attempts} failed: {type(exc).__name__}: {str(exc)[:180]}')
            if attempt < max_attempts:
                time.sleep(min(attempt, 3))
    return {k: None for k in required}


def evaluate_generation(df,cfg,limit=8,with_judge=True):
    import pandas as pd
    records=[]
    for row in df.head(limit).to_dict('records'):
        # Reuse retrieval contexts from stored output; no reindex necessary.
        contexts=row['contexts']; question=row['question']
        from llama_index.core.schema import NodeWithScore,TextNode
        hits=[NodeWithScore(node=TextNode(text=t),score=1.) for t in contexts]
        answer=generate_answer(question,hits,cfg)
        scores=judge_scores(question,answer,contexts,cfg.generator_model) if with_judge else {}
        records.append({**row,'answer':answer,**scores})
    return pd.DataFrame(records)


def reference_token_f1(answer, references):
    """Deterministic best-over-references token F1 (not an LLM judge)."""
    import re
    from collections import Counter
    from ast import literal_eval
    if isinstance(references, str):
        try:
            parsed = literal_eval(references)
            references = parsed if isinstance(parsed, (list, tuple)) else [references]
        except (ValueError, SyntaxError):
            references = [references]
    if not isinstance(references, (list, tuple)):
        return float("nan")
    def toks(s):
        return re.findall(r"\w+", str(s).casefold())
    pred = toks(answer)
    scores = []
    for reference in references:
        gold = toks(reference)
        if not gold:
            continue
        common = sum((Counter(pred) & Counter(gold)).values())
        if not pred:
            scores.append(0.0)
            continue
        precision, recall = common / len(pred), common / len(gold)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return max(scores) if scores else float("nan")



async def run_ragas_optional(
    generation_df,
    model="qwen2.5:7b",
    *,
    models=None,
    max_attempts=2,
    retry_base_seconds=1.5,
    ollama_url="http://localhost:11434",
    embedding_model=None,
):
    """Evaluate cached QASPER answers with RAGAS and LOCAL Ollama judges.

    Defaults to qwen2.5:7b as judge, with optional ordered local-model
    fallbacks. Uses local sentence-transformer embeddings for AnswerRelevancy.
    No Gemini, OpenAI cloud API, or GOOGLE_API_KEY is involved.

    Keep this function async; run `await run_ragas_optional(...)` in Jupyter.
    """
    import ast as ast_module
    import asyncio
    import random
    import re
    import numpy as np
    import pandas as pd
    import requests
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.metrics.collections import (
        Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy,
    )

    if models is None:
        models = [model]
    elif isinstance(models, str):
        models = [models]
    models = list(dict.fromkeys(m for m in models if m))
    if not models:
        raise ValueError("Specify at least one local Ollama judge model.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    required = {"question", "answer", "contexts", "references"}
    missing = required.difference(generation_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Got {list(generation_df.columns)}")

    # Fail fast with an actionable message if Ollama is not running or a model is missing.
    base = ollama_url.rstrip("/")
    try:
        response = requests.get(f"{base}/api/tags", timeout=10)
        response.raise_for_status()
        installed = {m["name"] for m in response.json().get("models", [])}
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base}. Open the Ollama app or run `ollama serve`."
        ) from exc
    available = [m for m in models if m in installed]
    if not available:
        raise RuntimeError(
            f"Judge model(s) {models} are not installed. Available: {sorted(installed)}. "
            f"Run `ollama pull {models[0]}` in Terminal."
        )
    if len(available) != len(models):
        print("Skipping missing local judge models:", sorted(set(models) - set(available)))
    models = available

    # Async client works inside Jupyter's already-running asyncio event loop.
    # No hidden provider retries: we own the metric-level retry loop below.
    client = AsyncOpenAI(
        api_key="ollama", base_url=f"{base}/v1/",
        timeout=180.0, max_retries=0,
    )
    # Same local embedding model used for the retrieval experiments by default.
    embeddings = HuggingFaceEmbeddings(model=embedding_model or EMBED_NAME)
    metric_cache = {}
    metric_names = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")

    def metrics_for(model_id):
        if model_id not in metric_cache:
            llm = llm_factory(model_id, provider="openai", client=client)
            metric_cache[model_id] = {
                "faithfulness": Faithfulness(llm=llm),
                "context_precision": ContextPrecision(llm=llm),
                "context_recall": ContextRecall(llm=llm),
                "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            }
        return metric_cache[model_id]

    def as_list(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(x) for x in value if x is not None and str(x).strip()]
        if isinstance(value, str) and value.strip():
            try:
                parsed = ast_module.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x is not None and str(x).strip()]
            except (ValueError, SyntaxError):
                pass
            return [value]
        return []

    output = []
    try:
        for index, row in generation_df.iterrows():
            question = str(row["question"])
            answer = str(row["answer"])
            contexts = as_list(row["contexts"])
            references = as_list(row["references"])
            reference = references[0] if references else None
            record = {
                "question": question,
                "answer": answer,
                "reference": reference,
                "reference_count": len(references),
                "reference_token_f1": reference_token_f1(answer, references),
                "judge_model": None,
            }
            inputs = {
                "faithfulness": dict(user_input=question, response=answer, retrieved_contexts=contexts),
                "context_precision": dict(user_input=question, reference=reference, retrieved_contexts=contexts),
                "context_recall": dict(user_input=question, reference=reference, retrieved_contexts=contexts),
                "answer_relevancy": dict(user_input=question, response=answer),
            }
            for name in metric_names:
                record[name] = float("nan")
                record[name + "_error"] = ""
                record[name + "_model"] = None
                if not contexts or (name in ("context_precision", "context_recall") and not reference):
                    record[name + "_error"] = "Missing context or reference"
                    continue
                errors = []
                for model_id in models:
                    try:
                        metric = metrics_for(model_id)[name]
                    except Exception as exc:
                        errors.append(f"{model_id}: initialization: {type(exc).__name__}: {str(exc)[:180]}")
                        continue
                    for attempt in range(1, max_attempts + 1):
                        try:
                            score = await metric.ascore(**inputs[name])
                            value = float(score.value)
                            if not np.isfinite(value):
                                raise ValueError("Metric returned non-finite score")
                            record[name] = value
                            record[name + "_model"] = model_id
                            if record["judge_model"] is None:
                                record["judge_model"] = model_id
                            record[name + "_error"] = ""
                            break
                        except Exception as exc:
                            message = str(exc)
                            errors.append(f"{model_id} attempt {attempt}: {type(exc).__name__}: {message[:160]}")
                            # Local model failures are often structured-output problems.
                            if attempt < max_attempts:
                                delay = retry_base_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                                print(f"Row {index}: {name} / {model_id} retrying in {delay:.1f}s: {message[:100]}")
                                await asyncio.sleep(delay)
                    if pd.notna(record[name]):
                        break
                    print(f"Row {index}: {name} failed with {model_id}; trying next local judge if configured")
                if pd.isna(record[name]):
                    record[name + "_error"] = " | ".join(errors)[-1200:]
                    print(f"Row {index}: {name} failed after local judge attempts")
            output.append(record)
    finally:
        await client.close()
    return pd.DataFrame(output)

# --- Compact, controlled experiment grids ----------------------------------
# Changing one family at a time makes results interpretable and avoids the
# full Cartesian-product explosion.
CHUNK_OPTIONS = {
    'paragraph': [{}],
    'fixed': [{'chunk_size':256,'overlap':32},
              {'chunk_size':512,'overlap':32}],
    'recursive': [{'chunk_size':256,'overlap':32},
                  {'chunk_size':512,'overlap':32}],
    'semantic': [{'semantic_buffer':1,'semantic_percentile':90},
                 {'semantic_buffer':2,'semantic_percentile':90}],
    'sentence_window': [{'window_size':1},{'window_size':3}],
    'parent': [{'parent_chunk_size':512,'child_chunk_size':128},
               {'parent_chunk_size':1024,'child_chunk_size':256}],
    'auto_merging': [{'parent_chunk_size':512,'child_chunk_size':128,'merge_threshold':0.35},
                     {'parent_chunk_size':512,'child_chunk_size':128,'merge_threshold':0.60}],
}
RETRIEVAL_OPTIONS = {
    'dense': [{'dense_top_k':10},{'dense_top_k':20}],
    'bm25': [{'sparse_top_k':10},{'sparse_top_k':20}],
    'hybrid': [{'dense_top_k':10,'sparse_top_k':10,'rrf_k':60},
               {'dense_top_k':20,'sparse_top_k':20,'rrf_k':30}],
}
RERANK_OPTIONS = [{'rerank':True,'rerank_top_n':5},
                  {'rerank':True,'rerank_top_n':10}]
TRANSFORM_OPTIONS = {
    'rewrite': [{}],
    'multi_query': [{'query_count':2},{'query_count':3}],
    'hyde': [{}]
}


def make_controlled_grid(base=None,chunk_options=None,retrieval_options=None,
                         rerank_options=None,transform_options=None,
                         include_llm=False):
    """Generate compact ablations (not a cross-product of all families).

    1. Baseline: paragraph + dense
    2. Chunk variants with dense retrieval
    3. Retriever variants with recursive-256 chunks
    4. Reranker variants with hybrid retrieval
    5. Optional LLM query transforms with the same hybrid setup
    Configs are deduplicated and ordered deterministically.
    """
    from dataclasses import replace
    base=base or Config(top_k=5,candidate_k=20)
    chunk_options=CHUNK_OPTIONS if chunk_options is None else chunk_options
    retrieval_options=RETRIEVAL_OPTIONS if retrieval_options is None else retrieval_options
    rerank_options=RERANK_OPTIONS if rerank_options is None else rerank_options
    transform_options=TRANSFORM_OPTIONS if transform_options is None else transform_options
    seen=set(); output=[]
    def add(cfg):
        if cfg.chunking in ('fixed','recursive') and cfg.overlap>=cfg.chunk_size:
            raise ValueError('Overlap must be smaller than chunk size')
        if cfg.rerank and cfg.rerank_top_n<cfg.top_k:
            raise ValueError('rerank_top_n must be >= top_k')
        key=run_id(cfg)
        if key not in seen:seen.add(key);output.append(cfg)
    baseline=replace(base,chunking='paragraph',retrieval='dense',rerank=False,transformation='none')
    add(baseline)
    for method,variations in chunk_options.items():
        for params in variations:
            add(replace(baseline,chunking=method,**params))
    retrieval_base=replace(base,chunking='recursive',chunk_size=256,overlap=32,
                            retrieval='dense',rerank=False,transformation='none')
    for method,variations in retrieval_options.items():
        for params in variations:
            add(replace(retrieval_base,retrieval=method,**params))
    hybrid_base=replace(retrieval_base,retrieval='hybrid',dense_top_k=10,
                        sparse_top_k=10,rrf_k=60)
    for params in rerank_options:
        add(replace(hybrid_base,**params))
    if include_llm:
        for method,variations in transform_options.items():
            for params in variations:
                add(replace(hybrid_base,transformation=method,**params))
    return output


def run_id(cfg):
    values=asdict(cfg)
    hashed=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:12]
    return f"{cfg.chunking}_{cfg.retrieval}_{cfg.transformation}_{'rerank' if cfg.rerank else 'raw'}_{hashed}"


def run_sweep(papers,configs,max_questions=30,skip_completed=True,
              fail_fast=False,progress=None):
    """Run every selected configuration. NO time cutoff or max_runs limit."""
    import pandas as pd
    failures=[];records=[]
    configs=list(configs)
    folder=ART/'runs';folder.mkdir(exist_ok=True,parents=True)
    for i,cfg in enumerate(configs,1):
        name=run_id(cfg)
        path=folder/f'{name}_summary.json'
        if skip_completed and path.exists():
            item=json.loads(path.read_text())
            records.append({'run':name,'status':'cached',**item})
            if progress: progress(i,len(configs),name,'cached')
            continue
        try:
            df,mapping=evaluate_retrieval(papers,cfg,max_questions=max_questions)
            save_run(df,cfg,mapping,name)
            records.append({'run':name,'status':'ok',**summary(df,cfg,mapping)})
            if progress:progress(i,len(configs),name,'ok')
        except Exception as exc:
            import traceback
            failures.append({'run':name,'exception':repr(exc),
                             'traceback':traceback.format_exc(),**asdict(cfg)})
            if progress:progress(i,len(configs),name,f'error: {exc}')
            if fail_fast:raise
    pd.DataFrame(failures).to_json(ART/'sweep_failures.json',orient='records',indent=2)
    return pd.DataFrame(records),pd.DataFrame(failures)
