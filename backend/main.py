from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import sqlite3, json, uuid, shutil, mimetypes, io, os, math, statistics
from datetime import datetime, timezone
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
UPLOADS = DATA / 'uploads'; EXPORTS = DATA / 'exports'; REPORTS = DATA / 'reports'
for p in (UPLOADS, EXPORTS, REPORTS): p.mkdir(parents=True, exist_ok=True)
DB = DATA / 'platform.db'
app = FastAPI(title='Multimodal Multi-Agent Intelligent Analytics Platform', version='1.0.0')

AGENTS = [
 ('Data Ingestion Agent','Detect file format, validate schema, and route processors.'),
 ('Data Quality & Cleaning Agent','Inspect missing values, duplicates, constants, and outliers.'),
 ('Statistical Analysis Agent','Compute descriptive statistics and numeric correlations.'),
 ('Visualization Agent','Recommend chart types and generate visualization-ready data.'),
 ('AI Decision Support Agent','Turn measured results into evidence-based recommendations.'),
 ('Report Generation Agent','Compile analysis into a downloadable executive report.')]

def now(): return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
def conn():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
def init_db():
    with conn() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,title TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY,session_id TEXT,original_filename TEXT,stored_path TEXT,file_type TEXT,size INTEGER,status TEXT,extracted_metadata TEXT,has_text_data INTEGER,created_at TEXT);
        CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,session_id TEXT,file_id TEXT,name TEXT,raw_path TEXT,working_path TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS cleaning(id INTEGER PRIMARY KEY AUTOINCREMENT,dataset_id TEXT,operation_type TEXT,rows_affected INTEGER,summary TEXT,applied_at TEXT);
        CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY,session_id TEXT,dataset_id TEXT,workflow_type TEXT,status TEXT,created_at TEXT,execution_records TEXT,summary TEXT);
        CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY,session_id TEXT,dataset_id TEXT,title TEXT,path TEXT,summary TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,role TEXT,content TEXT,created_at TEXT);
        ''')
        if not c.execute('SELECT 1 FROM sessions LIMIT 1').fetchone():
            sid=str(uuid.uuid4()); c.execute('INSERT INTO sessions VALUES(?,?,?,?)',(sid,'Primary Analysis Session',now(),now()))
init_db()

def rowdict(r): return dict(r) if r else None
def get_dataset(did):
    with conn() as c: r=c.execute('SELECT * FROM datasets WHERE id=?',(did,)).fetchone()
    if not r: raise HTTPException(404,'Dataset not found')
    return dict(r)
def read_df(path):
    p=Path(path)
    try:
        if p.suffix.lower()=='.csv': return pd.read_csv(p)
        if p.suffix.lower() in ('.xlsx','.xls'): return pd.read_excel(p)
        return None
    except Exception as e: raise HTTPException(400, f'Could not read dataset: {e}')
def write_df(df,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix.lower()=='.csv': df.to_csv(p,index=False)
    else: df.to_excel(p,index=False)
def classify(name):
    ext=Path(name).suffix.lower()
    if ext=='.csv': return 'csv'
    if ext in ('.xlsx','.xls'): return 'excel'
    if ext=='.pdf': return 'pdf'
    if ext in ('.png','.jpg','.jpeg','.webp','.bmp'): return 'image'
    if ext in ('.wav','.mp3','.m4a','.ogg','.flac'): return 'audio'
    if ext in ('.txt','.md','.log'): return 'text'
    return 'unknown'
def extract_text(path, typ):
    p=Path(path)
    if typ=='text':
        try: return p.read_text(errors='ignore')[:100000]
        except: return ''
    if typ=='pdf':
        try:
            import pypdf
            reader=pypdf.PdfReader(str(p)); return '\n'.join((page.extract_text() or '') for page in reader.pages)[:100000]
        except: return 'PDF stored. Install pypdf for text extraction.'
    if typ=='image':
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(p))[:50000]
        except: return 'Image stored. Optional OCR requires Pillow and a local Tesseract installation.'
    if typ=='audio':
        return 'Audio stored. Optional speech-to-text requires a configured transcription engine.'
    return ''
def profile(df):
    missing=int(df.isna().sum().sum()); dup=int(df.duplicated().sum())
    mem=int(df.memory_usage(deep=True).sum())
    schema=[]
    for col in df.columns:
        s=df[col]; vals=s.dropna().astype(str).head(4).tolist()
        schema.append({'name':str(col),'dtype':str(s.dtype),'missing':int(s.isna().sum()),'unique':int(s.nunique(dropna=True)),'samples':vals})
    missing_pct=round(missing/(len(df)*len(df.columns))*100,2) if len(df)*len(df.columns) else 0; dup_pct=round(dup/len(df)*100,2) if len(df) else 0
    schemas=[{'name':x['name'],'data_type':x['dtype'],'missing_count':x['missing'],'missing_percentage':round(x['missing']/len(df)*100,2) if len(df) else 0,'unique_count':x['unique'],'sample_values':x['samples']} for x in schema]
    return {'rows':int(len(df)),'columns':int(len(df.columns)),'missing_cells':missing,'duplicate_rows':dup,'memory_bytes':mem,'schema':schema,'row_count':int(len(df)),'column_count':int(len(df.columns)),'total_missing_cells':missing,'missing_percentage':missing_pct,'total_duplicate_rows':dup,'duplicate_percentage':dup_pct,'memory_usage_bytes':mem,'column_schemas':schemas}

@app.get('/api/v1/health')
def health(): return {'status':'ok','timestamp':now()}
@app.get('/api/v1/config')
def config():
    active=bool(os.getenv('OPENAI_API_KEY'))
    return {'llm_provider':'openai' if active else 'deterministic_analytical_engine','llm_model':os.getenv('OPENAI_MODEL','gpt-4.1-mini') if active else 'rule-based-statistical-reasoning','llm_active':active,'database_url':'SQLite (Persisted)','ocr_engine':'Tesseract OCR (optional)','stt_engine':'Optional transcription adapter','max_upload_size_mb':50}
@app.get('/api/v1/sessions')
def sessions():
    with conn() as c: return [rowdict(r) for r in c.execute('SELECT * FROM sessions ORDER BY created_at DESC')]
@app.post('/api/v1/sessions')
def create_session(title:str='Analysis Session'):
    sid=str(uuid.uuid4()); t=now()
    with conn() as c: c.execute('INSERT INTO sessions VALUES(?,?,?,?)',(sid,title[:120],t,t))
    return {'id':sid,'title':title[:120],'created_at':t,'updated_at':t}
@app.delete('/api/v1/sessions/{sid}')
def delete_session(sid:str):
    with conn() as c:
        c.execute('DELETE FROM sessions WHERE id=?',(sid,)); c.execute('DELETE FROM files WHERE session_id=?',(sid,)); c.execute('DELETE FROM datasets WHERE session_id=?',(sid,)); c.execute('DELETE FROM workflows WHERE session_id=?',(sid,)); c.execute('DELETE FROM reports WHERE session_id=?',(sid,)); c.execute('DELETE FROM chats WHERE session_id=?',(sid,))
    return {'deleted':True}
@app.get('/api/v1/files')
def files(session_id:str):
    with conn() as c: rows=c.execute('SELECT * FROM files WHERE session_id=? ORDER BY created_at DESC',(session_id,)).fetchall()
    out=[]
    for r in rows:
        d=rowdict(r); d['extracted_metadata']=json.loads(d['extracted_metadata'] or '{}'); d['has_text_data']=bool(d['has_text_data']); d['size_bytes']=d['size']; d.pop('size',None); d.pop('stored_path',None); out.append(d)
    return out
@app.post('/api/v1/files/upload')
async def upload_files(session_id:str=Form(...), files:list[UploadFile]=File(...)):
    with conn() as c:
        if not c.execute('SELECT 1 FROM sessions WHERE id=?',(session_id,)).fetchone(): raise HTTPException(404,'Session not found')
    count=0
    for f in files:
        name=Path(f.filename or 'upload.bin').name; typ=classify(name)
        raw=await f.read()
        if len(raw)>50*1024*1024: raise HTTPException(413,f'{name} exceeds 50 MB')
        fid=str(uuid.uuid4()); target=UPLOADS/f'{fid}{Path(name).suffix.lower()}'
        target.write_bytes(raw); text=extract_text(target,typ); meta={'full_text':text,'ocr_text':text if typ=='image' else ''}
        if typ in ('csv','excel'):
            try:
                temp_df=read_df(target); meta.update({'row_count':int(len(temp_df)),'column_count':int(len(temp_df.columns))})
            except Exception: pass
        elif typ=='pdf':
            try:
                import pypdf
                meta['page_count']=len(pypdf.PdfReader(str(target)).pages); meta['total_words']=len(text.split())
            except Exception: pass
        status='processed' if typ!='unknown' else 'stored - unsupported format'
        with conn() as c:
            c.execute('INSERT INTO files VALUES(?,?,?,?,?,?,?,?,?,?)',(fid,session_id,name,str(target),typ,len(raw),status,json.dumps(meta),int(bool(text)),now()))
            if typ in ('csv','excel'):
                df=read_df(target)
                if df is not None:
                    did=str(uuid.uuid4()); work=EXPORTS/f'{did}.csv'; df.to_csv(work,index=False)
                    c.execute('INSERT INTO datasets VALUES(?,?,?,?,?,?,?)',(did,session_id,fid,name,str(target),str(work),now()))
        count+=1
    return {'uploaded_count':count}
@app.delete('/api/v1/files/{fid}')
def delete_file(fid:str):
    with conn() as c:
        r=c.execute('SELECT * FROM files WHERE id=?',(fid,)).fetchone()
        if not r: raise HTTPException(404,'File not found')
        c.execute('DELETE FROM files WHERE id=?',(fid,)); c.execute('DELETE FROM datasets WHERE file_id=?',(fid,))
    try: Path(r['stored_path']).unlink(missing_ok=True)
    except: pass
    return {'deleted':True}
@app.get('/api/v1/data/datasets')
def datasets(session_id:str):
    with conn() as c: rows=c.execute('SELECT d.id,d.session_id,d.file_id,d.name,d.created_at,f.original_filename FROM datasets d LEFT JOIN files f ON f.id=d.file_id WHERE d.session_id=? ORDER BY d.created_at DESC',(session_id,)).fetchall()
    out=[]
    for r in rows:
        item=rowdict(r); df=read_df(get_dataset(item['id'])['working_path']); item['row_count']=len(df); item['column_count']=len(df.columns); out.append(item)
    return out
@app.get('/api/v1/data/{did}/profile')
def dataset_profile(did:str):
    d=get_dataset(did); df=read_df(d['working_path'])
    if df is None: raise HTTPException(400,'Unsupported dataset format')
    return profile(df)
@app.get('/api/v1/data/{did}/preview')
def dataset_preview(did:str,page:int=1,page_size:int=20):
    d=get_dataset(did); df=read_df(d['working_path']); page=max(1,page); page_size=max(1,min(page_size,200)); start=(page-1)*page_size
    data=df.iloc[start:start+page_size].replace({np.nan:None}).to_dict(orient='records')
    schemas=profile(df)['column_schemas']
    return {'columns':[str(c) for c in df.columns],'rows':data,'page':page,'page_size':page_size,'total_rows':len(df),'total_pages':max(1,math.ceil(len(df)/page_size)),'column_schemas':schemas}

class CleanRequest(BaseModel): operation:str; parameters:dict={}
@app.post('/api/v1/cleaning/{did}/operations')
def clean_dataset(did:str, req:CleanRequest):
    d=get_dataset(did); df=read_df(d['working_path']); before=len(df); op=req.operation; p=req.parameters; summary=''; affected=0
    if op=='fill_missing':
        strategy=p.get('strategy','mean')
        for col in df.columns:
            if not df[col].isna().any(): continue
            if strategy=='forward_fill': df[col]=df[col].ffill().bfill()
            elif pd.api.types.is_numeric_dtype(df[col]):
                if strategy=='median': val=df[col].median()
                elif strategy=='mode': val=df[col].mode().iloc[0] if not df[col].mode().empty else 0
                else: val=df[col].mean()
                df[col]=df[col].fillna(val)
            else:
                mode=df[col].mode(); df[col]=df[col].fillna(mode.iloc[0] if not mode.empty else 'Unknown')
        affected=int(df.isna().sum().sum()); summary=f'Applied {strategy} imputation to missing values.'
    elif op=='drop_missing_rows':
        old=len(df); df=df.dropna(); affected=old-len(df); summary=f'Dropped {affected} rows containing missing values.'
    elif op=='drop_duplicates':
        old=len(df); df=df.drop_duplicates(); affected=old-len(df); summary=f'Removed {affected} duplicate rows.'
    elif op=='remove_outliers':
        numeric=df.select_dtypes(include=np.number).columns; mask=pd.Series(True,index=df.index)
        for col in numeric:
            q1,q3=df[col].quantile(.25),df[col].quantile(.75); iqr=q3-q1
            if pd.isna(iqr): continue
            lo,hi=q1-1.5*iqr,q3+1.5*iqr
            if p.get('action')=='winsorize': df[col]=df[col].clip(lo,hi)
            else: mask &= df[col].between(lo,hi) | df[col].isna()
        if p.get('action')!='winsorize': old=len(df); df=df[mask]; affected=old-len(df)
        summary='Winsorized numeric outliers using IQR.' if p.get('action')=='winsorize' else f'Filtered {affected} rows containing IQR outliers.'
    elif op=='normalize_text':
        case=p.get('case','lower')
        for col in df.select_dtypes(include=['object','string']).columns:
            s=df[col].astype('string').str.strip()
            df[col]=s.str.upper() if case=='upper' else s.str.title() if case=='title' else s.str.lower()
        summary=f'Trimmed whitespace and normalized text to {case} case.'; affected=len(df)
    else: raise HTTPException(400,'Unknown cleaning operation')
    write_df(df,d['working_path'])
    with conn() as c: c.execute('INSERT INTO cleaning(dataset_id,operation_type,rows_affected,summary,applied_at) VALUES(?,?,?,?,?)',(did,op,affected,summary,now()))
    return {'summary':summary,'rows_affected':affected,'rows_before':before,'rows_after':len(df)}
@app.get('/api/v1/cleaning/{did}/history')
def cleaning_history(did:str):
    with conn() as c: return [rowdict(r) for r in c.execute('SELECT * FROM cleaning WHERE dataset_id=? ORDER BY id DESC',(did,))]
@app.get('/api/v1/cleaning/{did}/export')
def export_cleaned(did:str,format:str='csv'):
    d=get_dataset(did); df=read_df(d['working_path']); fmt=format.lower()
    if fmt not in ('csv','xlsx'): raise HTTPException(400,'format must be csv or xlsx')
    out=EXPORTS/f'{did}_cleaned.{fmt}'; write_df(df,out)
    return FileResponse(out,filename=f'{Path(d["name"]).stem}_cleaned.{fmt}')

class ChartRequest(BaseModel): dataset_id:str; chart_type:str='bar'
@app.post('/api/v1/analytics/chart')
def chart(req:ChartRequest):
    d=get_dataset(req.dataset_id); df=read_df(d['working_path']); nums=df.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in df.columns if c not in nums]
    typ=req.chart_type; data=[]; layout={'title':f'{typ.title()} chart — {d["name"]}','paper_bgcolor':'white','plot_bgcolor':'white','margin':{'l':45,'r':20,'t':55,'b':60}}
    if typ=='heatmap':
        corr=df[nums].corr() if nums else pd.DataFrame(); data=[{'type':'heatmap','z':corr.fillna(0).values.tolist(),'x':corr.columns.tolist(),'y':corr.index.tolist(),'colorscale':'Blues'}]; layout['title']='Numeric Correlation Heatmap'
    elif typ=='histogram':
        col=nums[0] if nums else (df.columns[0] if len(df.columns) else None); data=[{'type':'histogram','x':df[col].dropna().tolist(),'name':str(col)}] if col else []
    elif typ=='box':
        col=nums[0] if nums else None; data=[{'type':'box','y':df[col].dropna().tolist(),'name':str(col)}] if col else []
    elif typ=='scatter':
        if len(nums)>=2: data=[{'type':'scatter','mode':'markers','x':df[nums[0]].tolist(),'y':df[nums[1]].tolist(),'name':f'{nums[0]} vs {nums[1]}'}]
        elif nums: data=[{'type':'scatter','mode':'markers','x':list(range(len(df))), 'y':df[nums[0]].tolist(),'name':str(nums[0])}]
    elif typ=='line':
        col=nums[0] if nums else (df.columns[0] if len(df.columns) else None); data=[{'type':'scatter','mode':'lines+markers','x':list(range(min(len(df),500))),'y':df[col].head(500).tolist(),'name':str(col)}] if col else []
    elif typ=='pie':
        col=cats[0] if cats else (df.columns[0] if len(df.columns) else None)
        if col:
            counts=df[col].astype(str).value_counts().head(12); data=[{'type':'pie','labels':counts.index.tolist(),'values':counts.values.tolist()}]
    else:
        col=cats[0] if cats else (nums[0] if nums else (df.columns[0] if len(df.columns) else None))
        if col:
            counts=df[col].astype(str).value_counts().head(20); data=[{'type':'bar','x':counts.index.tolist(),'y':counts.values.tolist(),'name':str(col)}]; layout['xaxis']={'title':str(col)}; layout['yaxis']={'title':'Count'}
    if not data: raise HTTPException(400,'This chart needs suitable columns in the selected dataset.')
    return {'plotly_data':data,'plotly_layout':layout}
class AnalysisRequest(BaseModel): dataset_id:str; analysis_type:str='hypothesis_test'; test_type:str='t_test_ind'; col1:str; col2:str
@app.post('/api/v1/analytics/execute')
def execute_analysis(req:AnalysisRequest):
    d=get_dataset(req.dataset_id); df=read_df(d['working_path']); a=df[req.col1]; b=df[req.col2]
    try:
        from scipy import stats
        if req.test_type=='t_test_ind':
            x=pd.to_numeric(a,errors='coerce').dropna(); y=pd.to_numeric(b,errors='coerce').dropna()
            if len(x)<2 or len(y)<2: raise HTTPException(400,'Welch t-test requires two numeric columns with at least two values each.')
            stat,p=stats.ttest_ind(x,y,equal_var=False); pooled=math.sqrt((x.var(ddof=1)+y.var(ddof=1))/2); effect=(x.mean()-y.mean())/pooled if pooled else 0; name="Welch's Two-Sample t-Test"
        elif req.test_type=='anova':
            x=pd.to_numeric(a,errors='coerce'); groups=[g.dropna().values for _,g in df.assign(_v=x).groupby(req.col2)['_v'] if len(g.dropna())>1]
            if len(groups)<2: raise HTTPException(400,'ANOVA requires a numeric variable and at least two groups in the second column.')
            stat,p=stats.f_oneway(*groups); effect=None; name='One-Way ANOVA'
        else:
            tab=pd.crosstab(a,b)
            if tab.shape[0]<2 or tab.shape[1]<2: raise HTTPException(400,'Chi-square requires two categorical columns with at least two categories each.')
            stat,p,_,_=stats.chi2_contingency(tab); effect=None; name='Chi-Square Independence Test'
    except ImportError: raise HTTPException(500,'Install scipy to run hypothesis tests.')
    sig=bool(p<.05)
    return {'test_name':name,'statistic':round(float(stat),6),'p_value':round(float(p),8),'is_significant':sig,'effect_size_cohens_d':round(float(effect),5) if effect is not None else None,'conclusion':'Statistically significant result at α=0.05; reject the null hypothesis.' if sig else 'No statistically significant result at α=0.05; do not reject the null hypothesis.'}

class ChatRequest(BaseModel): session_id:str; question:str; dataset_id:Optional[str]=None
@app.post('/api/v1/chat')
def chat(req:ChatRequest):
    q=req.question.strip(); pieces=[]; evidence=[]
    if req.dataset_id:
        d=get_dataset(req.dataset_id); df=read_df(d['working_path']); prof=profile(df)
        if any(w in q.lower() for w in ['missing','null','quality','clean']):
            pieces.append(f'Dataset **{d["name"]}** has {prof["rows"]} rows and {prof["columns"]} columns, {prof["missing_cells"]} missing cells, and {prof["duplicate_rows"]} duplicate rows.')
            pieces.append('Recommended actions: inspect columns with missing values, choose imputation based on column meaning, and review duplicates before deleting them.')
        elif any(w in q.lower() for w in ['correlation','correlate']):
            nums=df.select_dtypes(include=np.number)
            pieces.append('Numeric correlation matrix (Pearson):\n'+(nums.corr().round(3).to_string() if len(nums.columns)>1 else 'At least two numeric columns are required.'))
        elif any(w in q.lower() for w in ['summary','statistics','describe','key finding','analyse','analyze']):
            pieces.append(f'Profile for **{d["name"]}**: {prof["rows"]} rows, {prof["columns"]} columns, {prof["missing_cells"]} missing cells, {prof["duplicate_rows"]} duplicate rows.')
            if len(df.select_dtypes(include=np.number).columns): pieces.append('Descriptive statistics:\n'+df.describe(include='all').transpose().fillna('').round(3).to_string())
            for s in prof['schema']:
                if s['missing']: pieces.append(f'Column **{s["name"]}** has {s["missing"]} missing values.')
        else:
            pieces.append(f'Your selected dataset **{d["name"]}** contains {prof["rows"]} rows and {prof["columns"]} columns. Ask for a summary, missing values, duplicates, descriptive statistics, or correlations for evidence-based results.')
    with conn() as c:
        docs=c.execute('SELECT original_filename,extracted_metadata FROM files WHERE session_id=?',(req.session_id,)).fetchall()
    qterms=[t.lower() for t in q.split() if len(t)>2]
    for doc in docs:
        meta=json.loads(doc['extracted_metadata'] or '{}'); content=meta.get('full_text') or meta.get('ocr_text') or ''
        if content and any(t in content.lower() for t in qterms):
            idx=min([content.lower().find(t) for t in qterms if content.lower().find(t)>=0] or [0]); snippet=content[max(0,idx-100):idx+300]
            evidence.append({'filename':doc['original_filename'],'page_number':1,'snippet':snippet,'score':0.8})
    if evidence: pieces.append('Relevant uploaded document passages were found; see the supporting evidence below.')
    if not pieces: pieces=['I can answer using uploaded datasets and extracted document text. Upload a CSV/Excel dataset or a text-readable PDF/TXT file, then ask a specific question. No external LLM is assumed; calculations are performed locally.']
    answer='\n\n'.join(pieces)
    with conn() as c:
        c.execute('INSERT INTO chats(session_id,role,content,created_at) VALUES(?,?,?,?)',(req.session_id,'user',q,now())); c.execute('INSERT INTO chats(session_id,role,content,created_at) VALUES(?,?,?,?)',(req.session_id,'assistant',answer,now()))
    return {'content':answer,'supporting_evidence':evidence}
@app.delete('/api/v1/chat/{sid}')
def clear_chat(sid:str):
    with conn() as c: c.execute('DELETE FROM chats WHERE session_id=?',(sid,))
    return {'cleared':True}

class WorkflowRequest(BaseModel): session_id:str; dataset_id:Optional[str]=None; workflow_type:str='full_analytics'; user_query:str=''
@app.post('/api/v1/workflows/start')
def start_workflow(req:WorkflowRequest):
    if req.dataset_id: d=get_dataset(req.dataset_id); df=read_df(d['working_path']); p=profile(df); summary=f'Analyzed {d["name"]}: {p["rows"]} rows, {p["columns"]} columns, {p["missing_cells"]} missing cells, {p["duplicate_rows"]} duplicates.'
    else: p=None; summary='No tabular dataset selected; workflow checked uploaded files and session state.'
    records=[]
    for i,(name,desc) in enumerate(AGENTS,1):
        t=now(); out={'summary':desc}
        if i==2 and p: out.update({'missing_cells':p['missing_cells'],'duplicate_rows':p['duplicate_rows']})
        if i==3 and req.dataset_id:
            nums=df.select_dtypes(include=np.number); out['numeric_columns']=nums.columns.tolist(); out['descriptive_statistics']=nums.describe().round(3).fillna('').to_dict() if len(nums.columns) else {}
        if i==5: out['recommendations']=['Validate data definitions before interpreting results.','Review missing values and duplicates.','Use statistical significance alongside practical impact.']
        records.append({'agent_name':name,'started_at':t,'completed_at':now(),'status':'COMPLETED','outputs':out})
    wid=str(uuid.uuid4()); t=now()
    with conn() as c: c.execute('INSERT INTO workflows VALUES(?,?,?,?,?,?,?,?)',(wid,req.session_id,req.dataset_id,req.workflow_type,'completed',t,json.dumps(records),summary))
    return {'id':wid,'status':'completed','summary':summary,'execution_records':records}
@app.get('/api/v1/workflows/session/{sid}')
def workflow_list(sid:str):
    with conn() as c: return [rowdict(r) for r in c.execute('SELECT id,session_id,dataset_id,workflow_type,status,created_at,summary FROM workflows WHERE session_id=? ORDER BY created_at DESC',(sid,))]
@app.get('/api/v1/workflows/{wid}')
def workflow_detail(wid:str):
    with conn() as c: r=c.execute('SELECT * FROM workflows WHERE id=?',(wid,)).fetchone()
    if not r: raise HTTPException(404,'Workflow not found')
    d=rowdict(r); d['execution_records']=json.loads(d['execution_records']); return d

@app.post('/api/v1/reports/generate')
def generate_report(body:dict):
    sid=body.get('session_id'); did=body.get('dataset_id'); title=body.get('title','Analytics Report'); summary='Report generated for the current session.'; stats_text='No dataset selected.'
    if did:
        d=get_dataset(did); df=read_df(d['working_path']); p=profile(df); summary=f'{d["name"]}: {p["rows"]} rows, {p["columns"]} columns, {p["missing_cells"]} missing cells, {p["duplicate_rows"]} duplicate rows.'; stats_text=df.describe(include='all').transpose().fillna('').round(3).to_string()
    rid=str(uuid.uuid4()); out=REPORTS/f'{rid}.pdf'
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        styles=getSampleStyleSheet(); story=[Paragraph(title,styles['Title']),Spacer(1,12),Paragraph('Generated: '+now(),styles['Normal']),Spacer(1,10),Paragraph('Executive Summary',styles['Heading2']),Paragraph(summary.replace('&','&amp;'),styles['BodyText']),Spacer(1,10)]
        if did:
            story += [Paragraph('Dataset Profile',styles['Heading2']),Table([['Metric','Value'],['Rows',str(p['rows'])],['Columns',str(p['columns'])],['Missing cells',str(p['missing_cells'])],['Duplicate rows',str(p['duplicate_rows'])]],style=TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('GRID',(0,0),(-1,-1),.5,colors.grey),('PADDING',(0,0),(-1,-1),6)])),Spacer(1,12),Paragraph('Descriptive Statistics',styles['Heading2']),Preformatted(stats_text[:9000],styles['Code'])]
        story += [Spacer(1,12),Paragraph('Recommendations',styles['Heading2']),Paragraph('Review missingness and duplicates; verify column meaning before transformations; distinguish statistical significance from practical importance.',styles['BodyText'])]
        SimpleDocTemplate(str(out),pagesize=letter).build(story)
    except Exception as e: raise HTTPException(500,f'PDF generation failed: {e}')
    with conn() as c: c.execute('INSERT INTO reports VALUES(?,?,?,?,?,?,?)',(rid,sid,did,title,str(out),summary,now()))
    return {'id':rid,'title':title,'summary':summary,'created_at':now()}
@app.get('/api/v1/reports')
def reports(session_id:str):
    with conn() as c: return [rowdict(r) for r in c.execute('SELECT id,session_id,dataset_id,title,summary,created_at FROM reports WHERE session_id=? ORDER BY created_at DESC',(session_id,))]
@app.get('/api/v1/reports/{rid}/download')
def download_report(rid:str):
    with conn() as c: r=c.execute('SELECT * FROM reports WHERE id=?',(rid,)).fetchone()
    if not r: raise HTTPException(404,'Report not found')
    return FileResponse(r['path'],filename=f'{r["title"][:60]}.pdf',media_type='application/pdf')

# Serve the supplied frontend at root; keep API routes above static fallback.
app.mount('/', StaticFiles(directory=BASE/'frontend', html=True), name='frontend')
