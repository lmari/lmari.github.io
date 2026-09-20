#!/usr/bin/env python3
"""One-time import of the pilot PDFs; preserves existing articles unless --replace is explicit.
Requires PyMuPDF. Review imported text, figures and tables before accepting a new article.
"""
import argparse, base64, hashlib, html, json, re
from pathlib import Path
import fitz
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'metrologia/dati/articoli.json'
SOURCES = [
 (18, '2019.2', 'Verso un’incertezza di classificazione – La cultura metrologica nella valutazione delle proprietà classificatorie', ['Luca Mari','Claudio Narduzzi']),
 (30, '2022.2', 'Le scale di misura: un ponte tra mondo empirico e mondo dell’informazione', ['Luca Mari','Alessandro Ferrero','Dario Petri']),
 (43, '2025.4', 'Riflessioni sulla caratterizzazione metrologica dei sistemi di Machine Learning per la classificazione – Seconda parte', ['Luca Mari','Dario Petri'])]
FIGURES = {(43,2,0): 'Dieci immagini di cifre scritte a mano dal dataset MNIST, con le rispettive etichette.', (43,3,1): 'Un neurone con 784 ingressi, coefficienti k e termine di bias k₀.', (43,3,2): 'Rete senza strati interni: ciascuno dei 784 ingressi è collegato ai dieci valori di uscita.'}
def clean(s): return re.sub(r'\s+', ' ', s).strip()
def text_blocks(s):
    # Keep all words and punctuation; reflow line wraps, retaining blank-line boundaries.
    for chunk in re.split(r'\n\s*\n', s):
        lines=chunk.strip().splitlines(); para=[]
        for line in lines:
            line=line.strip()
            if not line: continue
            if para and (line.startswith('– ') or re.search(r'[.!?:;][”»"]?$',para[-1])):
                yield clean(' '.join(para)); para=[]
            para.append(line)
        if para: yield clean(' '.join(para))
def convert():
    articles=[]
    for n,issue,title,authors in SOURCES:
        src=ROOT/f'publ/t_m/t_m{n}.pdf'; doc=fitz.open(src); blocks=[]
        for pn,page in enumerate(doc,1):
            events=[]
            for table in page.find_tables().tables:
                events.append((table.bbox[1],table.bbox[3],'table',table.extract()))
            images=[b for b in page.get_text('dict')['blocks'] if b['type']==1]
            for k,b in enumerate(images):
                if n==43 and pn==3 and k==0: continue # Inline equation, transcribed below.
                events.append((b['bbox'][1],b['bbox'][3],'figure', (b,k)))
            y=0; seq=0
            for top,bottom,kind,value in sorted(events)+[(page.rect.height,page.rect.height,'end',None)]:
                text=page.get_text('text',clip=fitz.Rect(0,y,page.rect.width,top),sort=True)
                if n==43 and pn==3:
                    text=re.sub(r'(parametri ki,)\s*,',r'\1 y = k₀ + Σ kᵢ zᵢ,',text)
                for t in text_blocks(text):
                    seq+=1; blocks.append({'id':f'p{pn}-{seq:02d}','page':pn,'type':'paragraph','text':t})
                if kind=='table':
                    seq+=1
                    if n==43:
                        value[1][0]='w₁ (probabilità minima della moda)';value[2][0]='w₂ (differenza minima tra le due probabilità più grandi)'
                    blocks.append({'id':f'p{pn}-{seq:02d}','page':pn,'type':'table','rows':[[clean(c or '') for c in row] for row in value], 'caption': 'Confronto tra misurazione e classificazione'+(' (continua)' if pn==2 else '') if n==18 else 'Scenari di classificazione: risultati riportati nell’articolo', 'header':not(n==18 and pn==2)})
                elif kind=='figure':
                    b,k=value; seq+=1
                    blocks.append({'id':f'p{pn}-{seq:02d}','page':pn,'type':'figure','src':f'data:image/{b["ext"]};base64,'+base64.b64encode(b['image']).decode(),'alt':FIGURES[(n,pn,k)]})
                y=bottom
        articles.append({'id':f'tm-{n}','issue':issue,'title':title,'authors':authors,'pdf':f'publ/t_m/t_m{n}.pdf','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'pages':len(doc),'blocks':blocks})
    OUT.write_text(json.dumps({'schema_version':1,'articles':articles},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replace',action='store_true');args=ap.parse_args()
    if OUT.exists() and not args.replace: ap.error('Import already exists. Use --replace only after checking stable passage IDs and manual corrections.')
    convert()
