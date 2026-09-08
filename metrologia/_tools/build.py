#!/usr/bin/env python3
"""Validate reviewed JSON inputs and generate static Jekyll-compatible reading pages.
Only Python's standard library is needed. Run from any directory; --check detects stale output.
"""
import argparse, hashlib, html, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'metrologia'
def esc(s):return html.escape(str(s),quote=True)
def read(name):return json.loads((BASE/'dati'/name).read_text())
def unique(items,name):
    ids=[i['id'] for i in items]
    if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate {name} IDs')
    if not all(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',i) for i in ids):raise ValueError(f'Invalid {name} ID')
    return {i['id']:i for i in items}
A=read('articoli.json');D=read('navigazione.json')
assert A['schema_version']==D['schema_version']==1
articles=unique(A['articles'],'article');concepts=unique(D['concepts'],'concept')
questions=unique(D['questions'],'question');relations=unique(D['relations'],'relation');paths=unique(D['paths'],'path')
blocks={}
for a in articles.values():
    blocks[a['id']]=unique(a['blocks'],'block')
    assert re.fullmatch(r'publ/t_m/t_m\d+\.pdf',a['pdf'])
    assert hashlib.sha256((ROOT/a['pdf']).read_bytes()).hexdigest()==a['source_sha256'],f'Source changed: {a["id"]}'
    for b in a['blocks']:
        assert b['type'] in ('paragraph','table','figure')
        assert 1<=b['page']<=a['pages']
        if b['type']=='paragraph': assert b['text'].strip()
        if b['type']=='figure':assert b['src'].startswith('data:image/') and b['alt']
        if b['type']=='table':assert b['rows'] and len(set(map(len,b['rows'])))==1

def ref(r):
    a=articles[r['article']];b=blocks[r['article']][r['block']];return a,b
for c in concepts.values():
    assert c['evidence']
    for r in c['evidence']:ref(r)
for q in questions.values():
    assert q['steps']
    for c in q['concepts']:assert c in concepts
    for step in q['steps']:ref(step['evidence'])
for r in relations.values():
    assert r['source'] in concepts and r['target'] in concepts
    assert r['basis'] in ('explicit','interpretive') and r['review_status'] in ('proposed','reviewed')
    assert r['evidence']
    for e in r['evidence']:ref(e)
for p in paths.values():
    assert p['steps']
    for s in p['steps']:assert s['article'] in articles

def shell(title,body,root,css):
    return f'''---
section: metrology
root: "{root}"
---
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{esc(title)} — percorsi di lettura di Metrologia Generale">
<title>{esc(title)} | Luca Mari</title>
<link rel="stylesheet" href="{root}mystyles.css">
<link rel="stylesheet" href="{css}">
</head>
<body>
<a class="mg-skip" href="#lettura">Vai al contenuto</a>
{{% include sidebar.html %}}
<div class="content mg"><main id="lettura">
{body}
</main></div>
</body></html>
'''
def source_link(r,prefix=''):
    a,b=ref(r)
    return f'<a href="{prefix}articoli/{a["id"]}.html#{b["id"]}">{esc(a["issue"])} · {esc(a["title"])} · p. {b["page"]}</a>'
def evidence_list(refs):return '<ul class="mg-sources">'+''.join('<li>'+source_link(r)+'</li>' for r in refs)+'</ul>'
def excerpt(r):
    a,b=ref(r)
    if b['type']=='paragraph':
        s=b['text'];cut=s if len(s)<=320 else s[:320].rsplit(' ',1)[0]+' […]'
        return '<blockquote>'+esc(cut)+'</blockquote>'
    return '<p class="mg-meta">'+esc(b.get('caption',b.get('alt','')))+'</p>'
def concept_links(ids,prefix=''):
    return ' · '.join(f'<a href="{prefix}#concetto-{i}">{esc(concepts[i]["label"])}</a>' for i in ids)

def index():
    out=['<p><a href="../mg.html">Metrologia Generale · elenco cronologico</a></p>',
    '<h1>Percorsi nella metrologia</h1>',
    '<p>Tre articoli, dal 2019 al 2025, da esplorare attraverso domande, concetti e collegamenti ai testi.</p>',
    '<nav class="mg-local" aria-label="Esplora la raccolta"><a href="#domande">Domande</a><a href="#concetti">Concetti</a><a href="#percorsi">Percorsi di lettura</a><a href="#articoli">Articoli</a></nav>',
    '<details class="mg-note"><summary>Una prima prova di lettura semantica</summary><p>'+esc(D['editorial']['note'])+'</p><p>La navigazione è interamente statica. Ogni collegamento rimanda a un passaggio della fonte. “Esplicito nel testo” indica il fondamento del collegamento; “Lettura trasversale” indica una relazione interpretativa proposta tra testi.</p></details>',
    '<section id="domande"><h2>Da quale domanda partire?</h2>']
    for q in questions.values():
        out.append(f'<article class="mg-entry" id="domanda-{q["id"]}"><h3>{esc(q["title"])}</h3><p>{esc(q["intro"])}</p><p class="mg-meta">{concept_links(q["concepts"])}</p><details><summary>Segui la domanda · {len(q["steps"])} passaggi</summary><ol>')
        for s in q['steps']:out.append('<li><p>'+esc(s['why'])+'</p>'+excerpt(s['evidence'])+source_link(s['evidence'])+'</li>')
        out.append('</ol></details></article>')
    out.append('</section><section id="concetti"><h2>Esplora i concetti</h2><div class="mg-grid">')
    for c in concepts.values():
        out.append(f'<article class="mg-entry" id="concetto-{c["id"]}"><h3>{esc(c["label"])}</h3><p>{esc(c["description"])}</p><p class="mg-meta">Termini collegati: {esc(", ".join(c["aliases"]))}</p><details><summary>Passaggi e relazioni</summary>'+evidence_list(c['evidence']))
        for r in relations.values():
            if c['id'] not in (r['source'],r['target']):continue
            label='Esplicito nel testo' if r['basis']=='explicit' else 'Lettura trasversale'
            state='Da rivedere con l’autore' if r['review_status']=='proposed' else 'Revisionato'
            out.append('<div class="mg-relation"><p><strong>'+esc(concepts[r['source']]['label'])+'</strong> '+esc(r['predicate'])+' '+concept_links([r['target']])+f'</p><p>{esc(r["explanation"])}</p><p class="mg-meta">{label} · {state}</p>'+evidence_list(r['evidence'])+'</div>')
        out.append('</details></article>')
    out.append('</div></section><section id="percorsi"><h2>Percorsi di lettura</h2>')
    for p in paths.values():
        out.append(f'<article class="mg-entry" id="percorso-{p["id"]}"><h3>{esc(p["title"])}</h3><p>{esc(p["description"])}</p><ol>')
        for s in p['steps']:
            a=articles[s['article']];out.append(f'<li><a href="articoli/{a["id"]}.html">{esc(a["title"])}</a><p>{esc(s["why"])}</p></li>')
        out.append('</ol></article>')
    out.append('</section><section id="articoli"><h2>Gli articoli della prova</h2><ul>')
    for a in articles.values():out.append(f'<li>{esc(a["issue"])} · <a href="articoli/{a["id"]}.html">{esc(a["title"])}</a> · <a href="../{a["pdf"]}">PDF</a><p class="mg-meta">{esc(", ".join(a["authors"]))}</p></li>')
    out.append('</ul></section>')
    return shell('Percorsi nella metrologia','\n'.join(out),'../','assets/metrologia.css')

def article(a):
    aid=a['id'];related=[c['id'] for c in concepts.values() if any(r['article']==aid for r in c['evidence'])]
    out=[f'<p><a href="../index.html">Percorsi nella metrologia</a> · <a href="../../mg.html">Elenco cronologico</a></p><h1 id="p1-01">{esc(a["title"])}</h1>',f'<p class="mg-meta">{esc(", ".join(a["authors"]))} · Tutto_Misure {esc(a["issue"])}</p>',f'<p><a href="../../{a["pdf"]}">Leggi il PDF originale</a></p>', '<p class="mg-meta">Testo integrale del preprint con impaginazione adattata alla lettura web. I riferimenti temporali e le affermazioni sono quelli dell’articolo originale.</p>',f'<aside class="mg-note" aria-label="Collegamenti di lettura"><p>Concetti: {concept_links(related,"../index.html")}</p><p>Domande: '+ ' · '.join(f'<a href="../index.html#domanda-{q["id"]}">{esc(q["title"])}</a>' for q in questions.values() if any(s['evidence']['article']==aid for s in q['steps']))+'</p></aside>']
    lastpage=0
    for b in a['blocks']:
        if b['id']=='p1-01':continue
        pn=b['page']
        if pn!=lastpage:
            if lastpage:out.append('</section>')
            out.append(f'<section class="mg-page" id="pagina-{pn}" aria-label="Pagina {pn} del PDF"><p class="mg-page-label"><a href="../../{a["pdf"]}#page={pn}">Pagina {pn} del PDF</a></p>');lastpage=pn
        id=b['id']
        if b['type']=='paragraph':out.append(f'<p class="mg-passage" id="{id}">{esc(b["text"])} <a class="mg-anchor" href="#{id}" aria-label="Collegamento a questo passaggio">¶</a></p>')
        elif b['type']=='figure':out.append(f'<figure id="{id}"><img src="{b["src"]}" alt="{esc(b["alt"])}" loading="lazy"><figcaption>{esc(b["alt"])}</figcaption></figure>')
        elif b['type']=='table':
            out.append(f'<div class="mg-table" id="{id}" role="region" aria-label="{esc(b["caption"])}" tabindex="0"><table><caption>{esc(b["caption"])}</caption>')
            for i,row in enumerate(b['rows']):
                out.append('<tr>')
                for j,cell in enumerate(row):
                    tag='th' if (i==0 and b['header']) or j==0 else 'td';scope=' scope="col"' if i==0 and b['header'] else (' scope="row"' if j==0 else '')
                    out.append(f'<{tag}{scope}>{esc(cell)}</{tag}>')
                out.append('</tr>')
            out.append('</table></div>')
    out.extend(['</section>','<p><a href="../index.html">Torna ai percorsi nella metrologia</a></p>'])
    return shell(a['title'],'\n'.join(out),'../../','../assets/metrologia.css')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    generated={BASE/'index.html':index(),**{BASE/'articoli'/f'{a["id"]}.html':article(a) for a in articles.values()}}
    for path,content in generated.items():
        if args.check:assert path.exists() and path.read_text()==content,f'Outdated output: {path}'
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    print(f'OK: {len(articles)} articles, {sum(len(a["blocks"]) for a in articles.values())} blocks, {len(concepts)} concepts, {len(questions)} questions, {len(relations)} relations, {len(paths)} paths; HTML '+('verified' if args.check else 'generated'))
