# Metrologia: esperimento di navigazione semantica

La pagina `index.html` e gli articoli HTML sono statici: la consultazione non richiede chatbot, API, JavaScript o servizi di IA. Le pagine usano lo stile e l'include di navigazione del sito Jekyll esistente. L'ingresso è anche collegato da `mg.html`.

## Contenuto della prima prova

- Tre preprint: Tutto_Misure 2019.2 (Mari, Narduzzi), 2022.2 (Mari, Ferrero, Petri), 2025.4 (Mari, Petri).
- Tre domande, sei concetti, cinque relazioni motivate, due percorsi.
- Testi integrali, tre figure estratte dal PDF, tabelle HTML e collegamenti a pagina e passaggio. La tabella del 2019 mantiene la divisione tra pagine dell'originale.
- Sintesi e collegamenti sono proposte editoriali da rivedere con l'autore. `explicit` indica che il collegamento è affermato nel testo, non che sia già stato approvato. `interpretive` indica una lettura trasversale proposta.

## File sorgente e file generati

`dati/articoli.json` conserva metadati bibliografici, impronta SHA-256 del PDF, pagine e blocchi della trascrizione. I blocchi hanno ID persistenti; i tipi sono `paragraph`, `table` e `figure`. Le figure sono incorporate come data URI per mantenere l'esperimento autonomo.

`dati/navigazione.json` contiene concetti (con sinonimi), domande con tappe motivate, relazioni tipizzate con fonti e stato editoriale, percorsi ordinati. Ogni evidenza identifica un articolo e un blocco. La pagina PDF si ricava dal blocco, così non esistono due numerazioni da mantenere separatamente.

`dati/schema-*.json` descrivono il formato degli input. `schema_version` vale 1. Gli ID degli articoli (`tm-18`, ecc.) derivano dai nomi dei PDF; le altre chiavi sono identificatori editoriali, indipendenti dai titoli visibili.

`_tools/build.py` valida integrità, riferimenti e impronte delle fonti, quindi genera `index.html` e `articoli/*.html`. Servono soltanto Python 3 e i PDF già presenti nel repository. La directory `_tools` non è pubblicata da Jekyll perché inizia con underscore. Non modificare direttamente gli HTML generati.

Dalla radice del repository:

```bash
python3 metrologia/_tools/build.py
python3 metrologia/_tools/build.py --check
bash preview.sh
```

Aprire `/metrologia/index.html` nel server locale. La generazione deve avvenire prima del normale build Jekyll; GitHub Pages serve gli HTML già generati e non deve eseguire Python.

## Aggiungere un articolo

1. Inserire il PDF nell'archivio usuale e aggiungerlo all'elenco cronologico.
2. Trascrivere il testo in un nuovo record di `articoli.json`, includendo titolo, tutti gli autori, numero della rivista, percorso PDF, SHA-256 e numero di pagine. Estrarre e controllare anche figure, formule e tabelle. `_tools/convert_pdfs.py` documenta l'importazione della prova (richiede PyMuPDF); non è un convertitore universale.
3. Assegnare nuovi ID ai blocchi e conservarli nelle revisioni: inserire nuovi blocchi con nuovi ID senza rinumerare quelli esistenti. La pagina di provenienza può essere aggiornata senza cambiare l'ID.
4. Collegare il nuovo articolo a concetti, domande, relazioni e percorsi pertinenti in `navigazione.json`. Ogni collegamento deve avere una motivazione e fonti puntuali. L'IA può proporre l'apparato, ma non dichiarare svolta la revisione dell'autore.
5. Revisionare testo, citazioni, interpretazioni e coautori. Aggiornare la data dell'apparato. Impostare `review_status: reviewed` solo sulle relazioni effettivamente riviste; `editorial.author_reviewed` riguarda l'intero apparato.
6. Rigenerare, controllare i riferimenti, confrontare la conversione con il PDF e provare il sito con Jekyll. Commettere insieme JSON e HTML generati.

## Aggiungere una chiave di lettura

Riesaminare tutti gli articoli della prova rispetto alla nuova domanda o al nuovo concetto. Aggiungere un ID nuovo, una spiegazione e le evidenze; inserire eventuali relazioni e tappe. Non è necessario riconvertire i PDF o cambiare il codice dell'interfaccia. Rigenerare con `build.py`.

## Criteri editoriali e limiti

- Le sintesi e le descrizioni appartengono all'apparato; non sono presentate come citazioni degli autori. Gli estratti sono copiati dai blocchi e le abbreviazioni sono segnalate da `[…]`.
- Il testo originale è conservato, compresi riferimenti storici, formulazioni e possibili refusi. Non è una revisione scientifica delle affermazioni degli articoli.
- L'impaginazione è adattata al web: gli a capo sono ricomposti, i blocchi restano legati alle pagine del PDF; grassetti, corsivi e notazioni in apice/pedice non sono riprodotti sistematicamente. Per l'impaginazione originale resta disponibile il PDF.
- La formula incorporata come immagine nel testo del 2025 è trascritta `y = k₀ + Σ kᵢ zᵢ`. Le figure conservano l'immagine originale; le descrizioni alternative e le didascalie HTML sono editoriali.
- Nella tabella del 2025 i tre esiti (corretto, indecisione, errore) sono percentuali sul totale; non confondere la riga “accuratezza” con l'accuratezza condizionata ai soli casi classificati.
- Non si inferisce una genealogia delle idee dalla sola successione cronologica degli articoli.
- `convert_pdfs.py --replace` sovrascrive la trascrizione iniziale e può cambiare gli ID: è consentito soltanto se si ricontrollano tutte le evidenze e si preservano i vecchi permalink. Per aggiornamenti ordinari modificare i JSON e usare `build.py`.
