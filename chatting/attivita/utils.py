from textwrap import wrap
import json
from openai import OpenAI


class llm():
    '''Inizializza l'oggetto per le richieste al LLM.

    Parametri:
        max_tokens: il numero massimo di token per richiesta.
        temperature: la temperatura per la generazione del testo.
    '''
    def __init__(self, max_tokens=1000, temperature=0.7):
        self.completions = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed").chat.completions
        self.max_tokens = max_tokens
        self.temperature = temperature

    def request(self, prompt, stream=False):
        '''Richiede una risposta al modello.

        Parametri:
            prompt: la stringa di input per il modello.'''
        return self.completions.create(
            model="not-needed",
            messages=[
                {"role": "system", "content": "Be clear in your responses."},
                {"role": "user", "content": prompt}
            ],
            stream=stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    

def bprint(txt, width=100):
    '''Scrive il testo generato da un modello.

    Parametri:
        completion: un oggetto con il testo generato dal modello.
        width: la lunghezza massima di una riga.'''
    x = txt if type(txt) == str else txt.choices[0].message.content
    print('\n'.join(wrap(x, width, break_long_words=False)))


def jprint(txt, indent=2):
    '''Scrive il JSON generato da un modello.
    
    Parametri:
        completion: un oggetto con il testo generato dal modello.
        indent: il numero di spazi per indentazione.'''
    print(json.dumps(json.loads(txt.model_dump_json()), indent=indent))


def aprint(txt, width=100):
    '''Scrive il testo generato da un modello un token per volta
    (funziona con LMStudio, e non con GPT4All).

    Parametri:
        completion: un oggetto con il testo generato dal modello.
        width: la lunghezza massima di una riga.'''
    l = 0
    for chunk in txt:
        x = chunk.choices[0].delta.content
        if type(x) == str:
            print(x or "", end="")
            l += len(x)
            if l > width:
                print()
                l = 0
