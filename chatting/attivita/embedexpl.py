''' Un'esplorazione del word embedding
Luca Mari, marzo 2024

Un'applicazione Flask che permette di esplorare il word embedding.

Installare i moduli Python richiesti, eseguendo dal terminale:  
    pip install torch transformers multimethod flask networkx

È un'applicazione Flask: occorre anche embedtempl.html, che usa d3.js scaricandolo dal web.
Una volta avviata l'applicazione, aprire un browser e digitare l'indirizzo:
    http://127.0.0.1:5000/
'''

from embedutils import Model
from flask import Flask, request, render_template
import networkx as nx

model = Model('dbmdz/bert-base-italian-xxl-cased', True)

app = Flask(__name__, template_folder='.')
G = nx.DiGraph()

@app.route('/')
def index():
    global G
    G.clear()
    default_token = 'cosa'
    token = request.args.get('token')
    if not token: token = default_token
    if model.token_in_vocab(token):
        code = 0
    else:
        token = default_token
        code = -1
    number = request.args.get('number')
    if not number: number = 10
    lista = model.most_similar(token, top_n=int(number), filter=True)
    for i in range(len(lista)):
        G.add_node(lista[i][0], sim=str(lista[i][1]))
        G.add_edge(token, lista[i][0])
    return render_template('embedtempl.html', token=token, number=number, code=code)

@app.route('/data')
def data():
    data = nx.node_link_data(G)
    return data

if __name__ == '__main__':
    app.run(debug=True)
