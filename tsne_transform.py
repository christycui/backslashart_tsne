from flask import Flask
from flask import request
from flask import render_template
import json
import numpy as np
from sklearn.cluster import KMeans
import sqlite3
from flask import g
import scipy

with open('config.json', 'r') as f:
    config = json.load(f)

PATH = config['path']
DATABASE = PATH + '/db/tsne.db'
INTERVAL = config['interval']
NUM_CLUSTERS = config['num_clusters']
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def start():
	return render_template('tsne2.html')

@app.route('/tsne', methods = ['POST'])
def raw_to_tsne():
    jsdata = request.form['pts']
    step = int(request.form['step'])
    data = np.array(json.loads(jsdata))
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(data)
    centers = kmeans.cluster_centers_.tolist()
    # reorder according to distance
    if step != INTERVAL:
        old_centers = json.loads(
            query_db('SELECT * FROM tsne WHERE step = ?;',
                [step - INTERVAL], one=True)[1])
        centers = mapping(centers, old_centers)
    centers_json = json.dumps(centers)
    # write to db
    row_id = insert('tsne', ['step', 'coords'], [step, centers_json])
    print('Row id: ', row_id)
    return centers_json

@app.route('/tsne_clusters', methods = ['GET'])
def get_tsne():
    step = request.args.get('step')
    row = query_db('SELECT * FROM tsne WHERE step = ?',
        [int(step)], one=True)
    return row[1]

@app.route('/reset_db', methods = ['GET'])
def init_db():
    db = get_db()
    with app.open_resource(PATH + '/db/schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    return 'recreated table'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def insert(table, fields=(), values=()):
    db = get_db()
    cur = db.cursor()
    query = 'INSERT INTO %s (%s) VALUES (%s)' % (
        table,
        ', '.join(fields),
        ', '.join(['?'] * len(values))
    )
    cur.execute(query, values)
    db.commit()
    id = cur.lastrowid
    cur.close()
    return id

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def mapping(centers, old_centers):
    res = [None for _ in range(len(centers))]
    for pt in centers:
        distances = scipy.spatial.distance.cdist([pt], old_centers)
        ind = np.argmin(distances)
        old_centers[ind] = [float('inf'), float('inf'), float('inf')]
        res[ind] = pt
    return res