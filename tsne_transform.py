from flask import Flask
from flask import request
from flask import render_template
import json
import numpy as np
from sklearn.cluster import KMeans
import sqlite3
from flask import g
import scipy

PATH = '/Users/christy/Downloads/_art'
DATABASE = PATH + '/db/tsne.db'
INTERVAL = 50
MAX_ITERATIONS = 100
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def start():
	return render_template('tsne2.html')

@app.route('/tsne', methods = ['POST'])
def raw_to_tsne():
    jsdata = request.form['pts']
    step = int(request.form['step'])
    data = np.array(json.loads(jsdata))
    # centroids = kmeans(data, 5)
    # print(centroids)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
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

#### OLD CODE ##################################
def kmeans(data, k, centroids=None):
    if not centroids:
        centroids = getRandomCentroids(data, k)
    iterations = 0
    old_centroids = None
    print('Running K means.')
    while not converge(old_centroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        old_centroids = centroids
        iterations += 1
        
        # Assign labels to each datapoint based on centroids
        labels = getLabels(data, centroids)
        
        # Assign centroids based on datapoint labels
        centroids, new_k = getCentroids(data, labels, k)
        k = new_k
    print('Converged.')
    return centroids

def getRandomCentroids(data, k):
    idx = np.random.choice(data.shape[0], k, replace=False)
    return data[idx]

def converge(old_centroids, centroids, iterations):
    # convergence test
    if not old_centroids: return False
    if (old_centroids and old_centroids.shape[0] != centroids.shape[0]) or iterations > MAX_ITERATIONS: 
        return True
    return (old_centroids == centroids) # TODO: or similar

def getLabels(data, centroids):
    res = []
    for pt in data:
    	distances = np.sqrt(((centroids-pt)**2).sum(axis=0))
    	res.append(np.argmin(distances))
    return res

# Returns k random centroids (k may be different)
def getCentroids(data, labels, k):
    # note: if a label is empty or contains only a few points, throw it out.
    data_by_labels = [[] for _ in range(k)]
    for ind in range(data.shape[0]):
        data_by_labels[labels[ind]].append(data[ind])
    # check if empty
    for pts in data_by_labels:
        if not pts:
            data_by_labels.remove(pts)
    data_by_labels = np.asarray(data_by_labels, dtype=np.float32)
    print('There are ', data_by_labels.shape[0], 'labels.')
    print('Shape of vector is ', data_by_labels.shape)
    print('Centroids array is', np.average(data_by_labels, axis=2))

    return np.average(data_by_labels, axis=1), data_by_labels.shape[0]