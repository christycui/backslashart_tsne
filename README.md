### How to run T-SNE processes

1. Start the application: `FLASK_APP=tsne_transform.py flask run`
2. Reset the database: go to localhost:5000/reset_db
3. Run the process and write the output to database: go to localhost:5000 and click on the start button at the bottom right of the page

Notes: 
- It takes time to run T-SNE, so things will show up a little while after you press start.
- If you want to stop the calculation in the middle, you can press the stop button right next to the start button. It won't resume until you press start again.

### Endpoints
- GET '/': main page of the client
- GET '/tsne_clusters' with param 'step': returns the 3D coordinates of the cluster centers at a specified step
Note: It retrieves from the database, so make sure the step you requested is populated in the database.
- GET '/reset_db': resets the database
Note: It rewrites/sets up the database.

### Dependencies
- Python 3.6.7
- flask 0.12.2
- sqlite3 3.25.2
- sklean 0.19.1
- numpy 1.14.2
- scipy 1.0.0
- json 2.0.9