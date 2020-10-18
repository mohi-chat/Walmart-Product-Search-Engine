from flask import Flask, render_template, request, redirect
import pandas as pd
import search_engine
app = Flask(__name__)

#endpoint for search
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        query = request.form['product']
        data =[] 
        df = engine.search_query(query)
        sim = search_engine.check_similarity(df, query,engine.model_wiki)
        if(sim<threshold):
             df = search_engine.clustering_method(query,"combined_pickle.pickle",engine.model_wiki)
		# Iterate over each row 
        for index, rows in df.head(10).iterrows(): 
        	# Create list for the current row 
        	my_list =[rows['url'], rows['title'], rows['price']]
        	# append the list to the final list s
        	data.append(my_list)
        return render_template('search.html', data=data)
    return render_template('search.html')


if __name__ == '__main__':
    print("Loading search engine...")
    engine = search_engine.bm25()
    print("Loaded.")
    threshold = 0.35
    df = pd.read_csv('combined.csv')
    app.debug = True
    app.run()