import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df_clean', engine)

# load model
model = joblib.load("../models/pickle_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    
    # Q1
    # Get the number of Messages per Category
    Mesg_counts = df.iloc[:,5:].sum().sort_values(ascending=False)[1:11]
    Mesg_names = list(Mesg_counts.index)
    
    # Q2
    
    # Top 10 categories count
    top_cat_mean = df.iloc[:,4:].mean().sort_values()[1:11]
    top_cat_names = list(top_cat_mean.index)
    
    # Q3
    
    # Top 10 categories count
    category_names = df.iloc[:,4:].columns
    category_bool = (df.iloc[:,4:] != 0).sum().sort_values(ascending=False).values
       
    #############################
    

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # graph 2
        {
            'data': [
                Bar(
                    x=Mesg_names,
                    y=Mesg_counts,
                    orientation = 'v',
                    
                
                )
            ],
           
            'layout': {
                'title': 'Top 10 Number of Messages per Category',
                
                'xaxis': {
                    'title': "Number of Messages"
                    
                },
            }
        },
      # graph 3
        {
            'data': [
                Bar(
                    x=top_cat_mean,
                    y=top_cat_names,
                    orientation = 'h',
                )
            ],

            'layout': {
                'title': 'Top 10 Average Messages Categories',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Percentage"
                }
            }
        },
        # graph 4
        {
            'data': [
                Bar(
                    x= category_names,
                    y= category_bool
                )
            ],
            'layout': {
                'title': 'Distribution of Categories across Messages',
                'yaxis':{
                    'title':"Count"
                }, 
                'xaxis': {
                    'title':"Categories"
                    
                
                    
                }
            }
        }
       
      ########################
    ]
    
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()