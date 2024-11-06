from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

file_path = 'imdb-movies-dataset.csv'
movies_df = pd.read_csv(file_path)
for column in ['Genre', 'Title' , 'Director']:
    movies_df[column] = movies_df[column].fillna('')


movies_df['soup'] = (
        movies_df['Genre'] + ' ' +
        movies_df['Title'] + ' ' +
        movies_df['Director']
)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies_df.index, index=movies_df['Title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=movies_df, indices=indices):
    if title not in indices:
        return ["Movie not found. Please try another title."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommendations = get_recommendations(movie_title)
    return render_template('homepage.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
