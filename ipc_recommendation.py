import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

class IPCRecommenderKNN:
    def __init__(self, ipc_path='data/ipc_sections.csv', fir_path='data/FIR_DATASET.csv'):
        self.ipc_path = ipc_path
        self.fir_path = fir_path
        self.vectorizer = CountVectorizer(stop_words='english')
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.merged_data = None
        self.features = None

    def load_and_prepare_data(self):
        ipc_data = pd.read_csv(self.ipc_path)
        fir_data = pd.read_csv(self.fir_path)

        ipc_data.fillna('', inplace=True)
        fir_data.fillna('', inplace=True)

        merged_data = ipc_data.copy()
        if {'Cognizable', 'Bailable', 'Court'}.issubset(fir_data.columns):
            merged_data = merged_data.merge(
                fir_data[['Description', 'Cognizable', 'Bailable', 'Court']],
                on='Description',
                how='left'
            )

        merged_data.fillna('', inplace=True)
        merged_data['Combined_Text'] = (
            merged_data['Description'] + ' ' +
            merged_data['Offense'] + ' ' +
            merged_data['Punishment'] + ' ' +
            merged_data.get('Cognizable', '') + ' ' +
            merged_data.get('Bailable', '') + ' ' +
            merged_data.get('Court', '')
        )

        self.merged_data = merged_data
        self.features = self.vectorizer.fit_transform(merged_data['Combined_Text'])
        self.model.fit(self.features)

    def recommend(self, case_description, top_n=3):
        case_vector = self.vectorizer.transform([case_description])

        if case_vector.shape[1] != self.features.shape[1]:
            raise ValueError(f"Vectorizer mismatch: expected {self.features.shape[1]} features, got {case_vector.shape[1]}")

        distances, indices = self.model.kneighbors(case_vector, n_neighbors=top_n)

        recommendations = []
        for idx, dist in zip(indices[0], distances[0]):
            row = self.merged_data.iloc[idx]
            recommendations.append((row['Section'], row['Description'], 1 - dist))

        return recommendations


    def save_model(self, model_path='model/ipc_knn_model.joblib'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'merged_data': self.merged_data,
            'features': self.features
        }, model_path)


    def load_model(self, model_path='model/ipc_knn_model.joblib'):
        model = joblib.load(model_path)
        self.vectorizer = model['vectorizer']
        self.model = model['model']
        self.merged_data = model['merged_data']
        self.features = model['features']



recommender = IPCRecommenderKNN()
recommender.load_and_prepare_data()

# # Recommend sections for a test description
# description = "Murder of accused with a weapon"
# recommendations = recommender.recommend(description)
# print("Recommendations:")
# for section, desc, score in recommendations:
#     print(f"Section: {section} | Score: {score:.2f} | Desc: {desc}")

recommender.save_model('models/ipc_knn_model.joblib')

