import streamlit as st

# Import your model class
from ipc_recommendation import IPCRecommenderKNN

st.set_page_config(page_title="IPC Section Recommender", layout="centered")

st.title("🔍 IPC Section Recommender")

description = st.text_area("Enter FIR/Case Description:")
if st.button("Get Recommendations"):
    with st.spinner("Loading model and predicting..."):
        recommender = IPCRecommenderKNN()
        recommender.load_model('models/ipc_knn_model.joblib')
        recommendations = recommender.recommend(description)
        st.success("Top Matches:")
        for section, desc, score in recommendations:
            st.markdown(f"**Section {section}**\n\n_{desc}_\n\n**Score**: {score:.2f}")
            st.markdown("---")
