import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Intelligent Digital Library",
    page_icon="📚",
    layout="wide"
)

# Sample document dataset
documents = [
    {
        "title": "Introduction to Machine Learning",
        "author": "John Smith",
        "category": "Machine Learning",
        "year": 2023,
        "description": (
            "Machine learning studies algorithms that learn "
            "from data for prediction and classification."
        )
    },
    {
        "title": "Deep Learning for Medical Imaging",
        "author": "Sarah Johnson",
        "category": "Deep Learning",
        "year": 2024,
        "description": (
            "Deep neural networks are used for medical image "
            "classification, disease detection, and diagnosis."
        )
    },
    {
        "title": "Natural Language Processing Fundamentals",
        "author": "David Brown",
        "category": "NLP",
        "year": 2022,
        "description": (
            "Natural language processing helps computers "
            "understand text, language, and human communication."
        )
    },
    {
        "title": "Information Retrieval Systems",
        "author": "Emily Wilson",
        "category": "Information Retrieval",
        "year": 2023,
        "description": (
            "Information retrieval systems search document "
            "collections and rank relevant search results."
        )
    },
    {
        "title": "Artificial Intelligence in Healthcare",
        "author": "Michael Davis",
        "category": "Artificial Intelligence",
        "year": 2025,
        "description": (
            "Artificial intelligence supports healthcare through "
            "medical diagnosis, patient data analysis, and prediction."
        )
    },
    {
        "title": "Text Mining and Document Classification",
        "author": "Priya Sharma",
        "category": "NLP",
        "year": 2024,
        "description": (
            "Text mining uses feature extraction and classification "
            "to discover useful patterns in text documents."
        )
    },
    {
        "title": "Computer Vision Fundamentals",
        "author": "Robert Lee",
        "category": "Deep Learning",
        "year": 2021,
        "description": (
            "Computer vision enables computers to analyze images "
            "using image processing and neural networks."
        )
    },
    {
        "title": "Data Science with Python",
        "author": "Anita Patel",
        "category": "Data Science",
        "year": 2025,
        "description": (
            "Python supports data analysis, visualization, "
            "statistical modeling, and machine learning."
        )
    }
]

df = pd.DataFrame(documents)

# Prepare searchable document text
df["search_text"] = (
    df["title"].fillna("") + " " +
    df["author"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["description"].fillna("")
)

# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(df["search_text"])

# Interface
st.title("📚 Intelligent Digital Library Search")
st.write(
    "Search books and research papers using NLP "
    "and TF-IDF-based document ranking."
)

st.sidebar.header("Search Settings")

category_options = ["All"] + sorted(df["category"].unique())
selected_category = st.sidebar.selectbox(
    "Select category",
    category_options
)

top_k = st.sidebar.slider(
    "Number of results",
    min_value=1,
    max_value=len(df),
    value=5
)

query = st.text_input(
    "Enter your search query",
    placeholder="e.g. deep learning for medical image classification"
)

if query.strip():
    # Transform query using the same vectorizer
    query_vector = vectorizer.transform([query])

    # Compute similarity scores
    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = df.copy()
    results["similarity"] = scores

    # Apply category filter
    if selected_category != "All":
        results = results[
            results["category"] == selected_category
        ]

    # Sort by relevance
    results = results.sort_values(
        "similarity",
        ascending=False
    )

    # Remove documents with zero similarity
    results = results[results["similarity"] > 0]

    results = results.head(top_k)

    st.subheader("Search Results")
    st.caption(f"Found {len(results)} relevant document(s).")

    if results.empty:
        st.warning(
            "No matching documents found. Try different keywords."
        )
    else:
        for _, result in results.iterrows():
            with st.container(border=True):
                st.markdown(f"### {result['title']}")
                st.write(
                    f"**Author:** {result['author']}  |  "
                    f"**Year:** {result['year']}"
                )
                st.write(f"**Category:** {result['category']}")
                st.write(result["description"])
                st.progress(
                    min(float(result["similarity"]), 1.0),
                    text=(
                        f"Cosine similarity: "
                        f"{result['similarity']:.2%}"
                    )
                )

else:
    st.info(
        "Enter a query above to discover relevant books "
        "and research papers."
    )

with st.expander("View complete document collection"):
    st.dataframe(
        df.drop(columns=["search_text"]),
        use_container_width=True
    )
