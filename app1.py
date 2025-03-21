import joblib
import streamlit as st
import numpy as np

# Set page layout to wide
st.set_page_config(layout="wide")

# Title
st.title('📚 Book Recommender System - Find Your Next Favorite Book')

# Load the model and data
model = joblib.load(open('model.pkl', 'rb'))
books_name = joblib.load(open('books_name.pkl', 'rb'))
final_rating = joblib.load(open('final_rating.pkl', 'rb'))
book_pivot = joblib.load(open('book_pivot.pkl', 'rb'))

# Function to fetch book details
def fetch_book_details(book_title):
    book_info = final_rating[final_rating['Book-Title'] == book_title].iloc[0]
    return {
        'title': book_info['Book-Title'],
        'author': book_info['Book-Author'],
        'year': book_info['Year-Of-Publication'],
        'url': book_info['Image-URL-M']
    }

# Function to recommend books
def recommend_book(selected_book):
    book_id = np.where(book_pivot.index == selected_book)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=5)
    
    book_details = []
    for book_id in suggestion[0]:
        book_title = book_pivot.index[book_id]
        book_details.append(fetch_book_details(book_title))
    return book_details

# Book selection dropdown
selected_book = st.selectbox(
    "Search or select a book:",
    books_name
)

# Display selected book details
if selected_book:
    book = fetch_book_details(selected_book)
    st.subheader("Selected Book")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(book['url'], width=170)
    with col2:
        st.write(f"**Title:** {book['title']}")
        st.write(f"**Author:** {book['author']}")
        st.write(f"**Year:** {book['year']}")

# Show recommendations
if st.button('Show Recommendation'):
    recommended_books = recommend_book(selected_book)
    
    st.subheader("Recommended Books")
    cols = st.columns(4)  # Display 4 recommended books in a row

    for i, book in enumerate(recommended_books):
        with cols[i % 4]:
            st.image(book['url'], use_column_width=True)
            st.markdown(f"**{book['title']}**", unsafe_allow_html=True)
            st.write(f"by {book['author']}")
            st.write(f"Year: {book['year']}")

# CSS for styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #FF9900;
        color: white;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
        font-weight: bold;
        width: 100%;
        margin-top: 10px;
    }
    .stButton button:hover {
        background-color: #cc7a00;
        color: white;
    }
    .stImage img {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)