import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import string
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Memuat model
rf = joblib.load('Hotel_Review_Analysis_Model.pkl')

# Fungsi untuk melakukan prediksi sentimen
def prepare_predict_data(str_predict, score_review):
    
    predict_df = pd.DataFrame({"review": [str_predict], "Reviewer_Score":[score_review]})
    predict_df["is_bad_review"] = predict_df["Reviewer_Score"].apply(lambda x: 1 if x < 5 else 0)
    predict_df = predict_df[["review", "is_bad_review"]]
    predict_df["review"] = predict_df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    
    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    

    def clean_text(text):
        text = text.lower()
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        text = [word for word in text if not any(c.isdigit() for c in word)]
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        text = [t for t in text if len(t) > 0]
        pos_tags = pos_tag(text)
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        text = [t for t in text if len(t) > 1]
        text = " ".join(text)
        return(text)

    predict_df["review_clean"] = predict_df["review"].apply(lambda x: clean_text(x))
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
    predict_df["sentiments"] = predict_df["review"].apply(lambda x: sid.polarity_scores(x))
    predict_df = pd.concat([predict_df.drop(['sentiments'], axis=1), predict_df['sentiments'].apply(pd.Series)], axis=1)
    
    predict_df["nb_chars"] = predict_df["review"].apply(lambda x: len(x))

    predict_df["nb_words"] = predict_df["review"].apply(lambda x: len(x.split(" ")))
    
    from gensim.test.utils import common_texts
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(predict_df["review_clean"].apply(lambda x: x.split(" ")))]


    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)


    doc2vec_df = predict_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    predict_df = pd.concat([predict_df, doc2vec_df], axis=1)
    
    predict_df["is_bad_review"].value_counts(normalize = True)
    
    label = "is_bad_review"
    ignore_cols = [label, "review", "review_clean"]
    features = [c for c in predict_df.columns if c not in ignore_cols]

    x_predict = predict_df[features]
    
    result = rf.predict(x_predict)[0]
    return result

st.title('Hotel Review Sentiment Prediction')

comment = st.text_area('Masukkan komentar:')
rating = st.slider('Pilih skor bintang (0-5)', 0, 5, 3)

if st.button('Prediksi Sentimen'):
    sentiment = prepare_predict_data(comment, rating)
    if sentiment:
        st.write(f'prediksi saya, review itu memiliki sentimen negatif!')
    else:
        st.write(f'prediksi saya, review itu memiliki sentimen positif!')
    

