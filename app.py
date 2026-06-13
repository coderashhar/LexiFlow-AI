import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure the Streamlit page
st.set_page_config(
    page_title="TextFlow AI",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for a modern, premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput>div>div>input {
        background-color: #1e2532;
        color: #ffffff;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        color: white;
    }
    .prediction-box {
        background-color: #1e2532;
        border-left: 4px solid #6366f1;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 1.2rem;
        color: #e2e8f0;
        line-height: 1.6;
    }
    .highlight {
        color: #818cf8;
        font-weight: 700;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    """Load the model and tokenizer to prevent reloading on every interaction."""
    model = load_model('lstm_model.h5')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open('max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
        
    return model, tokenizer, max_len

def predictor(model, tokenizer, text, max_len, index_to_word):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    # Pad sequences exactly as done during training
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    
    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    
    # Return the predicted word or an empty string if index 0
    return index_to_word.get(pred_index, "")

def generate_text(model, tokenizer, seed_text, max_len, n_words, index_to_word):
    current_text = seed_text
    generated_words = []
    
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, current_text, max_len, index_to_word)
        if not next_word:
            break
        generated_words.append(next_word)
        current_text += " " + next_word
        
    return generated_words

# --- MAIN APP UI ---
st.title("🔮 TextFlow AI")
st.markdown("### *AI-Powered Next Word Prediction*")

st.markdown("Type a few words below, and let TextFlow AI guess what comes next. Adjust the slider if you want it to generate an entire phrase!")

# Load model and assets
try:
    with st.spinner('Loading AI Model...'):
        model, tokenizer, max_len = load_assets()
        # Create the reverse dictionary for predictions
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}
except Exception as e:
    st.error(f"Error loading the model files. Make sure `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` are present. ({e})")
    st.stop()

# Input area
seed_text = st.text_input("Enter your starting text:", placeholder="It is our...", key="user_input")

# Generation controls
col1, col2 = st.columns([3, 1])
with col1:
    n_words = st.slider("Words to predict:", min_value=1, max_value=20, value=3, help="How many words should the AI generate after your prompt?")
with col2:
    st.markdown("<br>", unsafe_allow_html=True) # spacer
    predict_btn = st.button("Predict 🚀")

# Action
if predict_btn or (seed_text and n_words):
    if not seed_text.strip():
        st.warning("Please enter some text to get started.")
    else:
        with st.spinner("Thinking..."):
            predicted_words = generate_text(model, tokenizer, seed_text, max_len, n_words, index_to_word)
            
            if predicted_words:
                formatted_prediction = " ".join(predicted_words)
                
                # Display beautiful output
                st.markdown(f"""
                <div class="prediction-box">
                    {seed_text} <span class="highlight">{formatted_prediction}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("The model couldn't predict a confident next word for this sequence.")
