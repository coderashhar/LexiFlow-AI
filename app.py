import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure the Streamlit page
st.set_page_config(
    page_title="TextFlow AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern, premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }

    /* Background and theme */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #2e1065, #0f172a 60%);
        color: #f8fafc;
    }
    
    /* Clean UI (Hide header/footer) */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Headers */
    .main-header {
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(to right, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        padding-top: 1rem;
        letter-spacing: -1.5px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 300;
        margin-bottom: 3.5rem;
        letter-spacing: 0.5px;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 16px 24px !important;
        font-size: 1.15rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2) !important;
        background: rgba(30, 41, 59, 0.6) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 12px 24px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2) !important;
        height: 100% !important;
        min-height: 52px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.3) !important;
        background: linear-gradient(135deg, #4f46e5, #9333ea) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #a78bfa !important;
    }

    /* Prediction Result Box */
    .prediction-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 20px;
        padding: 35px 40px;
        margin-top: 40px;
        font-size: 1.5rem;
        color: #f1f5f9;
        line-height: 1.8;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(16px);
        text-align: center;
        animation: scaleIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .highlight {
        background: linear-gradient(120deg, rgba(167, 139, 250, 0.15) 0%, rgba(56, 189, 248, 0.15) 100%);
        color: #a78bfa;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 10px;
        border: 1px solid rgba(167, 139, 250, 0.3);
        display: inline-block;
        margin-left: 5px;
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95) translateY(10px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    /* Labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
        padding-bottom: 8px !important;
    }
    
    /* Help Tooltip Icon */
    .stTooltipIcon {
        color: #94a3b8 !important;
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
st.markdown('<div class="main-header">TextFlow AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Next Word Prediction Engine ✨</div>', unsafe_allow_html=True)

# Load model and assets
try:
    with st.spinner('Waking up the neural network...'):
        model, tokenizer, max_len = load_assets()
        # Create the reverse dictionary for predictions
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}
except Exception as e:
    st.error(f"Error loading the model files. Make sure `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` are present. ({e})")
    st.stop()

# Input area
seed_text = st.text_input("Enter your starting text:", placeholder="The future of artificial intelligence is...", key="user_input")

# Generation controls
col1, col2 = st.columns([3, 1])
with col1:
    n_words = st.slider("Words to predict:", min_value=1, max_value=20, value=5, help="How many words should the AI generate after your prompt?")
with col2:
    st.markdown("<div style='height: 31px;'></div>", unsafe_allow_html=True) # Perfect alignment spacer
    predict_btn = st.button("Generate ✨", use_container_width=True)

# Action
if predict_btn or (seed_text and n_words):
    if not seed_text.strip():
        st.warning("Please enter some text to get started.")
    else:
        with st.spinner("Generating..."):
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
