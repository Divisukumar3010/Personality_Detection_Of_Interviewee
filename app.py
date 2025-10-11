import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Page configuration
st.set_page_config(
    page_title="Personality Detection of Interviewee",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# st.markdown("<style>h2,h3,h4{color:#000!important}</style>", unsafe_allow_html=True)
# Custom CSS for styling
st.markdown("""
<style>
    .stApp { background-color: #F7F9FC; }
    .stAlert div[role="alert"] { color: #111827 !important; }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .question-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .trait-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        border-radius: 15px;
        margin-top: 3rem;
    }
    
    .progress-container {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stTextArea textarea{
        background-color:#F7F7F7 !important;
        color:#111827 !important;
        border:1px solid #E5E7EB !important;
        font-size:16px !important;
        line-height:1.6 !important;
    }
    .stTextArea textarea::placeholder{
        color:#9CA3AF !important;
    }
    .pill-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 0;
        padding: 0;
    }
    .pill {
        display: inline-block;
        padding: 10px 16px;
        border-radius: 9999px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 15px;
        line-height: 1;
        letter-spacing: 0.2px;
        box-shadow: 0 6px 16px rgba(106, 111, 245, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.25);
        white-space: nowrap;
    }
    .stTextArea textarea {
        color: #111 !important;
        caret-color: #111 !important;
    }
    .stTextArea textarea:focus {
        outline: 2px solid #4a90e2 !important;
        outline-offset: 1px;
    }
    .stTextArea, .stTextArea textarea {
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Personality types data
PERSONALITY_TYPES = {
    'INTJ': {
        'title': 'The Architect',
        'description': 'Imaginative and strategic thinkers, with a plan for everything. You approach life with deep thoughtfulness and imagination, always seeking to improve systems and ideas.',
        'traits': ['Strategic', 'Independent', 'Decisive', 'Hardworking', 'Determined'],
        'areas_to_watch': ['May seem aloof', 'Can be overly critical', 'Perfectionist tendencies'],
        'career_fits': ['Software Engineer', 'Research Scientist', 'Strategic Planner', 'System Analyst', 'Consultant'],
        'strengths': ['Long-term vision', 'Strategic thinking', 'Problem-solving', 'Independence', 'Efficiency'],
        'famous_people': ['Elon Musk', 'Isaac Newton', 'Nikola Tesla', 'Stephen Hawking']
    },
    'INTP': {
        'title': 'The Thinker',
        'description': 'Innovative inventors with an unquenchable thirst for knowledge. You love exploring theoretical possibilities and seek to understand how things work.',
        'traits': ['Logical', 'Creative', 'Objective', 'Curious', 'Independent'],
        'areas_to_watch': ['May procrastinate', 'Can be insensitive', 'Struggle with routine'],
        'career_fits': ['Research Scientist', 'Software Developer', 'Professor', 'Philosopher', 'Analyst'],
        'strengths': ['Analytical thinking', 'Creativity', 'Objectivity', 'Theoretical knowledge', 'Adaptability'],
        'famous_people': ['Albert Einstein', 'Bill Gates', 'Marie Curie', 'Charles Darwin']
    },
    'ENTJ': {
        'title': 'The Commander',
        'description': 'Bold, imaginative, and strong-willed leaders, always finding a way or making one. You are driven to organize and direct others toward common goals.',
        'traits': ['Leadership', 'Confident', 'Strategic', 'Efficient', 'Charismatic'],
        'areas_to_watch': ['Can be impatient', 'May seem aggressive', 'Struggle with emotions'],
        'career_fits': ['CEO', 'Executive', 'Entrepreneur', 'Manager', 'Consultant'],
        'strengths': ['Natural leadership', 'Strategic planning', 'Confidence', 'Efficiency', 'Goal-oriented'],
        'famous_people': ['Steve Jobs', 'Franklin D. Roosevelt', 'Margaret Thatcher', 'Gordon Ramsay']
    },
    'ENTP': {
        'title': 'The Debater',
        'description': 'Smart and curious thinkers who cannot resist an intellectual challenge. You are quick-witted and enjoy exploring new ideas and possibilities.',
        'traits': ['Innovative', 'Enthusiastic', 'Charismatic', 'Quick-witted', 'Original'],
        'areas_to_watch': ['May be disorganized', 'Can neglect routine', 'Difficulty with follow-through'],
        'career_fits': ['Entrepreneur', 'Marketing Manager', 'Journalist', 'Consultant', 'Inventor'],
        'strengths': ['Innovation', 'Enthusiasm', 'Charisma', 'Quick thinking', 'Adaptability'],
        'famous_people': ['Mark Twain', 'Walt Disney', 'Thomas Edison', 'Steve Wozniak']
    },
    'INFJ': {
        'title': 'The Advocate',
        'description': 'Creative and insightful, inspired and independent. You speak with warmth and conviction, especially about causes close to your heart.',
        'traits': ['Insightful', 'Creative', 'Inspiring', 'Decisive', 'Altruistic'],
        'areas_to_watch': ['Can be perfectionist', 'May burn out', 'Sensitive to criticism'],
        'career_fits': ['Counselor', 'Writer', 'Teacher', 'Social Worker', 'Psychologist'],
        'strengths': ['Insight', 'Creativity', 'Inspiration', 'Determination', 'Altruism'],
        'famous_people': ['Martin Luther King Jr.', 'Nelson Mandela', 'Mother Teresa', 'Oprah Winfrey']
    },
    'INFP': {
        'title': 'The Mediator',
        'description': 'Poetic, kind, and altruistic people, always eager to help a good cause. You are guided by your principles and seek harmony in everything you do.',
        'traits': ['Idealistic', 'Loyal', 'Adaptive', 'Curious', 'Creative'],
        'areas_to_watch': ['Can be overly idealistic', 'May take things personally', 'Difficulty with criticism'],
        'career_fits': ['Writer', 'Artist', 'Counselor', 'Social Worker', 'Designer'],
        'strengths': ['Idealism', 'Loyalty', 'Adaptability', 'Creativity', 'Open-mindedness'],
        'famous_people': ['J.R.R. Tolkien', 'Virginia Woolf', 'Kurt Cobain', 'Princess Diana']
    },
    'ENFJ': {
        'title': 'The Protagonist',
        'description': 'Charismatic and inspiring leaders, able to mesmerize their listeners. You are passionate about helping others reach their potential.',
        'traits': ['Charismatic', 'Altruistic', 'Natural Leader', 'Reliable', 'Tolerant'],
        'areas_to_watch': ['Can be overly idealistic', 'May neglect own needs', 'Sensitive to criticism'],
        'career_fits': ['Teacher', 'Coach', 'Counselor', 'HR Manager', 'Politician'],
        'strengths': ['Leadership', 'Charisma', 'Altruism', 'Reliability', 'Communication'],
        'famous_people': ['Barack Obama', 'Maya Angelou', 'Jennifer Lawrence', 'Ben Affleck']
    },
    'ENFP': {
        'title': 'The Campaigner',
        'description': 'Enthusiastic, creative, and sociable free spirits, who can always find a reason to smile. You see life as full of possibilities.',
        'traits': ['Enthusiastic', 'Creative', 'Sociable', 'Energetic', 'Independent'],
        'areas_to_watch': ['Can be disorganized', 'May struggle with routine', 'Overthinking'],
        'career_fits': ['Marketing', 'Journalism', 'Psychology', 'Teaching', 'Consulting'],
        'strengths': ['Enthusiasm', 'Creativity', 'People skills', 'Energy', 'Flexibility'],
        'famous_people': ['Robin Williams', 'Ellen DeGeneres', 'Will Smith', 'Robert Downey Jr.']
    },
    'ISTJ': {
        'title': 'The Logistician',
        'description': 'Practical and fact-minded, reliable and responsible. You believe in hard work and take pride in completing tasks thoroughly.',
        'traits': ['Responsible', 'Realistic', 'Practical', 'Reliable', 'Orderly'],
        'areas_to_watch': ['Can be inflexible', 'May resist change', 'Overly critical'],
        'career_fits': ['Accountant', 'Manager', 'Administrator', 'Engineer', 'Doctor'],
        'strengths': ['Reliability', 'Practicality', 'Organization', 'Dedication', 'Honesty'],
        'famous_people': ['George Washington', 'Warren Buffett', 'Angela Merkel', 'Natalie Portman']
    },
    'ISFJ': {
        'title': 'The Protector',
        'description': 'Warm-hearted and dedicated, always ready to protect loved ones. You combine kindness with reliability and attention to detail.',
        'traits': ['Supportive', 'Reliable', 'Patient', 'Imaginative', 'Observant'],
        'areas_to_watch': ['May neglect own needs', 'Can be overly humble', 'Reluctant to change'],
        'career_fits': ['Nurse', 'Teacher', 'Social Worker', 'Administrator', 'Counselor'],
        'strengths': ['Supportiveness', 'Reliability', 'Patience', 'Practical skills', 'Loyalty'],
        'famous_people': ['Mother Teresa', 'Kate Middleton', 'Captain America', 'Mr. Rogers']
    },
    'ESTJ': {
        'title': 'The Executive',
        'description': 'Excellent administrators, unsurpassed at managing things or people. You bring order and structure to any situation.',
        'traits': ['Organized', 'Practical', 'Realistic', 'Logical', 'Decisive'],
        'areas_to_watch': ['Can be inflexible', 'May be impatient', 'Difficulty with emotions'],
        'career_fits': ['Manager', 'Administrator', 'Executive', 'Judge', 'Military Officer'],
        'strengths': ['Organization', 'Leadership', 'Practicality', 'Decisiveness', 'Dedication'],
        'famous_people': ['Vince Lombardi', 'Frank Sinatra', 'John D. Rockefeller', 'Hillary Clinton']
    },
    'ESFJ': {
        'title': 'The Consul',
        'description': 'Extraordinarily caring, social, and popular people, always eager to help. You bring out the best in others and create harmony.',
        'traits': ['Caring', 'Social', 'Loyal', 'Organized', 'Dutiful'],
        'areas_to_watch': ['Can be people-pleasing', 'May neglect own needs', 'Sensitive to criticism'],
        'career_fits': ['Teacher', 'Nurse', 'Social Worker', 'Event Coordinator', 'HR Specialist'],
        'strengths': ['People skills', 'Organization', 'Loyalty', 'Practical support', 'Harmony'],
        'famous_people': ['Taylor Swift', 'Danny Glover', 'Mary Tyler Moore', 'Sally Field']
    },
    'ISTP': {
        'title': 'The Virtuoso',
        'description': 'Bold and practical experimenters, masters of all kinds of tools. You love understanding how things work and solving problems hands-on.',
        'traits': ['Practical', 'Adaptable', 'Reserved', 'Logical', 'Spontaneous'],
        'areas_to_watch': ['Can be unpredictable', 'May seem insensitive', 'Difficulty with emotions'],
        'career_fits': ['Engineer', 'Mechanic', 'Pilot', 'Firefighter', 'Detective'],
        'strengths': ['Problem-solving', 'Adaptability', 'Practical skills', 'Crisis management', 'Efficiency'],
        'famous_people': ['Clint Eastwood', 'Tom Cruise', 'Daniel Craig', 'Megan Thee Stallion']
    },
    'ISFP': {
        'title': 'The Adventurer',
        'description': 'Flexible and charming artists, always ready to explore new possibilities. You live in the moment and enjoy helping others.',
        'traits': ['Flexible', 'Charming', 'Artistic', 'Curious', 'Passionate'],
        'areas_to_watch': ['Can be unpredictable', 'May be overly competitive', 'Stress-prone'],
        'career_fits': ['Artist', 'Designer', 'Musician', 'Counselor', 'Veterinarian'],
        'strengths': ['Artistic ability', 'Flexibility', 'Passion', 'Curiosity', 'Loyalty'],
        'famous_people': ['Michael Jackson', 'Rihanna', 'Bob Dylan', 'Frida Kahlo']
    },
    'ESTP': {
        'title': 'The Entrepreneur',
        'description': 'Smart, energetic, and perceptive people, who truly enjoy living on the edge. You are spontaneous and love being the center of attention.',
        'traits': ['Energetic', 'Spontaneous', 'Realistic', 'Pragmatic', 'Bold'],
        'areas_to_watch': ['May be impulsive', 'Can be insensitive', 'Difficulty with long-term planning'],
        'career_fits': ['Sales', 'Marketing', 'Entertainment', 'Sports', 'Emergency Services'],
        'strengths': ['Energy', 'Spontaneity', 'Practicality', 'People skills', 'Adaptability'],
        'famous_people': ['Donald Trump', 'Madonna', 'Eddie Murphy', 'Bruce Willis']
    },
    'ESFP': {
        'title': 'The Entertainer',
        'description': 'Spontaneous, energetic, and enthusiastic people – life is never boring around you. You love inspiring others and bringing joy to every situation.',
        'traits': ['Enthusiastic', 'Friendly', 'Spontaneous', 'Flexible', 'People-oriented'],
        'areas_to_watch': ['Can be disorganized', 'May avoid conflict', 'Difficulty with criticism'],
        'career_fits': ['Performer', 'Teacher', 'Social Worker', 'Counselor', 'Sales'],
        'strengths': ['Enthusiasm', 'People skills', 'Flexibility', 'Optimism', 'Practical support'],
        'famous_people': ['Elvis Presley', 'Marilyn Monroe', 'Will Smith', 'Jamie Foxx']
    }
}

# Interview questions
QUESTIONS = [
    "Tell me about yourself in a few sentences.",
    "How do you usually prepare for an important task or project?",
    "Do you prefer working alone or in a team? Why?",
    "How do you handle stress or pressure in the workplace?",
    "Describe a situation where you solved a difficult problem.",
    "How do you usually make decisions – logically or based on intuition?",
    "What motivates you to perform well in your work or studies?",
    "How do you handle conflicts with teammates or colleagues?",
    "Are you more comfortable following structured rules or being flexible?",
    "How do you adapt when faced with unexpected challenges?",
    "Do you prefer detailed planning or going with the flow?",
    "Describe how you manage your time when handling multiple tasks.",
    "Do you focus more on facts/data or possibilities/ideas?",
    "How do you usually contribute during group discussions?",
    "What is your preferred way of learning something new?",
    "Do you find it easy to express your feelings to others?",
    "How do you usually react to feedback or criticism?",
    "Do you prefer long-term goals or short-term achievements?",
    "What type of role do you usually take in a group (leader, supporter, strategist, etc.)?",
    "If given full freedom, how would you ideally like to work?"
]

class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # ------------------------------------Convert to lowercase------------------------------------ 
        text = text.lower()
        
        #----------------------------- Remove special characters and digits---------------------------
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # ------------------------------------------ Tokenization ------------------------------------
        tokens = word_tokenize(text)
        
        # ----------------------------------- Remove stopwords and lemmatize---------------------------
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)

class PersonalityModel:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False
    
    def create_synthetic_data(self):
        """Create synthetic training data for demonstration"""
        np.random.seed(42)
        
        # Keywords associated with each personality dimension
        extraversion_words = ['team', 'people', 'social', 'group', 'collaborate', 'talk', 'discuss', 'share']
        introversion_words = ['alone', 'individual', 'quiet', 'focus', 'concentrate', 'independent']
        
        sensing_words = ['practical', 'concrete', 'detail', 'fact', 'data', 'step', 'procedure']
        intuition_words = ['possibility', 'idea', 'concept', 'theory', 'innovation', 'creative', 'vision']
        
        thinking_words = ['logic', 'analyze', 'objective', 'rational', 'efficient', 'system', 'solve']
        feeling_words = ['feel', 'emotion', 'value', 'harmony', 'care', 'help', 'relationship']
        
        judging_words = ['plan', 'organize', 'schedule', 'structure', 'deadline', 'goal', 'decide']
        perceiving_words = ['flexible', 'adapt', 'spontaneous', 'open', 'explore', 'flow', 'change']
        
        synthetic_data = []
        labels = []
        
        for personality_type in PERSONALITY_TYPES.keys():
            for _ in range(50):  # 50 samples per type
                text_parts = []
                
                # Generate text based on personality type
                if 'E' in personality_type:
                    text_parts.extend(np.random.choice(extraversion_words, 3))
                else:
                    text_parts.extend(np.random.choice(introversion_words, 3))
                
                if 'S' in personality_type:
                    text_parts.extend(np.random.choice(sensing_words, 3))
                else:
                    text_parts.extend(np.random.choice(intuition_words, 3))
                
                if 'T' in personality_type:
                    text_parts.extend(np.random.choice(thinking_words, 3))
                else:
                    text_parts.extend(np.random.choice(feeling_words, 3))
                
                if 'J' in personality_type:
                    text_parts.extend(np.random.choice(judging_words, 3))
                else:
                    text_parts.extend(np.random.choice(perceiving_words, 3))
                
                # Add some random words for variety
                random_words = ['work', 'project', 'time', 'important', 'situation', 'approach', 'method']
                text_parts.extend(np.random.choice(random_words, 5))
                
                synthetic_text = ' '.join(text_parts)
                synthetic_data.append(synthetic_text)
                labels.append(personality_type)
        
        return synthetic_data, labels
    
    # ------------------------------------ Train the personality prediction model ------------------------------------

    def train_model(self):
        """Train the personality prediction model"""
        # ------------------------------------ Create synthetic training data----------------------
        texts, labels = self.create_synthetic_data()
        
        # ------------------------------------ Preprocess texts ------------------------------------
        processed_texts = [self.nlp_processor.preprocess_text(text) for text in texts]
        
        # ------------------------------------ Vectorize------------------------------------
        X = self.vectorizer.fit_transform(processed_texts)
        
        # ------------------------------------ Train model ------------------------------------
        self.model.fit(X, labels)
        self.is_trained = True
        
        return self.model
    
    def predict_personality(self, answers):
        """Predict personality type from answers"""
        if not self.is_trained:
            self.train_model()
        
        # Combine all answers into one text
        combined_text = ' '.join(answers)
        
        # Preprocess
        processed_text = self.nlp_processor.preprocess_text(combined_text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities) * 100
        
        # Get top matches
        classes = self.model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_matches = [(classes[i], probabilities[i] * 100) for i in top_indices]
        
        return prediction, confidence, top_matches

def _level_from_score(pct: float) -> str:
    if pct >= 70:
        return "High"
    if pct >= 40:
        return "Medium"
    return "Low"

def create_circular_progress(percentage, title, *,
                            ring_color="#6D5EF3",
                            bg_ring="#E9ECF3",
                            center_text="#1F2937"):
    """Donut KPI with auto subtitle based on score."""
    pct = max(0, min(100, float(percentage)))
    subtitle_text = _level_from_score(pct)

    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct],
        hole=0.7,
        sort=False,
        direction="clockwise",
        marker=dict(
            colors=[ring_color, bg_ring],
            line=dict(color=bg_ring, width=2)
        ),
        textinfo="none",
        hoverinfo="skip",
        showlegend=False
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5, xanchor="center",
            y=0.96, yanchor="top",
            font=dict(size=16, color=center_text)
        ),
        height=360,
        width=360,
        margin=dict(t=60, b=40, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    # Center value
    fig.add_annotation(
        x=0.5, y=0.56,
        text=f"<b>{pct:.0f}%</b>",
        showarrow=False,
        font=dict(size=34, color=center_text),
        xref="paper", yref="paper"
    )
    # Auto subtitle from score
    fig.add_annotation(
        x=0.5, y=0.40,
        text=f"<span style='color:#6B7280;font-size:13px'>{subtitle_text}</span>",
        showarrow=False,
        xref="paper", yref="paper"
    )

    return fig

def show_home_page():
    """Display the home page"""
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🧠 Personality Detection of Interviewee</h1>
        <h3 style="font-weight: 300; margin-bottom: 2rem;">Answer 20 interview-style questions and discover your personality</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <h3 style='color:#000'>🎯 Professional Assessment</h3>
            <p style='color:#000'>Get insights into your personality type through AI-powered analysis of your interview responses.</p>
            <br>
            <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
                <div>
                    <h4 style='color:#000'>📊 Accurate Results</h4>
                    <p style='color:#000'>Advanced NLP analysis</p>
                </div>
                <div>
                    <h4 style='color:#000'>⚡ Instant Analysis</h4>
                    <p style='color:#000'>Get results immediately</p>
                </div>
                <div>
                    <h4 style='color:#000'>🎨 Beautiful Reports</h4>
                    <p style='color:#000'>Professional visualizations</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Personality Test", key="start_test", use_container_width=True):
            st.session_state.page = 'questionnaire'
            st.session_state.current_question = 0
            st.session_state.answers = [''] * 20
            st.rerun()

def show_questionnaire_page():
    """Display the questionnaire page"""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = [''] * 20
    
    current_q = st.session_state.current_question
    progress = (current_q + 1) / len(QUESTIONS) * 100
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h2>📝 Personality Assessment</h2>
        <p>Question {current_q + 1} of {len(QUESTIONS)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    st.markdown(f"""
    <div class="progress-container">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 10px; border-radius: 5px; width: {progress}%;"></div>
        <p style="text-align: center; margin-top: 0.5rem; color: #666;">{progress:.0f}% Complete</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question card
    st.markdown(f"""
    <div class="question-card">
        <h3 style='color:#000'>Question {current_q + 1}</h3>
        <p style="font-size: 1.1rem; color: #333; margin-bottom: 1rem;">{QUESTIONS[current_q]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer input
    # answer = st.text_area(
    #     <p>"Your Answer:"</p>,
    #     value=st.session_state.answers[current_q],
    #     height=150,
    #     placeholder="Type your answer here... Be as detailed as you'd like.",
    #     key=f"answer_{current_q}"
    # )
    # --------------------------
    st.markdown("<span style='color:#000;font-weight:600'>Your Answer:</span>", unsafe_allow_html=True)
    answer = st.text_area(
    "",
        value=st.session_state.answers[current_q],
        height=150,
        placeholder="Type your answer here... Be as detailed as you'd like.",
        key=f"answer_{current_q}",
        label_visibility="collapsed"
    )
    
    # Update answer in session state
    st.session_state.answers[current_q] = answer
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_q > 0:
            if st.button("⬅️ Previous", key="prev_btn"):
                st.session_state.current_question -= 1
                st.rerun()
        else:
            if st.button("🏠 Back to Home", key="home_btn"):
                st.session_state.page = 'home'
                st.rerun()
    
    with col3:
        if len(answer.strip()) >= 10:  # Minimum answer length
            if current_q < len(QUESTIONS) - 1:
                if st.button("Next ➡️", key="next_btn"):
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                if st.button("🎯 Get My Results", key="results_btn"):
                    st.session_state.page = 'results'
                    st.rerun()
        else:
            st.warning("Please provide at least 10 characters in your answer.")

def show_results_page():
    """Display the results page"""
    if 'personality_result' not in st.session_state:
        # Initialize model and make prediction
        model = PersonalityModel()
        
        with st.spinner("<span style='color:#000;'>🔮 Analyzing your personality... This may take a moment.</span>"):
            # Filter out empty answers
            valid_answers = [answer for answer in st.session_state.answers if answer.strip()]
            
            if len(valid_answers) < 10:
                st.error("Not enough answers provided. Please complete more questions.")
                return
            
            personality_type, confidence, top_matches = model.predict_personality(valid_answers)
            
            st.session_state.personality_result = {
                'type': personality_type,
                'confidence': confidence,
                'top_matches': top_matches
            }
    
    result = st.session_state.personality_result
    personality_info = PERSONALITY_TYPES[result['type']]
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎉 Your Personality Results</h1>
        <p>Based on your interview responses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main result card
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h2 style="color: #667eea; margin-bottom: 1rem;">
                {result['type']} - {personality_info['title']}
            </h2>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #555; margin-bottom: 1.5rem;">
                {personality_info['description']}
            </p>
            <div>
                <h4 style="color: #667eea; margin-bottom: 1rem;">Key Traits:</h4>
                <div style="margin-top: 1rem;">
                    {''.join([f'<span class="trait-pill">{trait}</span>' for trait in personality_info['traits']])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence score chart
        confidence_fig = create_circular_progress(result['confidence'], 'Confidence Score')
        st.plotly_chart(confidence_fig, use_container_width=True)
    
    # Detailed analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
        <h4 style="color:#667eea; margin-bottom:0.75rem;">💪 Strengths</h4>
        <div class="pill-wrap">
            {''.join([f'<span class="pill">{strength}</span>'
                    for strength in personality_info['strengths']])}
        </div>
        </div>
        """,
            unsafe_allow_html=True
        )

    
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color: #667eea; ">💼 Career Fits</h4>
            <div>
                {''.join([f'<span class="trait-pill">{career}</span>' for career in personality_info['career_fits']])}
            </div>
            
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color: #667eea;">⚠️ Areas to Watch</h4>
            <div>
                {''.join([f'<span class="trait-pill">{area}</span>' for area in personality_info['areas_to_watch']])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Famous people and top matches
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🌟 Famous People</h4>
            <div style="margin-top: 1rem;">
                {''.join([f'<span class="trait-pill">{person}</span>' for person in personality_info['famous_people']])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        results_html = f"""
        <div style="
            background: white;
            border-radius: 18px;
            padding: 2px 30px;
            max-width: 100%;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-top:15px
        ">
        <div style="display: flex; align-items: center;margin-top:15.5px;">
            <h4 style="color: #667eea;">🏆 Top Personality Matches</h4>
        </div>
        """

        for match_type, percentage in result['top_matches']:
            results_html += f"""
        <div style="display: flex; justify-content: space-between; margin: 8px 0;">
            <span style="color: #9CA3AF; font-weight: 500; font-size: 1rem;">{match_type}</span>
            <span style="color: #667eea; font-weight: 600; font-size: 1rem;">{percentage:.1f}%</span>
        </div>
        """

        results_html += "</div>"

        st.markdown(results_html, unsafe_allow_html=True)


    
    # Personality dimensions
    card_html = """
    <div style="
    background: #fff;
    border-radius: 18px;
    padding: 32px 36px 24px 36px;
    box-shadow: 0 6px 24px rgba(76,106,166,0.07);
    max-width: 100%;
    margin-top:15px;
    font-family: Arial, sans-serif;
    ">
    <h4 style="color: #667eea; margin-bottom: 1.5rem;">📊 Your Personality Dimensions</h4>
    """
    
    dimensions = {
        'Extrovert vs Introvert': 'E' in result['type'],
        'Sensing vs Intuition': 'S' in result['type'],
        'Thinking vs Feeling': 'T' in result['type'],
        'Judging vs Perceiving': 'J' in result['type']
    }
    
    for dimension, preference in dimensions.items():
        score = 70 if preference else 30
        preference_text = dimension.split(' vs ')[0 if preference else 1]
    
        card_html += f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
        <span style="font-weight: 500; color: #1f2937;">{dimension}</span>
        <span style="color: #667eea; font-weight: 600;">{score}%</span>
        </div>
        <div style="background: #e9ecef; border-radius: 10px; height: 8px;">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 8px; border-radius: 10px; width: {score}%;"></div>
        </div>
        <p style="margin-top: 0.5rem; color: #6b7280; font-size: 0.9rem;">{preference_text}</p>
    </div>
        """
    
    card_html += "</div>"
    
    st.markdown(card_html, unsafe_allow_html=True)
    

    
    # Restart button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Take Test Again", key="restart_btn", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.page = 'home'
            st.rerun()

def show_footer():
    """Display the footer"""
    st.markdown("""
    <div class="footer">
        <h4>🌟 Experience the Future of Personality Analysis</h4>
        <p>• Privacy-Focused • Lightning-fast results</p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">© 2025 Sukumar Divi. All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Navigation logic
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'questionnaire':
        show_questionnaire_page()
    elif st.session_state.page == 'results':
        show_results_page()
    
    # Always show footer
    show_footer()

if __name__ == "__main__":
    main()