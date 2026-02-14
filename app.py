import streamlit as st
import numpy as np
import re
import plotly.graph_objects as go
import json

# Page configuration
st.set_page_config(
    page_title="Personality Detection of Interviewee",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Modern CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root variables ──────────────────────────────────────── */
:root {
    --primary: #6C63FF;
    --primary-dark: #5A52D5;
    --secondary: #FF6584;
    --accent: #43E97B;
    --bg: #0F1117;
    --bg-card: #181B2A;
    --bg-card-hover: #1E2235;
    --bg-glass: rgba(24, 27, 42, 0.7);
    --text: #E8E8ED;
    --text-muted: #9BA1B7;
    --text-dim: #636A82;
    --border: rgba(108, 99, 255, 0.15);
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    --shadow-glow: 0 0 40px rgba(108, 99, 255, 0.15);
    --radius: 16px;
    --radius-sm: 10px;
    --radius-pill: 50px;
}

/* ── Global ──────────────────────────────────────────────── */
.stApp {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text) !important;
}

/* Hide Streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 1200px !important; }

/* ── Alerts ──────────────────────────────────────────────── */
.stAlert div[role="alert"] { color: #fbbf24 !important; background: rgba(251,191,36,0.08) !important;
    border: 1px solid rgba(251,191,36,0.2) !important; border-radius: var(--radius-sm) !important;}

/* ── Hero Section ────────────────────────────────────────── */
.hero {
    position: relative;
    text-align: center;
    padding: 4rem 2rem 3.5rem;
    border-radius: 24px;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    overflow: hidden;
    border: 1px solid rgba(108, 99, 255, 0.2);
    box-shadow: var(--shadow-glow);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(108,99,255,0.15) 0%, transparent 60%),
                radial-gradient(circle at 70% 80%, rgba(255,101,132,0.1) 0%, transparent 50%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 3rem; font-weight: 800; margin: 0 0 0.6rem;
    background: linear-gradient(135deg, #fff 0%, #c4b5fd 50%, #6C63FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; position: relative;
}
.hero .subtitle {
    font-size: 1.15rem; color: var(--text-muted); max-width: 560px;
    margin: 0 auto; line-height: 1.7; position: relative; font-weight: 400;
}
.hero .badge {
    display: inline-block; padding: 6px 16px; border-radius: var(--radius-pill);
    background: rgba(108,99,255,0.15); color: var(--primary); font-size: 0.8rem;
    font-weight: 600; margin-bottom: 1.2rem; border: 1px solid rgba(108,99,255,0.25);
    letter-spacing: 0.08em; text-transform: uppercase; position: relative;
}

/* ── Glass Card ──────────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 1.8rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow), var(--shadow-glow);
}
.glass-card h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--primary) !important; font-size: 1.1rem; font-weight: 600;
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px;
}
.glass-card p { color: var(--text-muted); line-height: 1.65; font-size: 0.95rem; }

/* ── Feature Row ─────────────────────────────────────────── */
.feature-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem;
    margin: 2rem 0;
}
.feature-item {
    text-align: center; padding: 2rem 1.4rem; border-radius: var(--radius);
    background: var(--bg-card); border: 1px solid var(--border);
    transition: all 0.3s ease;
}
.feature-item:hover { border-color: var(--primary); transform: translateY(-2px); }
.feature-icon {
    width: 56px; height: 56px; border-radius: 14px; display: inline-flex;
    align-items: center; justify-content: center; font-size: 1.6rem; margin-bottom: 1rem;
}
.feature-icon.purple { background: rgba(108,99,255,0.12); }
.feature-icon.pink   { background: rgba(255,101,132,0.12); }
.feature-icon.green  { background: rgba(67,233,123,0.12); }
.feature-item h5 {
    color: var(--text) !important; font-weight: 600; margin: 0 0 0.4rem; font-size: 1rem;
}
.feature-item p { color: var(--text-muted); font-size: 0.88rem; margin: 0; line-height: 1.55; }

/* ── Question Card ───────────────────────────────────────── */
.q-card {
    background: var(--bg-card); border-radius: var(--radius);
    padding: 2rem 2.2rem; border: 1px solid var(--border);
    border-left: 4px solid var(--primary); box-shadow: var(--shadow);
    margin: 1.5rem 0;
}
.q-number {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 10px;
    background: rgba(108,99,255,0.12); color: var(--primary);
    font-weight: 700; font-size: 0.95rem; margin-bottom: 0.8rem;
}
.q-text {
    font-size: 1.15rem; color: var(--text); line-height: 1.65; font-weight: 500;
}

/* ── Progress Bar ────────────────────────────────────────── */
.progress-track {
    background: rgba(108,99,255,0.08); border-radius: 12px; height: 8px;
    margin: 1rem 0 0.4rem; overflow: hidden;
}
.progress-fill {
    height: 100%; border-radius: 12px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transition: width 0.5s ease;
}
.progress-label {
    text-align: center; color: var(--text-dim); font-size: 0.82rem; font-weight: 500;
}
/* Step dots */
.step-dots {
    display: flex; justify-content: center; gap: 6px; margin: 0.8rem 0;
}
.step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(108,99,255,0.15); transition: all 0.3s ease;
}
.step-dot.done { background: var(--primary); }
.step-dot.active { background: var(--primary); width: 24px; border-radius: 4px; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important; border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.8rem 2.2rem !important; font-weight: 600 !important;
    font-size: 0.95rem !important; letter-spacing: 0.02em;
    transition: all 0.3s ease !important; box-shadow: 0 4px 16px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(108,99,255,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Text Area ───────────────────────────────────────────── */
.stTextArea textarea {
    background: var(--bg-card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
    font-size: 1rem !important; line-height: 1.7 !important;
    caret-color: var(--primary) !important; padding: 1rem !important;
    transition: border-color 0.3s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea::placeholder { color: var(--text-dim) !important; }
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.1) !important;
    outline: none !important;
}
.stTextArea, .stTextArea textarea { opacity: 1 !important; }
.stTextArea label { color: var(--text) !important; font-weight: 500 !important; }

/* ── Result Type Badge ───────────────────────────────────── */
.type-badge {
    display: inline-block; padding: 10px 22px; border-radius: var(--radius-pill);
    background: linear-gradient(135deg, var(--primary), #8B5CF6);
    color: #fff; font-weight: 700; font-size: 1.4rem;
    letter-spacing: 0.1em; box-shadow: 0 6px 20px rgba(108,99,255,0.35);
    margin-bottom: 0.8rem;
}
.type-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.6rem; font-weight: 700; color: var(--text) !important;
    margin: 0.4rem 0 0.8rem;
}
.type-desc { color: var(--text-muted); font-size: 1rem; line-height: 1.7; }

/* ── Trait / Pill Tags ───────────────────────────────────── */
.pill-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.6rem; }
.pill {
    display: inline-block; padding: 7px 16px; border-radius: var(--radius-pill);
    font-weight: 600; font-size: 0.85rem; line-height: 1;
    letter-spacing: 0.02em; white-space: nowrap; transition: transform 0.2s ease;
}
.pill:hover { transform: scale(1.04); }
.pill.purple {
    background: rgba(108,99,255,0.12); color: #A78BFA;
    border: 1px solid rgba(108,99,255,0.2);
}
.pill.pink {
    background: rgba(255,101,132,0.1); color: #FF6584;
    border: 1px solid rgba(255,101,132,0.18);
}
.pill.green {
    background: rgba(67,233,123,0.1); color: #43E97B;
    border: 1px solid rgba(67,233,123,0.18);
}
.pill.amber {
    background: rgba(251,191,36,0.1); color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.18);
}
.pill.blue {
    background: rgba(59,130,246,0.1); color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.18);
}

/* ── Dimension Bar ───────────────────────────────────────── */
.dim-row { margin-bottom: 1.6rem; }
.dim-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.dim-label { font-weight: 500; color: var(--text); font-size: 0.95rem; }
.dim-value { font-weight: 700; color: var(--primary); font-size: 0.95rem; }
.dim-track { background: rgba(108,99,255,0.08); border-radius: 8px; height: 8px; }
.dim-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, var(--primary), #8B5CF6);
    transition: width 0.6s ease;
}
.dim-pref { color: var(--text-dim); font-size: 0.82rem; margin-top: 4px; }

/* ── Match Row ───────────────────────────────────────────── */
.match-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid var(--border);
}
.match-row:last-child { border-bottom: none; }
.match-type { color: var(--text-muted); font-weight: 500; font-size: 0.95rem; }
.match-pct { color: var(--primary); font-weight: 700; font-size: 0.95rem; }

/* ── Footer ──────────────────────────────────────────────── */
.app-footer {
    text-align: center; padding: 2rem 1.5rem; margin-top: 3.5rem;
    border-top: 1px solid var(--border);
}
.app-footer p { color: var(--text-dim); font-size: 0.82rem; margin: 0.3rem 0; }
.app-footer .heart { color: var(--secondary); }

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero h1 { font-size: 2rem; }
    .feature-grid { grid-template-columns: 1fr; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}

/* ── Misc ────────────────────────────────────────────────── */
.section-label {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important; font-size: 1.25rem; font-weight: 700;
    margin: 2rem 0 1rem; display: flex; align-items: center; gap: 8px;
}
hr { border-color: var(--border) !important; }
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



# ─── MBTI Dimension Scoring Engine ─────────────────────────────────────────
# Each dimension is scored independently using extensive keyword/phrase matching.
# This is far more accurate than the previous synthetic-data ML approach.

# Which MBTI dimension each question is most relevant to (0-indexed)
# E/I = 0, S/N = 1, T/F = 2, J/P = 3
QUESTION_DIMENSION_MAP = {
    0: [0],       # Tell me about yourself
    1: [3, 1],    # Prepare for task → J/P, S/N
    2: [0],       # Alone vs team → E/I
    3: [2, 0],    # Handle stress → T/F, E/I
    4: [1, 2],    # Solve a problem → S/N, T/F
    5: [2, 1],    # Logic vs intuition → T/F, S/N
    6: [2, 0],    # Motivation → T/F, E/I
    7: [2, 0],    # Handle conflicts → T/F, E/I
    8: [3],       # Structured vs flexible → J/P
    9: [3, 1],    # Unexpected challenges → J/P, S/N
    10: [3],      # Planning vs flow → J/P
    11: [3, 1],   # Manage time → J/P, S/N
    12: [1],      # Facts vs possibilities → S/N
    13: [0, 2],   # Group discussions → E/I, T/F
    14: [1, 0],   # Learning style → S/N, E/I
    15: [2, 0],   # Express feelings → T/F, E/I
    16: [2, 3],   # React to feedback → T/F, J/P
    17: [3, 1],   # Long vs short-term → J/P, S/N
    18: [0, 2],   # Role in group → E/I, T/F
    19: [3, 0],   # Ideal work → J/P, E/I
}

# Keyword dictionaries: (keyword/phrase, weight)
# Higher weight = stronger indicator
DIMENSION_KEYWORDS = {
    # ── Extroversion vs Introversion ──────────────────────────
    'E': [
        # Social orientation
        ('team', 2), ('teams', 2), ('teamwork', 2), ('collaborate', 3),
        ('collaboration', 3), ('people', 1.5), ('social', 2), ('socialize', 2),
        ('group', 1.5), ('groups', 1.5), ('together', 1.5), ('outgoing', 3),
        ('extrovert', 3), ('talkative', 2), ('energetic', 1.5),
        # Communication
        ('discuss', 1.5), ('discussion', 1.5), ('share', 1), ('sharing', 1),
        ('talk', 1.5), ('talking', 1.5), ('communicate', 2), ('communication', 2),
        ('brainstorm', 2), ('brainstorming', 2), ('debate', 2),
        # Activity
        ('lead', 1.5), ('leader', 2), ('leadership', 2), ('engage', 1.5),
        ('interact', 2), ('interaction', 2), ('network', 2), ('networking', 2),
        ('meeting', 1), ('meetings', 1), ('present', 1.5), ('presentation', 1.5),
        # Phrases
        ('work with others', 3), ('enjoy working with', 3), ('prefer team', 3),
        ('like being around', 3), ('with colleagues', 1.5), ('with people', 2),
        ('bounce ideas', 2), ('think out loud', 2), ('open environment', 2),
        ('thrive in groups', 3), ('feed off energy', 3), ('love meeting', 2),
    ],
    'I': [
        # Solitude orientation
        ('alone', 2.5), ('independent', 2), ('independently', 2), ('solo', 2.5),
        ('solitude', 3), ('introvert', 3), ('privacy', 2), ('private', 2),
        ('quiet', 2), ('reserved', 2), ('self', 1), ('myself', 1.5),
        # Work style
        ('focus', 1.5), ('focused', 1.5), ('concentrate', 2), ('concentration', 2),
        ('deep work', 3), ('deep thinking', 3), ('reflect', 2), ('reflection', 2),
        ('think before', 2), ('think things through', 3), ('internal', 1.5),
        ('individual', 2), ('individually', 2), ('personal space', 2.5),
        # Phrases
        ('work alone', 3), ('prefer working alone', 4), ('by myself', 3),
        ('on my own', 2.5), ('need space', 2.5), ('recharge alone', 3),
        ('small group', 1.5), ('one on one', 2), ('written communication', 2),
        ('rather listen', 2), ('observe first', 2), ('take time to', 1.5),
    ],

    # ── Sensing vs Intuition ──────────────────────────────────
    'S': [
        # Practical focus
        ('practical', 2.5), ('realistic', 2), ('concrete', 2.5), ('specific', 1.5),
        ('detail', 2), ('details', 2), ('detailed', 2), ('fact', 2), ('facts', 2),
        ('data', 2), ('evidence', 2), ('proven', 2), ('experience', 1.5),
        ('experienced', 1.5), ('hands-on', 3), ('hands on', 3), ('tangible', 2.5),
        # Process
        ('step by step', 3), ('step-by-step',
                              3), ('procedure', 2), ('process', 1.5),
        ('routine', 2), ('systematic', 2), ('methodical', 2), ('thorough', 1.5),
        ('careful', 1.5), ('accurate', 2), ('accuracy', 2), ('precise', 2),
        # Focus
        ('present', 1), ('current', 1), ('actual', 1.5), ('real-world', 2.5),
        ('real world', 2.5), ('observable', 2), ('measurable', 2.5),
        # Phrases
        ('based on facts', 3), ('look at details', 3), ('pay attention to', 2),
        ('tried and tested', 3), ('what has worked', 2.5), ('proven method', 3),
        ('follow instructions', 2), ('established process', 3), ('track record', 2),
    ],
    'N': [
        # Conceptual focus
        ('idea', 2), ('ideas', 2), ('possibility', 2.5), ('possibilities', 2.5),
        ('concept', 2), ('conceptual', 2), ('theory', 2), ('theoretical', 2),
        ('abstract', 2.5), ('imagine', 2), ('imagination', 2.5), ('imaginative', 2.5),
        ('vision', 2), ('visionary', 3), ('innovative', 2.5), ('innovation', 2.5),
        ('creative', 2), ('creativity', 2), ('intuition', 3), ('intuitive', 3),
        # Exploration
        ('explore', 2), ('experiment', 2), ('pattern', 2), ('patterns', 2),
        ('big picture', 3), ('overall', 1.5), ('future', 2), ('potential', 2),
        ('transform', 2), ('transformation', 2), ('inspiration', 2),
        ('brainstorm', 1.5), ('novel', 2), ('unconventional', 2.5),
        # Phrases
        ('think outside', 3), ('outside the box', 3), ('what if', 2),
        ('long term vision', 3), ('new approach', 2.5), ('new ways', 2),
        ('bigger meaning', 3), ('underlying pattern', 3), ('connect the dots', 3),
        ('see connections', 3), ('open to possibilities', 3), ('new perspective', 2.5),
    ],

    # ── Thinking vs Feeling ───────────────────────────────────
    'T': [
        # Logic
        ('logic', 2.5), ('logical', 2.5), ('logically', 2.5), ('analyze', 2),
        ('analysis', 2), ('analytical', 2.5), ('objective', 2.5), ('objectively', 2.5),
        ('rational', 2.5), ('rationally', 2.5), ('reason', 1.5), ('reasoning', 2),
        ('critical thinking', 3), ('critique', 2), ('evaluate', 1.5),
        # Efficiency
        ('efficient', 2), ('efficiency', 2), ('effective', 1.5), ('optimize', 2.5),
        ('solve', 1.5), ('solution', 1.5), ('problem-solving', 2.5),
        ('systematic', 2), ('strategy', 2), ('strategic', 2),
        # Approach
        ('fair', 1.5), ('consistent', 1.5), ('principle', 2), ('principles', 2),
        ('framework', 2), ('criteria', 2), ('standard', 1.5), ('pros and cons', 3),
        # Phrases
        ('based on logic', 3), ('weigh the options', 2.5), ('think critically', 3),
        ('cost benefit', 3), ('data driven', 3), ('make sense', 1.5),
        ('figure out', 1.5), ('break down the problem', 3), ('root cause', 3),
        ('separate emotion', 3), ('set aside feelings', 3),
    ],
    'F': [
        # Emotion
        ('feel', 1.5), ('feeling', 2), ('feelings', 2), ('emotion', 2),
        ('emotional', 2), ('empathy', 3), ('empathetic', 3), ('compassion', 3),
        ('compassionate', 3), ('care', 1.5), ('caring', 2), ('kind', 1.5),
        ('kindness', 2), ('sensitive', 2), ('sensitivity', 2),
        # Relationships
        ('harmony', 3), ('value', 1.5), ('values', 2), ('relationship', 2),
        ('relationships', 2), ('help', 1), ('helping', 1.5), ('support', 1.5),
        ('supportive', 2), ('understand', 1), ('understanding', 1.5),
        ('connect', 1.5), ('personal', 1), ('heart', 2),
        # Impact
        ('impact on people', 3), ('how others feel', 3), ('team morale', 3),
        ('wellbeing', 2.5), ('well-being', 2.5), ('consensus', 2),
        ('appreciation', 2), ('grateful', 1.5), ('trust', 1.5),
        # Phrases
        ('gut feeling', 3), ('listen to my heart', 3), ('consider others', 2.5),
        ('people first', 3), ('human side', 3), ('mean to others', 2),
        ('makes me feel', 2), ('passionate about', 2), ('deeply about', 2),
    ],

    # ── Judging vs Perceiving ─────────────────────────────────
    'J': [
        # Organization
        ('plan', 2), ('planning', 2), ('planned', 2), ('organize', 2.5),
        ('organized', 2.5), ('organization',
                             2), ('schedule', 2.5), ('scheduled', 2.5),
        ('structure', 2.5), ('structured', 2.5), ('systematic', 2), ('order', 1.5),
        ('orderly', 2), ('tidy', 1.5), ('neat', 1.5),
        # Goals
        ('goal', 2), ('goals', 2), ('deadline', 2.5), ('deadlines', 2.5),
        ('target', 1.5), ('milestone', 2), ('priority', 2), ('prioritize', 2.5),
        ('checklist', 3), ('to-do', 2.5), ('todo', 2.5),
        # Control
        ('decide', 1.5), ('decision', 1.5), ('decisive', 2.5), ('commit', 2),
        ('committed', 2), ('discipline', 2.5), ('disciplined', 2.5),
        ('responsible', 2), ('reliable', 2), ('dependable', 2),
        ('punctual', 2.5), ('routine', 2), ('prepared', 2), ('preparation', 2),
        # Phrases
        ('ahead of time', 3), ('in advance', 2.5), ('stick to the plan', 3),
        ('set clear goals', 3), ('follow through', 2.5), ('long-term plan', 3),
        ('like to know', 2), ('well prepared', 3), ('make a list', 2.5),
        ('clear expectations', 3), ('step by step plan', 3),
    ],
    'P': [
        # Flexibility
        ('flexible', 2.5), ('flexibility', 2.5), ('adapt', 2), ('adaptable', 2.5),
        ('adjust', 2), ('spontaneous', 3), ('spontaneously', 3), ('improvise', 3),
        ('improvisation', 3), ('open', 1), ('open-minded', 2.5), ('open minded', 2.5),
        ('casual', 1.5), ('relaxed', 2), ('easygoing', 2.5), ('easy-going', 2.5),
        # Exploration
        ('explore', 2), ('exploring', 2), ('flow', 2), ('go with the flow', 3),
        ('change', 1.5), ('changing', 1.5), ('variety', 2), ('diverse', 1.5),
        ('freedom', 2.5), ('free', 1.5), ('curious', 2), ('curiosity', 2),
        # Approach
        ('last minute', 2.5), ('play it by ear', 3), ('see what happens', 3),
        ('wing it', 3), ('keep options', 3), ('keep my options', 3),
        ('on the fly', 3), ('as it comes', 2.5), ('go along', 2),
        # Phrases
        ('no strict plan', 3), ('see how things go', 3),
        ('prefer not to plan', 3), ('like surprises', 2.5),
        ('open to change', 3), ('take things as they come', 3),
        ('new experiences', 2), ('try different', 2), ('not tied down', 3),
    ],
}


class PersonalityScorer:
    """Rule-based MBTI personality scorer using keyword analysis."""

    def __init__(self):
        # Pre-compile lowercase keyword lists for faster matching
        self._compiled = {}
        for dim_key, kw_list in DIMENSION_KEYWORDS.items():
            self._compiled[dim_key] = [(kw.lower(), w) for kw, w in kw_list]

    def _score_text(self, text, dim_key):
        """Score a text for a given dimension key (e.g. 'E', 'I')."""
        text_lower = text.lower()
        score = 0.0
        for keyword, weight in self._compiled[dim_key]:
            # Count occurrences (non-overlapping)
            count = text_lower.count(keyword)
            if count > 0:
                score += weight * count
        return score

    def score_answers(self, answers):
        """
        Score all answers and return dimension scores + MBTI type.

        Returns: (mbti_type, dimension_scores, confidence, top_matches)
          dimension_scores = {
              'EI': {'E': float, 'I': float, 'pct': 0-100 toward first letter},
              'SN': {...}, 'TF': {...}, 'JP': {...}
          }
        """
        # Accumulate raw scores per dimension letter
        raw_scores = {k: 0.0 for k in 'EISNTFJP'}

        for q_idx, answer in enumerate(answers):
            if not answer or not answer.strip():
                continue

            # Get relevant dimensions for this question
            relevant_dims = QUESTION_DIMENSION_MAP.get(q_idx, [0, 1, 2, 3])
            # Weight boost for primary dimension
            for rank, dim_idx in enumerate(relevant_dims):
                boost = 1.5 if rank == 0 else 1.0  # Primary dimension gets 1.5x

                dim_pairs = [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]
                pair = dim_pairs[dim_idx]

                for letter in pair:
                    raw_scores[letter] += self._score_text(
                        answer, letter) * boost

            # Also score all dimensions at base weight for general signal
            for letter in 'EISNTFJP':
                raw_scores[letter] += self._score_text(answer, letter) * 0.3

        # Build dimension scores
        dim_pairs_info = [('E', 'I', 'EI'), ('S', 'N', 'SN'),
                          ('T', 'F', 'TF'), ('J', 'P', 'JP')]

        dimension_scores = {}
        mbti_type = ''

        total_margin = 0.0
        for first, second, key in dim_pairs_info:
            s1 = raw_scores[first]
            s2 = raw_scores[second]
            total = s1 + s2

            if total == 0:
                pct = 50.0
            else:
                pct = (s1 / total) * 100.0

            dimension_scores[key] = {
                first: s1,
                second: s2,
                'pct': round(pct, 1),  # % toward first letter
                'winner': first if s1 >= s2 else second
            }

            mbti_type += first if s1 >= s2 else second
            margin = abs(s1 - s2) / max(total, 1)
            total_margin += margin

        # Confidence: average margin across 4 dimensions (0-1), scaled to 40-95%
        avg_margin = total_margin / 4.0
        confidence = min(95.0, 40.0 + avg_margin * 110)

        # Generate top 3 matches by calculating "closeness" to each type
        type_scores = {}
        for ptype in PERSONALITY_TYPES.keys():
            score = 0.0
            for i, (first, second, key) in enumerate(dim_pairs_info):
                pct = dimension_scores[key]['pct']
                # How well does this type match the dimension score?
                if ptype[i] == first:
                    score += pct
                else:
                    score += (100 - pct)
            type_scores[ptype] = score / 4.0  # Average percentage match

        sorted_types = sorted(type_scores.items(),
                              key=lambda x: x[1], reverse=True)
        top_matches = [(t, s) for t, s in sorted_types[:3]]

        return mbti_type, dimension_scores, confidence, top_matches


def _level_from_score(pct: float) -> str:
    if pct >= 70:
        return "High"
    if pct >= 40:
        return "Medium"
    return "Low"


def create_circular_progress(percentage, title, *,
                             ring_color="#6C63FF",
                             bg_ring="rgba(108,99,255,0.1)",
                             center_text="#E8E8ED"):
    """Donut KPI with auto subtitle based on score."""
    pct = max(0, min(100, float(percentage)))
    subtitle_text = _level_from_score(pct)

    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct],
        hole=0.72,
        sort=False,
        direction="clockwise",
        marker=dict(
            colors=[ring_color, bg_ring],
            line=dict(color="rgba(0,0,0,0)", width=0)
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
            font=dict(size=15, color=center_text, family="Inter, sans-serif")
        ),
        height=320,
        width=320,
        margin=dict(t=55, b=35, l=15, r=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    fig.add_annotation(
        x=0.5, y=0.56,
        text=f"<b>{pct:.0f}%</b>",
        showarrow=False,
        font=dict(size=36, color=center_text,
                  family="Space Grotesk, sans-serif"),
        xref="paper", yref="paper"
    )
    fig.add_annotation(
        x=0.5, y=0.40,
        text=f"<span style='color:#9BA1B7;font-size:13px'>{subtitle_text}</span>",
        showarrow=False,
        xref="paper", yref="paper"
    )

    return fig

# ─── HOME PAGE ────────────────────────────────────────────────────────────────


def show_home_page():
    """Display the home page"""
    st.markdown("""
    <div class="hero">
        <span class="badge">AI-Powered Analysis</span>
        <h1>Personality Detection<br>of Interviewee</h1>
        <p class="subtitle">
            Answer 20 thoughtfully crafted interview questions and our AI will
            reveal your MBTI personality type with detailed insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon purple">📊</div>
            <h5>NLP-Powered Analysis</h5>
            <p>Advanced natural language processing interprets the nuance behind your words.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon pink">⚡</div>
            <h5>Instant Results</h5>
            <p>Get your full personality profile in seconds after completing the assessment.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon green">🎯</div>
            <h5>Career Insights</h5>
            <p>Discover career paths, strengths, and growth areas tailored to your type.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works section
    st.markdown('<div class="section-label">How It Works</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">1</div>
            <h4 style="justify-content:center;">Answer Questions</h4>
            <p>Respond to 20 open-ended interview questions in your own words.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">2</div>
            <h4 style="justify-content:center;">AI Analysis</h4>
            <p>Our NLP engine processes your language patterns and word choices.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">3</div>
            <h4 style="justify-content:center;">Get Your Profile</h4>
            <p>Receive a detailed MBTI report with traits, careers, and insights.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("Start Personality Test  →", key="start_test", use_container_width=True):
            st.session_state.page = 'questionnaire'
            st.session_state.current_question = 0
            st.session_state.answers = [''] * 20
            st.query_params['page'] = 'questionnaire'
            st.rerun()


# ─── QUESTIONNAIRE PAGE ──────────────────────────────────────────────────────
def show_questionnaire_page():
    """Display the questionnaire page"""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = [''] * 20

    current_q = st.session_state.current_question
    progress = (current_q + 1) / len(QUESTIONS) * 100
    answered = sum(1 for a in st.session_state.answers if a.strip())

    # ── Compact Hero ──
    st.markdown(f"""
    <div class="hero" style="padding:2.5rem 2rem 2rem;">
        <span class="badge">Question {current_q + 1} of {len(QUESTIONS)}</span>
        <h1 style="font-size:2rem;">Personality Assessment</h1>
        <p class="subtitle" style="font-size:0.95rem;">
            {answered} of {len(QUESTIONS)} answered
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress bar ──
    st.markdown(f"""
    <div class="progress-track">
        <div class="progress-fill" style="width:{progress}%;"></div>
    </div>
    <div class="progress-label">{progress:.0f}% complete</div>
    """, unsafe_allow_html=True)

    # ── Step dots ──
    dots_html = '<div class="step-dots">'
    for i in range(len(QUESTIONS)):
        cls = "done" if i < current_q else ("active" if i == current_q else "")
        dots_html += f'<div class="step-dot {cls}"></div>'
    dots_html += '</div>'
    st.markdown(dots_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Question Card ──
    st.markdown(f"""
    <div class="q-card">
        <div class="q-number">{current_q + 1}</div>
        <div class="q-text">{QUESTIONS[current_q]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Answer Input ──
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    answer = st.text_area(
        "Your Answer",
        value=st.session_state.answers[current_q],
        height=160,
        placeholder="Type your answer here... Be as detailed as you'd like.",
        key=f"answer_{current_q}",
    )

    st.session_state.answers[current_q] = answer

    # ── Character counter ──
    char_count = len(answer.strip())
    is_valid = char_count >= 10
    color = "#43E97B" if is_valid else "#FF6584" if char_count > 0 else "var(--text-dim)"
    hint = "✓" if is_valid else "(min 10 — press Ctrl+Enter to apply)"
    st.markdown(f"""
    <div style="text-align:right; font-size:0.8rem; color:{color}; margin-top:-8px;">
        {char_count} characters {hint}
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_q > 0:
            if st.button("← Previous", key="prev_btn"):
                st.session_state.current_question -= 1
                st.rerun()
        else:
            if st.button("← Home", key="home_btn"):
                st.session_state.page = 'home'
                st.query_params.clear()
                st.rerun()

    with col3:
        if current_q < len(QUESTIONS) - 1:
            if st.button("Next →", key="next_btn", disabled=not is_valid):
                st.session_state.current_question += 1
                st.rerun()
        else:
            if st.button("Get My Results  →", key="results_btn", disabled=not is_valid):
                st.session_state.page = 'results'
                st.query_params['page'] = 'results'
                st.rerun()


# ─── RESULTS PAGE ────────────────────────────────────────────────────────────
def show_results_page():
    """Display the results page"""
    if 'personality_result' not in st.session_state:
        # Check if we have results encoded in query params (page refresh)
        qp = st.query_params
        if qp.get('type') and qp.get('conf') and qp.get('dims'):
            try:
                saved_type = qp['type']
                saved_conf = float(qp['conf'])
                saved_dims = json.loads(qp['dims'])
                saved_top = json.loads(qp.get('top', '[]'))
                if saved_type in PERSONALITY_TYPES:
                    st.session_state.personality_result = {
                        'type': saved_type,
                        'confidence': saved_conf,
                        'top_matches': [(t, s) for t, s in saved_top],
                        'dimension_scores': saved_dims,
                    }
            except (ValueError, json.JSONDecodeError, KeyError):
                pass

    if 'personality_result' not in st.session_state:
        scorer = PersonalityScorer()

        with st.spinner("Analyzing your personality..."):
            answers = st.session_state.get('answers', [''] * 20)
            valid_answers = [a for a in answers if a.strip()]

            if len(valid_answers) < 10:
                st.error(
                    "Not enough answers provided. Please complete more questions.")
                if st.button("← Back to Questions"):
                    st.session_state.page = 'questionnaire'
                    st.rerun()
                return

            mbti_type, dimension_scores, confidence, top_matches = scorer.score_answers(
                answers)

            result_data = {
                'type': mbti_type,
                'confidence': confidence,
                'top_matches': top_matches,
                'dimension_scores': dimension_scores,
            }
            st.session_state.personality_result = result_data

            # Persist to query params so results survive refresh
            st.query_params['page'] = 'results'
            st.query_params['type'] = mbti_type
            st.query_params['conf'] = str(round(confidence, 1))
            st.query_params['dims'] = json.dumps(dimension_scores)
            st.query_params['top'] = json.dumps(top_matches)

    result = st.session_state.personality_result
    personality_info = PERSONALITY_TYPES[result['type']]

    # ── Hero ──
    st.markdown("""
    <div class="hero" style="padding:3rem 2rem 2.5rem;">
        <span class="badge">Analysis Complete</span>
        <h1 style="font-size:2.4rem;">Your Personality Results</h1>
        <p class="subtitle" style="font-size:0.95rem;">Based on your interview responses</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Result ──
    col1, col2 = st.columns([2, 1])

    with col1:
        traits_html = ''.join(
            [f'<span class="pill purple">{t}</span>' for t in personality_info['traits']]
        )
        st.markdown(f"""
        <div class="glass-card">
            <div class="type-badge">{result['type']}</div>
            <div class="type-title">{personality_info['title']}</div>
            <p class="type-desc">{personality_info['description']}</p>
            <div style="margin-top:1.2rem;">
                <p style="color:var(--primary); font-weight:600; margin-bottom:0.5rem; font-size:0.9rem;">
                    KEY TRAITS
                </p>
                <div class="pill-wrap">{traits_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        confidence_fig = create_circular_progress(
            result['confidence'], 'Confidence Score')
        st.plotly_chart(confidence_fig, use_container_width=True)

    # ── Three Column Cards ──
    col1, col2, col3 = st.columns(3)

    with col1:
        strengths_html = ''.join(
            [f'<span class="pill green">{s}</span>' for s in personality_info['strengths']]
        )
        st.markdown(f"""
        <div class="glass-card">
            <h4>💪 Strengths</h4>
            <div class="pill-wrap">{strengths_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        careers_html = ''.join(
            [f'<span class="pill blue">{c}</span>' for c in personality_info['career_fits']]
        )
        st.markdown(f"""
        <div class="glass-card">
            <h4>💼 Career Fits</h4>
            <div class="pill-wrap">{careers_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        areas_html = ''.join(
            [f'<span class="pill amber">{a}</span>' for a in personality_info['areas_to_watch']]
        )
        st.markdown(f"""
        <div class="glass-card">
            <h4>⚠️ Areas to Watch</h4>
            <div class="pill-wrap">{areas_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Famous People & Top Matches ──
    col1, col2 = st.columns(2)

    with col1:
        people_html = ''.join(
            [f'<span class="pill pink">{p}</span>' for p in personality_info['famous_people']]
        )
        st.markdown(f"""
        <div class="glass-card">
            <h4>🌟 Famous People Like You</h4>
            <div class="pill-wrap" style="margin-top:0.6rem;">{people_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        matches_html = ""
        for match_type, percentage in result['top_matches']:
            matches_html += f'<div class="match-row"><span class="match-type">{match_type}</span><span class="match-pct">{percentage:.1f}%</span></div>'
        st.markdown(
            f'<div class="glass-card"><h4>🏆 Top Personality Matches</h4>{matches_html}</div>', unsafe_allow_html=True)

    # ── Personality Dimensions ──
    dim_pairs = [
        ('Extrovert vs Introvert', 'EI', 'E', 'I'),
        ('Sensing vs Intuition', 'SN', 'S', 'N'),
        ('Thinking vs Feeling', 'TF', 'T', 'F'),
        ('Judging vs Perceiving', 'JP', 'J', 'P')
    ]

    dim_scores = result.get('dimension_scores', {})

    dim_html = ""
    for label, key, first, second in dim_pairs:
        ds = dim_scores.get(key, {})
        pct = ds.get('pct', 50)
        winner = ds.get('winner', first)
        # pct is "% toward first letter"
        score_display = round(pct) if winner == first else round(100 - pct)
        bar_width = round(pct)
        pref_text = f"{winner} — {first} {round(pct)}% / {second} {round(100-pct)}%"

        dim_html += (f'<div class="dim-row">'
                     f'<div class="dim-header"><span class="dim-label">{label}</span>'
                     f'<span class="dim-value">{winner} ({score_display}%)</span></div>'
                     f'<div class="dim-track"><div class="dim-fill" style="width:{bar_width}%;"></div></div>'
                     f'<div class="dim-pref">{pref_text}</div></div>')

    st.markdown(
        f'<div class="glass-card" style="margin-top:0.5rem;"><h4>📊 Your Personality Dimensions</h4>{dim_html}</div>', unsafe_allow_html=True)

    # ── Restart ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("Take Test Again  ↻", key="restart_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.query_params.clear()
            st.session_state.page = 'home'
            st.rerun()


# ─── FOOTER ──────────────────────────────────────────────────────────────────
def show_footer():
    """Display the footer"""
    st.markdown("""
    <div class="app-footer">
        <p>Built with <span class="heart">♥</span> using Streamlit & NLP</p>
        <p>Privacy-Focused · Lightning-fast · Open Source</p>
        <p style="margin-top:0.6rem;">© 2025 Sukumar Divi. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    """Main application function"""
    # Restore page from query params on refresh
    if 'page' not in st.session_state:
        qp = st.query_params
        saved_page = qp.get('page', 'home')
        if saved_page in ('home', 'questionnaire', 'results'):
            st.session_state.page = saved_page
        else:
            st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'questionnaire':
        show_questionnaire_page()
    elif st.session_state.page == 'results':
        show_results_page()

    show_footer()


if __name__ == "__main__":
    main()
