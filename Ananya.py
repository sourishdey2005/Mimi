"""
Medical Q&A Assistant (Streamlit single-file app) — Advanced
Created for: Context-grounded medical question-answer assistant with improved advice, remedies, follow-ups and triage
Author: ChatGPT (generated for user)

Features added in this version:
- Larger disease knowledge base and richer advice + home remedies + OTC suggestions (non-prescription only)
- Multi-turn, context-grounded follow-up questions (clarifying red flags and risk factors)
- Severity scoring and explicit triage recommendations (self-care / see GP / emergency)
- Export conversation and dataset, download trained model
- Optional ML model training on synthetic data (TF-IDF + RandomForest) for demonstrative predictions
- Clear safety reminders; no prescription dosages or controlled-med advice

Requirements (requirements.txt):
streamlit
pandas
numpy
scikit-learn
joblib
nltk

Run: streamlit run Medical_QA_Streamlit_App.py

Note: This is an educational demo; not a medical device. Always consult a qualified healthcare professional for diagnosis and treatment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import json
import io
import joblib
import nltk
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure NLTK tokenizer available
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

# ---------------------------
# Knowledge base and utilities
# ---------------------------
DISEASES = {
    'Common Cold': {
        'symptoms': ['sneezing', 'runny nose', 'sore throat', 'mild cough', 'congestion'],
        'advice': 'Rest, fluids, saline nasal spray, steam inhalation. Monitor fevers >3 days.',
        'home_remedies': ['Warm fluids', 'Saltwater gargle', 'Humidifier/steam'],
        'otc': ['paracetamol', 'ibuprofen', 'decongestant (short-term)'],
        'urgency': 'low'
    },
    'Influenza': {
        'symptoms': ['high fever', 'muscle aches', 'chills', 'severe cough', 'fatigue'],
        'advice': 'Rest, fluids, consult within 48 hours if high risk for antiviral therapy; seek care for breathing difficulty.',
        'home_remedies': ['Rest', 'Fluids', 'Warm compress for body aches'],
        'otc': ['paracetamol', 'ibuprofen'],
        'urgency': 'medium'
    },
    'COVID-19': {
        'symptoms': ['fever', 'dry cough', 'loss of taste', 'loss of smell', 'difficulty breathing'],
        'advice': 'Isolate per local guidelines, consider testing, monitor oxygenation. Seek urgent care for breathing difficulty.',
        'home_remedies': ['Rest', 'Hydration', 'Monitor symptoms'],
        'otc': ['paracetamol'],
        'urgency': 'medium'
    },
    'Pneumonia': {
        'symptoms': ['productive cough', 'high fever', 'chest pain', 'shortness of breath'],
        'advice': 'Often requires clinician evaluation, chest X-ray and antibiotics in bacterial cases. Seek prompt care.',
        'home_remedies': ['Rest', 'Fluids'],
        'otc': ['paracetamol'],
        'urgency': 'high'
    },
    'Gastroenteritis': {
        'symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'cramps'],
        'advice': 'Oral rehydration, avoid dairy/heavy foods; seek care for high fever, bloody stools, or severe dehydration.',
        'home_remedies': ['Oral rehydration solution (ORS)', 'BRAT diet (short-term)'],
        'otc': ['oral rehydration salts', 'antiemetic per clinician'],
        'urgency': 'medium'
    },
    'Appendicitis': {
        'symptoms': ['sharp abdominal pain', 'right lower abdomen pain', 'fever', 'loss of appetite', 'nausea'],
        'advice': 'Appendicitis is a surgical emergency — seek immediate medical evaluation.',
        'home_remedies': [],
        'otc': [],
        'urgency': 'high'
    },
    'Migraine': {
        'symptoms': ['intense headache', 'sensitivity to light', 'nausea', 'visual aura'],
        'advice': 'Rest in dark room, anti-nausea and analgesics as advised; seek care for first severe headache or neurological signs.',
        'home_remedies': ['Cold compress', 'Dark quiet room'],
        'otc': ['paracetamol', 'ibuprofen'],
        'urgency': 'low'
    },
    'Urinary Tract Infection': {
        'symptoms': ['burning urination', 'frequent urination', 'lower abdominal pain', 'cloudy urine'],
        'advice': 'Hydration and consult clinician for urine tests and antibiotics if confirmed.',
        'home_remedies': ['Hydration', 'Warm sitz baths for comfort'],
        'otc': ['phenazopyridine (symptom relief, clinician advised)'],
        'urgency': 'medium'
    },
    'Allergic Reaction': {
        'symptoms': ['rash', 'itching', 'swelling', 'hives', 'sneezing'],
        'advice': 'Antihistamines for mild cases; seek emergency care for facial/tongue swelling or breathing difficulty.',
        'home_remedies': ['Cool compress', 'Avoid trigger'],
        'otc': ['loratadine', 'cetirizine', 'diphenhydramine (sedating)'],
        'urgency': 'medium'
    },
    'Dengue Fever': {
        'symptoms': ['high fever', 'severe headache', 'pain behind eyes', 'joint pain', 'rash'],
        'advice': 'Maintain hydration; seek medical evaluation as dengue can cause bleeding and shock. Avoid NSAIDs due to bleeding risk.',
        'home_remedies': ['Hydration', 'Rest'],
        'otc': ['paracetamol only (avoid NSAIDs)'],
        'urgency': 'high'
    },
    'Malaria': {
        'symptoms': ['fever', 'chills', 'sweats', 'headache', 'nausea'],
        'advice': 'Urgent testing and antimalarial treatment required. Seek immediate care.',
        'home_remedies': ['Hydration'],
        'otc': [],
        'urgency': 'high'
    },
    'Hypertension': {
        'symptoms': ['headache', 'blurred vision', 'chest pain', 'shortness of breath'],
        'advice': 'Lifestyle changes and medical management; seek care for very high readings or chest pain.',
        'home_remedies': ['Reduce salt intake', 'Regular exercise'],
        'otc': [],
        'urgency': 'medium'
    },
    'Anemia': {
        'symptoms': ['fatigue', 'pallor', 'shortness of breath', 'dizziness'],
        'advice': 'Investigate cause (iron deficiency, chronic disease). Dietary iron and supplements per clinician.',
        'home_remedies': ['Iron-rich diet (spinach, legumes, meat if not vegetarian)'],
        'otc': ['iron supplements per doctor advice'],
        'urgency': 'medium'
    },
    'Otitis Media': {
        'symptoms': ['ear pain', 'fever', 'reduced hearing', 'ear discharge'],
        'advice': 'Pain control and ENT/GP assessment for recurrent or severe cases.',
        'home_remedies': ['Warm compress to the ear'],
        'otc': ['paracetamol', 'ibuprofen'],
        'urgency': 'medium'
    },
    'Eczema': {
        'symptoms': ['dry skin', 'itching', 'red patches', 'flaky skin'],
        'advice': 'Moisturizers, avoid triggers, topical steroids for flares under guidance.',
        'home_remedies': ['Regular emollient use', 'Avoid hot showers'],
        'otc': ['emollients', 'hydrocortisone 1% (short-term)'],
        'urgency': 'low'
    }
}

# Flatten knowledge for quick matching
KNOWLEDGE = []
for name, d in DISEASES.items():
    KNOWLEDGE.append({
        'label': name,
        'symptoms': d['symptoms'],
        'advice': d['advice'],
        'home_remedies': d['home_remedies'],
        'otc': d['otc'],
        'urgency': d['urgency']
    })

RISK_FACTORS = ['diabetes', 'hypertension', 'heart disease', 'asthma', 'immunocompromised', 'pregnant']

# ---------------------------
# Synthetic dataset generator (for optional model training)
# ---------------------------
SAMPLE_PHRASES = [
    'I have been feeling {symptom} for {duration} day(s).',
    'Experiencing {symptom} since {duration} days.',
    'Patient reports {symptom} and {symptom2} for {duration} day(s).',
    'I noticed {symptom} with {symptom2} and {symptom3}.'
]


def generate_case(condition, age_range=(1, 90)):
    label = condition['label']
    symptoms = random.sample(condition['symptoms'], k=random.randint(1, min(3, len(condition['symptoms']))))
    if random.random() < 0.25:
        other = random.choice(KNOWLEDGE)
        noise = random.sample(other['symptoms'], k=1)
        if noise[0] not in symptoms:
            symptoms.append(noise[0])
    duration = random.randint(1, 14)
    age = random.randint(*age_range)
    sex = random.choice(['male', 'female', 'other'])
    comorbid = []
    if random.random() < 0.25:
        comorbid = random.sample(RISK_FACTORS, k=random.randint(1, 2))
    template = random.choice(SAMPLE_PHRASES)
    fill = {
        'symptom': symptoms[0],
        'symptom2': symptoms[1] if len(symptoms) > 1 else symptoms[0],
        'symptom3': symptoms[2] if len(symptoms) > 2 else symptoms[0],
        'duration': duration
    }
    complaint = template.format(**fill)
    return {
        'age': age,
        'sex': sex,
        'symptoms_text': complaint,
        'symptoms_list': symptoms,
        'duration_days': duration,
        'comorbidities': ','.join(comorbid),
        'label': label,
        'urgency': condition['urgency']
    }


def generate_dataset(n=1500):
    rows = []
    for _ in range(n):
        cond = random.choice(KNOWLEDGE)
        case = generate_case(cond)
        rows.append(case)
    return pd.DataFrame(rows)

# ---------------------------
# Simple triage and follow-ups
# ---------------------------


def check_red_flags(symptoms_text, age, comorbidities_list):
    red_flag_keywords = [
        'difficulty breathing', 'shortness of breath', 'chest pain', 'loss of consciousness',
        'severe abdominal pain', 'blood', 'sudden weakness', 'slurred speech'
    ]
    text = symptoms_text.lower()
    for kw in red_flag_keywords:
        if kw in text:
            return True, kw
    if age < 2 or age > 75:
        return True, 'age risk'
    if 'pregnant' in comorbidities_list:
        return True, 'pregnancy'
    return False, None


def severity_score(symptoms_list, duration, comorbidities_list):
    score = 0
    for s in symptoms_list:
        if s.lower() in ['difficulty breathing', 'chest pain', 'severe abdominal pain', 'blood in sputum']:
            score += 3
        elif s.lower() in ['high fever', 'persistent vomiting', 'dehydration']:
            score += 2
        else:
            score += 1
    score += min(duration // 3, 3)
    score += len(comorbidities_list)
    return score

# ---------------------------
# Model utilities (optional)
# ---------------------------


@st.cache_data(show_spinner=False)
def train_model(df):
    X = df['symptoms_text'] + ' age:' + df['age'].astype(str) + ' dur:' + df['duration_days'].astype(str) + ' com:' + df['comorbidities'].fillna('')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=8000)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return pipeline, acc, report

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title='Medical Q&A Assistant — Advanced', layout='wide')
st.title('🩺 Medical Q&A Assistant — Advanced')
st.markdown('''
This enhanced demo collects context, asks follow-up questions, gives layered advice (home remedies, OTC options, when to see a clinician), and provides triage recommendations.
**Reminder:** Educational only — not a medical device.
''')

# Sidebar controls
with st.sidebar:
    st.header('Controls & Model')
    data_size = st.number_input('Synthetic dataset size', min_value=500, max_value=20000, value=1500, step=100)
    do_generate = st.button('Generate dataset & Train model')
    show_dataset = st.checkbox('Show sample dataset', value=False)
    st.markdown('---')
    st.write('Model & Exports')
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if st.button('Download trained model (.joblib)'):
        if st.session_state.model is None:
            st.warning('Train model first (use "Generate dataset & Train model")')
        else:
            buf = io.BytesIO()
            joblib.dump(st.session_state.model, buf)
            buf.seek(0)
            st.download_button('Download model file', data=buf, file_name='medical_qa_model.joblib')

# Generate and train
if do_generate:
    with st.spinner('Generating dataset and training model...'):
        df = generate_dataset(int(data_size))
        st.session_state.df = df
        pipeline, acc, report = train_model(df)
        st.session_state.model = pipeline
        st.success(f'Training complete. Test accuracy ≈ {acc:.2f}')
        st.text('Classification report:')
        st.text(report)

if show_dataset and st.session_state.get('df') is not None:
    st.subheader('Sample synthetic dataset')
    st.dataframe(st.session_state.df.sample(min(50, len(st.session_state.df))))
    csv_buf = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button('Download dataset CSV', data=csv_buf, file_name='synthetic_medical_dataset.csv')

# Conversation / context
if 'history' not in st.session_state:
    st.session_state.history = []

st.header('Start a conversation')

with st.form('patient_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
        sex = st.selectbox('Sex', ['male', 'female', 'other'])
    with col2:
        duration = st.number_input('How many days since symptoms started?', min_value=0, max_value=365, value=2)
        comorbidities = st.multiselect('Comorbidities (if any)', options=RISK_FACTORS)
    with col3:
        symptom_text = st.text_area('Describe your main symptoms in a sentence or two', value='I have a sore throat and runny nose for 2 days.')

    submit_btn = st.form_submit_button('Submit context')

if submit_btn:
    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'age': int(age),
        'sex': sex,
        'duration_days': int(duration),
        'comorbidities': ','.join(comorbidities),
        'symptoms_text': symptom_text
    }
    st.session_state.history.append({'role': 'patient', 'text': json.dumps(entry)})
    st.success('Context saved. The assistant may ask clarifying follow-up questions.')

# follow-up question logic
if st.session_state.history:
    last = json.loads(st.session_state.history[-1]['text'])
    # quick red-flag check
    com_list = last['comorbidities'].split(',') if last['comorbidities'] else []
    red_flag, keyword = check_red_flags(last['symptoms_text'], last['age'], com_list)
    if red_flag:
        st.warning('Potential red flag detected: ' + str(keyword))
        st.info('If you or the patient have this symptom, seek urgent medical care or call emergency services.')

    st.subheader('Ask the assistant a question')
    user_q = st.text_input('Your question (e.g., What could this be? How to manage at home?)')
    if st.button('Ask'):
        if not user_q.strip():
            st.warning('Please type a question.')
        else:
            ctx_text = last['symptoms_text'] + ' age:' + str(last['age']) + ' dur:' + str(last['duration_days']) + ' com:' + last['comorbidities']
            model = st.session_state.get('model')
            # Rule-based match first
            matched = None
            for item in KNOWLEDGE:
                for s in item['symptoms']:
                    if s.lower() in ctx_text.lower():
                        matched = item
                        break
                if matched:
                    break
            # If model available, get model prediction as well
            model_pred = None
            if model is not None:
                try:
                    model_pred = model.predict([ctx_text])[0]
                except Exception:
                    model_pred = None
            # Build assistant response
            if matched is None and model_pred is None:
                response_text = (
                    "**Preliminary possibility:** Undifferentiated/Uncertain\n\n"
                    "**Advice:** Unable to suggest a likely condition based on current information. Please see a clinician for evaluation.\n\n"
                    "**Reminder:** This is educational only."
                )
            else:
                # Prefer rule-based matched item, else model
                chosen_label = matched['label'] if matched is not None else model_pred
                if matched is not None:
                    advice = matched['advice']
                    home = matched['home_remedies']
                    otc = matched['otc']
                    urgency = matched['urgency']
                else:
                    # fallback look up in KNOWLEDGE by label
                    lab = model_pred
                    found = next((x for x in KNOWLEDGE if x['label'] == lab), None)
                    if found:
                        advice = found['advice']
                        home = found['home_remedies']
                        otc = found['otc']
                        urgency = found['urgency']
                    else:
                        advice = 'Supportive care and clinician evaluation.'
                        home = []
                        otc = []
                        urgency = 'medium'

                # parse user symptoms into list (simple split)
                sym_list = [s.strip() for s in last['symptoms_text'].split() if s.strip()]
                sev = severity_score(sym_list, last['duration_days'], com_list)
                triage = 'self-care'
                if sev >= 7 or urgency == 'high':
                    triage = 'emergency' if urgency == 'high' else 'see clinician'
                elif sev >= 4:
                    triage = 'see clinician'

                response_text = (
                    f"**Preliminary suggestion:** {chosen_label}\n\n"
                    f"**Advice (educational):** {advice}\n\n"
                    f"**Home remedies:** {', '.join(home) if home else 'None suggested'}\n\n"
                    f"**OTC (non-prescription) options:** {', '.join(otc) if otc else 'None suggested — consult clinician'}\n\n"
                    f"**Triage recommendation:** {triage}\n\n"
                    f"**Severity score (est.):** {sev} — higher means more urgent.\n\n"
                    f"**Reminder:** This guidance is educational only. For diagnosis or prescription, see a healthcare professional."
                )

            st.session_state.history.append({'role': 'assistant', 'text': response_text})
            st.markdown('### Assistant response')
            st.markdown(response_text)

    st.markdown('---')
    st.subheader('Conversation history')
    for turn in st.session_state.history[-12:]:
        if turn['role'] == 'patient':
            obj = json.loads(turn['text'])
            # Use \n for newlines inside a single string literal
            st.info(
                f"Patient — age: {obj['age']}, sex: {obj['sex']}, duration: {obj['duration_days']}d\n"
                f"Symptoms: {obj['symptoms_text']}\n"
                f"Comorbidities: {obj['comorbidities']}"
            )
        else:
            st.success('Assistant — ')
            st.markdown(turn['text'])

    # Export conversation
    if st.button('Export conversation (JSON)'):
        out = {'history': st.session_state.history}
        b = io.BytesIO(json.dumps(out, indent=2).encode('utf-8'))
        st.download_button('Download conversation JSON', data=b, file_name='conversation.json')

# Developer notes
st.sidebar.markdown('---')
if st.sidebar.checkbox('Show developer notes'):
    st.sidebar.markdown('''
    Developer notes:
    - This prototype uses a mix of rule-based symptom matching and an optional synthetic-trained model.
    - The advice, home remedies and OTC suggestions are educational and intentionally generic. Do NOT include precise prescription dosages.
    - Improve by adding real clinical datasets, better symptom parsing (medical NLP), and validation with clinicians.
    ''')

# Footer
st.markdown('---')
st.caption('Educational demo only — not a medical device. Always consult a healthcare professional for diagnosis and treatment. Made By Ananya Raj ')
