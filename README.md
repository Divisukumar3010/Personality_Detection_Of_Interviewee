# Personality Detection of Interviewee

A sophisticated Streamlit web application that analyzes personality types through interview-style questions using AI-powered natural language processing and machine learning.

## 🌟 Features

- **Interactive Assessment**: 20 carefully crafted interview-style questions
- **AI Analysis**: Advanced NLP preprocessing with tokenization, stopword removal, and lemmatization
- **MBTI Classification**: Comprehensive personality type detection using Logistic Regression with TF-IDF
- **Professional Results**: Detailed analysis including traits, career fits, confidence scores, and famous personalities
- **Modern UI**: Clean, responsive design with smooth animations and professional styling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## 📁 Project Structure

```
personality-detection/
├── app.py                      # Main Streamlit application
├── model_training.ipynb        # Jupyter notebook for model training
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── personality_model.pkl       # Generated model files (after training)
├── tfidf_vectorizer.pkl
└── complete_personality_model.pkl
```

## 🧠 How It Works

### 1. NLP Preprocessing Pipeline
- **Tokenization**: Converts text responses into individual words
- **Stopword Removal**: Filters out common words that don't carry personality information
- **Lemmatization**: Reduces words to their base forms for better analysis
- **TF-IDF Vectorization**: Creates numerical features from preprocessed text

### 2. Machine Learning Model
- **Algorithm**: Logistic Regression with One-vs-Rest classification
- **Features**: TF-IDF vectors with unigrams and bigrams (max 1000 features)
- **Training**: Synthetic data generation based on personality-specific keywords
- **Validation**: Cross-validation and hyperparameter tuning for optimal performance

### 3. Personality Classification
- **16 MBTI Types**: Complete Myers-Briggs Type Indicator classification
- **Confidence Scoring**: Probability-based confidence measurement
- **Dimension Analysis**: Individual scores for E/I, S/N, T/F, J/P preferences

## 📊 Model Training

To train your own model or retrain with new data:

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook model_training.ipynb
   ```

2. **Run all cells** to:
   - Generate synthetic training data
   - Preprocess text using NLP pipeline
   - Train Logistic Regression model with TF-IDF
   - Evaluate model performance
   - Save trained model components

3. **Model outputs**:
   - Classification accuracy and detailed performance metrics
   - Feature importance analysis for each personality type
   - Confusion matrix visualization
   - Cross-validation results

## 🎯 Usage Instructions

### For Interviewees:
1. **Start**: Click "Start Personality Test" on the homepage
2. **Answer Questions**: Respond to 20 interview-style questions thoughtfully
3. **Navigate**: Use Previous/Next buttons to move between questions
4. **Complete**: Answer all questions to receive your personality analysis
5. **Review Results**: Explore comprehensive personality insights and recommendations

### For Interviewers:
- Use this tool to gain insights into candidate personalities
- Combine results with other assessment methods for comprehensive evaluation
- Consider personality fit for specific roles and team dynamics

## 🔧 Technical Details

### NLP Processing
- **Text Cleaning**: Removes special characters and normalizes case
- **Feature Extraction**: TF-IDF with n-gram range (1,2) for context capture
- **Dimensionality**: Maximum 1000 features to balance performance and accuracy

### Model Architecture
- **Base Algorithm**: Logistic Regression (scikit-learn)
- **Multi-class Strategy**: One-vs-Rest for 16-class classification
- **Regularization**: L2 regularization with optimized C parameter
- **Training Data**: Synthetic dataset with personality-specific keyword patterns

### Performance Metrics
- **Accuracy**: Overall classification accuracy across all personality types
- **Precision/Recall**: Per-class performance for balanced evaluation
- **Confidence Scoring**: Probability-based confidence measurement
- **Cross-validation**: 5-fold CV for robust performance estimation

## 🎨 Design Philosophy

- **Professional Aesthetic**: Clean, modern design suitable for corporate environments
- **User Experience**: Intuitive navigation with clear progress indicators
- **Accessibility**: Responsive design supporting all device types
- **Visual Feedback**: Interactive charts and progress visualizations

## 🔒 Privacy & Ethics

- **Data Privacy**: No personal data is stored beyond the current session
- **Transparency**: Clear explanation of model limitations and intended use
- **Ethical Use**: Tool designed for professional development, not discrimination
- **Consent**: Users voluntarily participate in the assessment

## 🛠️ Customization

### Adding New Questions
Edit the `QUESTIONS` list in `app.py` to modify or add interview questions.

### Personality Type Definitions
Update the `PERSONALITY_TYPES` dictionary to modify descriptions, traits, or career recommendations.

### Styling
Modify the CSS in the `st.markdown()` sections to customize the visual appearance.

### Model Improvement
- Collect real interview data for better training
- Experiment with different algorithms (Random Forest, SVM, Neural Networks)
- Add more sophisticated NLP features (sentiment analysis, named entity recognition)

## 📈 Future Enhancements

- **Real Data Integration**: Train on actual personality assessment datasets
- **Advanced NLP**: Implement transformer-based models (BERT, RoBERTa)
- **Multi-language Support**: Extend to support multiple languages
- **API Integration**: Create REST API for external system integration
- **Database Storage**: Add user session management and result history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

© 2025 Sukumar Divi. All rights reserved.

## 🆘 Support

For questions, issues, or suggestions:
- Review the code documentation
- Check the Jupyter notebook for model training details
- Ensure all dependencies are properly installed
- Verify Python version compatibility (3.8+)

## 🔍 Troubleshooting

### Common Issues:

1. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Module Import Errors**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Streamlit Issues**:
   ```bash
   streamlit cache clear
   ```

### Performance Tips:
- Ensure stable internet connection for initial NLTK downloads
- Close other applications to free up memory during model training
- Use the latest versions of dependencies for optimal performance