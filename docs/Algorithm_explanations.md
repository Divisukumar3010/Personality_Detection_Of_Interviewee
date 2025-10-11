# Detailed Algorithm Explanations and Mathematical Foundations

## Overview
This document provides in-depth explanations of all algorithms used in the personality detection system, including mathematical foundations, implementation details, and practical applications.

## 1. Logistic Regression Algorithm

### 1.1 Mathematical Foundation

**Sigmoid Function**:
```
    σ(z) = 1 / (1 + e^(-z))
where z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**Probability Calculation**:
```
P(y=1|x) = σ(w^T x + b) = 1 / (1 + e^(-(w^T x + b)))
P(y=0|x) = 1 - P(y=1|x)
```

**Multi-class Extension (One-vs-Rest)**:
```
For class k: P(y=k|x) = σ(w_k^T x + b_k)
Final prediction: argmax_k P(y=k|x)
```

### 1.2 Cost Function and Optimization

**Log-Likelihood Cost Function**:
```
J(w) = -1/m * Σ[y_i * log(h_w(x_i)) + (1-y_i) * log(1-h_w(x_i))]

Where:
- m = number of training examples
- h_w(x_i) = σ(w^T x_i + b) = predicted probability
- y_i = actual label (0 or 1)
```

**Gradient Calculation**:
```
∂J/∂w = 1/m * X^T * (h_w(X) - y)
∂J/∂b = 1/m * Σ(h_w(x_i) - y_i)
```

**Gradient Descent Update**:
```
w := w - α * ∂J/∂w
b := b - α * ∂J/∂b

Where α = learning rate
```

### 1.3 Regularization Mathematics

**L2 Regularization (Ridge)**:
```
J_regularized(w) = J(w) + λ * Σ(w_i²)

Gradient with L2:
∂J_regularized/∂w = ∂J/∂w + 2λw
```

**L1 Regularization (Lasso)**:
```
J_regularized(w) = J(w) + λ * Σ|w_i|

Gradient with L1:
∂J_regularized/∂w = ∂J/∂w + λ * sign(w)
```

**Elastic Net (Combined L1 + L2)**:
```
J_regularized(w) = J(w) + λ₁ * Σ|w_i| + λ₂ * Σ(w_i²)
```

### 1.4 Implementation in Personality Detection

```python
class LogisticRegressionPersonality:
    def __init__(self, regularization='l2', C=1.0):
        self.C = C  # Inverse of regularization strength
        self.regularization = regularization
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, learning_rate=0.01, max_iter=1000):
        """Train the logistic regression model"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize weights for each class (One-vs-Rest)
        self.weights = np.random.normal(0, 0.01, (n_classes, n_features))
        self.bias = np.zeros(n_classes)
        
        for iteration in range(max_iter):
            for class_idx in range(n_classes):
                # Create binary labels for current class
                binary_y = (y == class_idx).astype(int)
                
                # Forward pass
                z = X.dot(self.weights[class_idx]) + self.bias[class_idx]
                predictions = self.sigmoid(z)
                
                # Calculate gradients
                dw = (1/n_samples) * X.T.dot(predictions - binary_y)
                db = (1/n_samples) * np.sum(predictions - binary_y)
                
                # Add regularization
                if self.regularization == 'l2':
                    dw += (2/self.C) * self.weights[class_idx]
                elif self.regularization == 'l1':
                    dw += (1/self.C) * np.sign(self.weights[class_idx])
                
                # Update weights
                self.weights[class_idx] -= learning_rate * dw
                self.bias[class_idx] -= learning_rate * db
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        scores = X.dot(self.weights.T) + self.bias
        probabilities = self.sigmoid(scores)
        
        # Normalize probabilities across classes
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        """Make class predictions"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
```

## 2. TF-IDF Algorithm Deep Dive

### 2.1 Mathematical Formulation

**Term Frequency (TF)**:
```
TF(t,d) = f(t,d) / Σ f(w,d)

Where:
- f(t,d) = frequency of term t in document d
- Σ f(w,d) = total number of terms in document d
```

**Inverse Document Frequency (IDF)**:
```
IDF(t,D) = log(|D| / |{d ∈ D : t ∈ d}|)

Where:
- |D| = total number of documents
- |{d ∈ D : t ∈ d}| = number of documents containing term t
```

**TF-IDF Score**:
```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

### 2.2 Normalization Techniques

**L2 Normalization**:
```
TF-IDF_normalized(t,d,D) = TF-IDF(t,d,D) / √(Σ TF-IDF(w,d,D)²)
```

**Why L2 Normalization**:
- Makes documents comparable regardless of length
- Prevents longer responses from dominating
- Ensures feature vectors have unit length
- Improves cosine similarity calculations

### 2.3 Implementation with Personality-Specific Optimizations

```python
class PersonalityTFIDF:
    def __init__(self, max_features=1000, ngram_range=(1,2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = {}
        self.idf_values = {}
        self.feature_names = []
    
    def fit(self, documents):
        """Fit TF-IDF on personality-specific corpus"""
        
        # Step 1: Build vocabulary with n-grams
        self.vocabulary = self._build_vocabulary(documents)
        
        # Step 2: Calculate IDF values
        self.idf_values = self._calculate_idf(documents)
        
        # Step 3: Select top features
        self.feature_names = self._select_top_features()
    
    def _build_vocabulary(self, documents):
        """Build vocabulary including personality-relevant terms"""
        vocabulary = {}
        
        for doc in documents:
            # Extract unigrams
            words = doc.split()
            for word in words:
                vocabulary[word] = vocabulary.get(word, 0) + 1
            
            # Extract bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                vocabulary[bigram] = vocabulary.get(bigram, 0) + 1
        
        return vocabulary
    
    def _calculate_idf(self, documents):
        """Calculate IDF values for all terms"""
        doc_count = len(documents)
        idf_values = {}
        
        for term in self.vocabulary:
            # Count documents containing this term
            docs_with_term = sum(1 for doc in documents if term in doc)
            
            # Calculate IDF with smoothing
            idf = np.log(doc_count / (docs_with_term + 1)) + 1
            idf_values[term] = idf
        
        return idf_values
    
    def transform(self, documents):
        """Transform documents to TF-IDF vectors"""
        feature_vectors = []
        
        for doc in documents:
            vector = np.zeros(len(self.feature_names))
            words = doc.split()
            
            # Calculate TF for each feature
            for i, feature in enumerate(self.feature_names):
                if feature in doc:
                    # Calculate term frequency
                    tf = doc.count(feature) / len(words)
                    
                    # Get IDF value
                    idf = self.idf_values.get(feature, 0)
                    
                    # Calculate TF-IDF
                    vector[i] = tf * idf
            
            # L2 normalization
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            feature_vectors.append(vector)
        
        return np.array(feature_vectors)
```

### 2.4 Personality-Specific Feature Weighting

```python
def apply_personality_weights(tfidf_matrix, feature_names):
    """Apply personality-specific weights to features"""
    
    # Define personality-relevant terms and their weights
    personality_weights = {
        'strategic': 2.0,      # Strong indicator for NT types
        'creative': 1.8,       # Strong indicator for NF types
        'practical': 1.5,      # Strong indicator for ST types
        'harmony': 1.7,        # Strong indicator for SF types
        'independent': 1.6,    # Strong indicator for I types
        'team': 1.4,          # Strong indicator for E types
        'plan': 1.5,          # Strong indicator for J types
        'flexible': 1.3       # Strong indicator for P types
    }
    
    # Apply weights to relevant features
    weighted_matrix = tfidf_matrix.copy()
    
    for i, feature in enumerate(feature_names):
        for term, weight in personality_weights.items():
            if term in feature:
                weighted_matrix[:, i] *= weight
    
    return weighted_matrix
```

## 3. Natural Language Processing Algorithms

### 3.1 Tokenization Algorithm

**Word Boundary Detection**:
```python
import re

def advanced_tokenize(text):
    """Advanced tokenization with personality-specific rules"""
    
    # Step 1: Handle contractions
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Step 2: Split on word boundaries
    # Keep personality-relevant punctuation patterns
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Step 3: Handle special cases
    processed_tokens = []
    for token in tokens:
        # Keep meaningful short words for personality detection
        if len(token) >= 2 or token in ['i', 'a']:
            processed_tokens.append(token)
    
    return processed_tokens
```

**Sentence Boundary Detection**:
```python
def sentence_tokenize(text):
    """Split text into sentences for analysis"""
    
    # Simple sentence splitting with personality context
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 5:  # Minimum sentence length
            clean_sentences.append(sentence)
    
    return clean_sentences
```

### 3.2 Stopword Removal Algorithm

**Personality-Aware Stopword Filtering**:
```python
class PersonalityStopwords:
    def __init__(self):
        # Standard English stopwords
        self.standard_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # Keep personality-relevant words that might be in standard stopwords
        self.personality_relevant = {
            'i', 'me', 'my', 'myself',  # Self-reference important for personality
            'we', 'us', 'our',         # Team orientation indicators
            'you', 'your',             # Other-focus indicators
            'very', 'really', 'quite', # Intensity modifiers
            'always', 'never', 'often', 'sometimes'  # Frequency indicators
        }
        
        # Final stopword set
        self.stopwords = self.standard_stopwords - self.personality_relevant
    
    def remove_stopwords(self, tokens):
        """Remove stopwords while preserving personality indicators"""
        return [token for token in tokens if token not in self.stopwords]
```

### 3.3 Lemmatization Algorithm

**Rule-Based Lemmatization**:
```python
class PersonalityLemmatizer:
    def __init__(self):
        # Personality-relevant lemmatization rules
        self.lemma_rules = {
            # Verb forms
            'working': 'work', 'worked': 'work', 'works': 'work',
            'thinking': 'think', 'thought': 'think', 'thinks': 'think',
            'planning': 'plan', 'planned': 'plan', 'plans': 'plan',
            'leading': 'lead', 'led': 'lead', 'leads': 'lead',
            'organizing': 'organize', 'organized': 'organize', 'organizes': 'organize',
            'creating': 'create', 'created': 'create', 'creates': 'create',
            'analyzing': 'analyze', 'analyzed': 'analyze', 'analyzes': 'analyze',
            
            # Adjective forms
            'better': 'good', 'best': 'good',
            'larger': 'large', 'largest': 'large',
            'smaller': 'small', 'smallest': 'small',
            'stronger': 'strong', 'strongest': 'strong',
            'creative': 'creative', 'creativity': 'creative',
            'logical': 'logic', 'logically': 'logic',
            
            # Noun forms
            'teams': 'team', 'groups': 'group',
            'ideas': 'idea', 'concepts': 'concept',
            'strategies': 'strategy', 'plans': 'plan',
            'solutions': 'solution', 'problems': 'problem'
        }
    
    def lemmatize(self, word):
        """Apply lemmatization rules"""
        # Check direct mapping first
        if word in self.lemma_rules:
            return self.lemma_rules[word]
        
        # Apply suffix rules
        return self._apply_suffix_rules(word)
    
    def _apply_suffix_rules(self, word):
        """Apply suffix-based lemmatization"""
        # Remove common suffixes
        if word.endswith('ing') and len(word) > 4:
            base = word[:-3]
            # Check if base word is valid
            if len(base) >= 3:
                return base
        
        if word.endswith('ed') and len(word) > 3:
            base = word[:-2]
            if len(base) >= 3:
                return base
        
        if word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
            return word[:-1]
        
        return word
```

### 3.4 Advanced Text Processing

**N-gram Generation Algorithm**:
```python
def generate_ngrams(tokens, n):
    """Generate n-grams from token list"""
    ngrams = []
    
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def extract_personality_ngrams(text, max_n=2):
    """Extract personality-relevant n-grams"""
    tokens = tokenize(text)
    all_ngrams = []
    
    # Generate unigrams and bigrams
    for n in range(1, max_n + 1):
        ngrams = generate_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    
    # Filter for personality relevance
    personality_ngrams = []
    personality_keywords = {
        'work_style': ['work alone', 'team work', 'independent work'],
        'decision_making': ['logical decision', 'intuitive choice', 'data driven'],
        'planning': ['detailed plan', 'flexible approach', 'structured method'],
        'communication': ['express feeling', 'share idea', 'discuss openly']
    }
    
    for ngram in all_ngrams:
        for category, keywords in personality_keywords.items():
            if any(keyword in ngram for keyword in keywords):
                personality_ngrams.append(ngram)
    
    return personality_ngrams
```

## 4. Synthetic Data Generation Algorithm

### 4.1 Personality-Based Text Generation

**Keyword-Based Generation**:
```python
class SyntheticDataGenerator:
    def __init__(self):
        self.personality_keywords = {
            'INTJ': {
                'primary': ['strategic', 'independent', 'plan', 'analyze', 'system', 'efficient'],
                'secondary': ['long-term', 'vision', 'improve', 'design', 'structure'],
                'context': ['project', 'goal', 'method', 'approach', 'solution']
            },
            'ENFP': {
                'primary': ['creative', 'people', 'energy', 'possibility', 'inspire', 'enthusiastic'],
                'secondary': ['new', 'exciting', 'potential', 'innovative', 'collaborative'],
                'context': ['idea', 'opportunity', 'team', 'future', 'change']
            },
            # ... definitions for all 16 types
        }
    
    def generate_response(self, personality_type, question_context):
        """Generate synthetic response for specific personality type"""
        
        keywords = self.personality_keywords[personality_type]
        
        # Select keywords based on question context
        primary_words = np.random.choice(keywords['primary'], size=3, replace=False)
        secondary_words = np.random.choice(keywords['secondary'], size=2, replace=False)
        context_words = np.random.choice(keywords['context'], size=3, replace=False)
        
        # Create response template based on personality type
        if personality_type.startswith('E'):  # Extraverted
            template = f"I really enjoy {primary_words[0]} with {context_words[0]}. " \
                      f"When I {secondary_words[0]}, I like to {primary_words[1]} and " \
                      f"{primary_words[2]} with others. This helps me {secondary_words[1]} " \
                      f"and achieve better {context_words[1]}."
        else:  # Introverted
            template = f"I prefer to {primary_words[0]} independently on {context_words[0]}. " \
                      f"My approach involves {secondary_words[0]} and {primary_words[1]} " \
                      f"to ensure {primary_words[2]} results. I find this method " \
                      f"helps me {secondary_words[1]} effectively."
        
        return template
    
    def generate_dataset(self, samples_per_type=100):
        """Generate complete synthetic dataset"""
        
        data = []
        labels = []
        
        for personality_type in self.personality_keywords.keys():
            for _ in range(samples_per_type):
                # Generate response for random question context
                question_contexts = ['work', 'team', 'problem', 'decision', 'goal']
                context = np.random.choice(question_contexts)
                
                response = self.generate_response(personality_type, context)
                
                data.append(response)
                labels.append(personality_type)
        
        return data, labels
```

### 4.2 Data Augmentation Techniques

**Synonym Replacement**:
```python
def augment_with_synonyms(text, replacement_rate=0.1):
    """Replace words with synonyms for data augmentation"""
    
    synonym_dict = {
        'work': ['job', 'task', 'project', 'assignment'],
        'team': ['group', 'colleagues', 'members', 'crew'],
        'plan': ['strategy', 'scheme', 'design', 'blueprint'],
        'create': ['make', 'build', 'develop', 'generate'],
        'analyze': ['examine', 'study', 'evaluate', 'assess'],
        'organize': ['arrange', 'structure', 'coordinate', 'manage']
    }
    
    words = text.split()
    augmented_words = []
    
    for word in words:
        if word in synonym_dict and np.random.random() < replacement_rate:
            synonym = np.random.choice(synonym_dict[word])
            augmented_words.append(synonym)
        else:
            augmented_words.append(word)
    
    return ' '.join(augmented_words)
```

**Sentence Reordering**:
```python
def augment_with_reordering(text, reorder_probability=0.3):
    """Reorder sentences for data augmentation"""
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 1 and np.random.random() < reorder_probability:
        np.random.shuffle(sentences)
    
    return '. '.join(sentences) + '.'
```

## 5. Confidence Score Calculation Algorithm

### 5.1 Probability-Based Confidence

**Maximum Probability Method**:
```python
def calculate_confidence_max_prob(probabilities):
    """Calculate confidence based on maximum probability"""
    max_prob = np.max(probabilities)
    confidence = max_prob * 100
    return confidence
```

**Entropy-Based Confidence**:
```python
def calculate_confidence_entropy(probabilities):
    """Calculate confidence based on probability distribution entropy"""
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    # Maximum entropy for uniform distribution over 16 classes
    max_entropy = np.log(16)
    
    # Confidence is inverse of normalized entropy
    normalized_entropy = entropy / max_entropy
    confidence = (1 - normalized_entropy) * 100
    
    return confidence
```

**Margin-Based Confidence**:
```python
def calculate_confidence_margin(probabilities):
    """Calculate confidence based on margin between top predictions"""
    
    sorted_probs = np.sort(probabilities)[::-1]  # Sort descending
    
    # Margin between top two predictions
    margin = sorted_probs[0] - sorted_probs[1]
    
    # Normalize margin to confidence score
    confidence = (margin * 100) + 50  # Scale and shift
    confidence = np.clip(confidence, 0, 100)
    
    return confidence
```

### 5.2 Combined Confidence Score

```python
def calculate_combined_confidence(probabilities, weights=None):
    """Calculate confidence using multiple methods"""
    
    if weights is None:
        weights = {'max_prob': 0.5, 'entropy': 0.3, 'margin': 0.2}
    
    # Calculate individual confidence scores
    conf_max = calculate_confidence_max_prob(probabilities)
    conf_entropy = calculate_confidence_entropy(probabilities)
    conf_margin = calculate_confidence_margin(probabilities)
    
    # Weighted combination
    combined_confidence = (
        weights['max_prob'] * conf_max +
        weights['entropy'] * conf_entropy +
        weights['margin'] * conf_margin
    )
    
    return round(combined_confidence)
```

### 5.3 Confidence Calibration

**Platt Scaling**:
```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_confidence_scores(model, X_val, y_val):
    """Calibrate confidence scores using Platt scaling"""
    
    # Create calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='platt',  # Platt scaling
        cv=3            # 3-fold cross-validation
    )
    
    # Fit calibration
    calibrated_model.fit(X_val, y_val)
    
    return calibrated_model

def apply_confidence_calibration(raw_probabilities):
    """Apply calibration to raw probabilities"""
    
    # Platt scaling formula: P_calibrated = 1 / (1 + exp(A * P_raw + B))
    # Where A and B are learned parameters
    
    A, B = -2.5, 1.2  # Example learned parameters
    
    calibrated_probs = []
    for prob in raw_probabilities:
        calibrated = 1 / (1 + np.exp(A * prob + B))
        calibrated_probs.append(calibrated)
    
    # Renormalize
    calibrated_probs = np.array(calibrated_probs)
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
    
    return calibrated_probs
```

## 6. Personality Dimension Analysis Algorithm

### 6.1 MBTI Dimension Extraction

**Four-Dimension Analysis**:
```python
def analyze_mbti_dimensions(text_responses):
    """Analyze individual MBTI dimensions from text"""
    
    dimension_keywords = {
        'E': ['team', 'people', 'social', 'group', 'collaborate', 'discuss', 'share'],
        'I': ['alone', 'individual', 'quiet', 'focus', 'independent', 'private'],
        'S': ['practical', 'concrete', 'detail', 'fact', 'data', 'step', 'specific'],
        'N': ['possibility', 'idea', 'concept', 'theory', 'creative', 'vision', 'future'],
        'T': ['logic', 'analyze', 'objective', 'rational', 'efficient', 'solve'],
        'F': ['feel', 'emotion', 'value', 'harmony', 'care', 'relationship'],
        'J': ['plan', 'organize', 'schedule', 'structure', 'deadline', 'goal'],
        'P': ['flexible', 'adapt', 'spontaneous', 'open', 'explore', 'change']
    }
    
    dimension_scores = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    
    # Combine all responses
    combined_text = ' '.join(text_responses).lower()
    
    # Count keyword occurrences
    for dimension, keywords in dimension_keywords.items():
        for keyword in keywords:
            count = combined_text.count(keyword)
            dimension_scores[dimension] += count
    
    # Calculate preference scores
    ei_score = dimension_scores['E'] / (dimension_scores['E'] + dimension_scores['I'] + 1)
    sn_score = dimension_scores['S'] / (dimension_scores['S'] + dimension_scores['N'] + 1)
    tf_score = dimension_scores['T'] / (dimension_scores['T'] + dimension_scores['F'] + 1)
    jp_score = dimension_scores['J'] / (dimension_scores['J'] + dimension_scores['P'] + 1)
    
    return {
        'EI': {'score': ei_score * 100, 'preference': 'E' if ei_score > 0.5 else 'I'},
        'SN': {'score': sn_score * 100, 'preference': 'S' if sn_score > 0.5 else 'N'},
        'TF': {'score': tf_score * 100, 'preference': 'T' if tf_score > 0.5 else 'F'},
        'JP': {'score': jp_score * 100, 'preference': 'J' if jp_score > 0.5 else 'P'}
    }
```

### 6.2 Weighted Dimension Analysis

```python
def weighted_dimension_analysis(responses, question_weights):
    """Analyze dimensions with question-specific weights"""
    
    # Different questions have different relevance to each dimension
    question_dimension_weights = {
        0: {'EI': 0.8, 'SN': 0.2, 'TF': 0.3, 'JP': 0.1},  # "Tell me about yourself"
        1: {'EI': 0.1, 'SN': 0.3, 'TF': 0.2, 'JP': 0.9},  # "How do you prepare"
        2: {'EI': 0.9, 'SN': 0.1, 'TF': 0.2, 'JP': 0.1},  # "Work alone or team"
        # ... weights for all 20 questions
    }
    
    weighted_scores = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    
    for i, response in enumerate(responses):
        if i in question_dimension_weights:
            weights = question_dimension_weights[i]
            dimension_analysis = analyze_single_response(response)
            
            # Apply question-specific weights
            for dim_pair, weight in weights.items():
                dim1, dim2 = dim_pair[0], dim_pair[1]
                score1 = dimension_analysis.get(dim1, 0)
                score2 = dimension_analysis.get(dim2, 0)
                
                weighted_scores[dim1] += score1 * weight
                weighted_scores[dim2] += score2 * weight
    
    return weighted_scores
```

## 7. Model Evaluation Algorithms

### 7.1 Cross-Validation Implementation

**Stratified K-Fold for Text Classification**:
```python
def stratified_cross_validation(X, y, model, k=5):
    """Implement stratified k-fold cross-validation"""
    
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_scores = []
    fold_predictions = []
    fold_true_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{k}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on fold
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation set
        y_pred_fold = fold_model.predict(X_val_fold)
        fold_score = accuracy_score(y_val_fold, y_pred_fold)
        
        fold_scores.append(fold_score)
        fold_predictions.extend(y_pred_fold)
        fold_true_labels.extend(y_val_fold)
    
    # Calculate overall metrics
    overall_accuracy = np.mean(fold_scores)
    std_accuracy = np.std(fold_scores)
    
    return {
        'mean_accuracy': overall_accuracy,
        'std_accuracy': std_accuracy,
        'fold_scores': fold_scores,
        'all_predictions': fold_predictions,
        'all_true_labels': fold_true_labels
    }
```

### 7.2 Performance Metrics Calculation

**Multi-class Precision and Recall**:
```python
def calculate_multiclass_metrics(y_true, y_pred, classes):
    """Calculate precision, recall, F1 for each personality type"""
    
    metrics = {}
    
    for class_label in classes:
        # Create binary labels for current class
        true_binary = (y_true == class_label).astype(int)
        pred_binary = (y_pred == class_label).astype(int)
        
        # Calculate metrics
        tp = np.sum((true_binary == 1) & (pred_binary == 1))
        fp = np.sum((true_binary == 0) & (pred_binary == 1))
        fn = np.sum((true_binary == 1) & (pred_binary == 0))
        tn = np.sum((true_binary == 0) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': np.sum(true_binary)
        }
    
    return metrics
```

**Confusion Matrix Analysis**:
```python
def analyze_confusion_matrix(y_true, y_pred, classes):
    """Detailed confusion matrix analysis"""
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    analysis = {
        'matrix': cm,
        'accuracy_per_class': {},
        'most_confused_pairs': [],
        'classification_difficulty': {}
    }
    
    # Calculate per-class accuracy
    for i, class_label in enumerate(classes):
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        accuracy = correct / total if total > 0 else 0
        analysis['accuracy_per_class'][class_label] = accuracy
    
    # Find most confused pairs
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / np.sum(cm[i, :])
                if confusion_rate > 0.1:  # More than 10% confusion
                    analysis['most_confused_pairs'].append({
                        'true_class': classes[i],
                        'predicted_class': classes[j],
                        'confusion_rate': confusion_rate
                    })
    
    return analysis
```

## 8. Optimization Algorithms

### 8.1 Hyperparameter Optimization

**Grid Search with Cross-Validation**:
```python
def optimize_hyperparameters(X, y):
    """Comprehensive hyperparameter optimization"""
    
    from sklearn.model_selection import GridSearchCV
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'lbfgs', 'newton-cg'],
        'max_iter': [1000, 2000, 5000],
        'class_weight': [None, 'balanced'],
        'multi_class': ['ovr', 'multinomial']
    }
    
    # Create base model
    base_model = LogisticRegression(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Return best model and parameters
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
```

**Bayesian Optimization**:
```python
def bayesian_hyperparameter_optimization(X, y):
    """Use Bayesian optimization for hyperparameter tuning"""
    
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    
    # Define search space
    search_space = [
        Real(0.01, 100.0, name='C', prior='log-uniform'),
        Categorical(['liblinear', 'lbfgs'], name='solver'),
        Integer(500, 3000, name='max_iter'),
        Categorical([None, 'balanced'], name='class_weight')
    ]
    
    def objective(params):
        """Objective function to minimize (negative accuracy)"""
        C, solver, max_iter, class_weight = params
        
        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42
        )
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)  # Minimize negative accuracy
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=50,
        random_state=42
    )
    
    return {
        'best_params': dict(zip(['C', 'solver', 'max_iter', 'class_weight'], result.x)),
        'best_score': -result.fun,
        'optimization_result': result
    }
```

### 8.2 Feature Selection Algorithms

**Mutual Information Feature Selection**:
```python
from sklearn.feature_selection import mutual_info_classif

def select_features_mutual_info(X, y, k=1000):
    """Select top k features based on mutual information"""
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Select top k features
    top_indices = np.argsort(mi_scores)[-k:]
    
    return top_indices, mi_scores[top_indices]

def select_features_chi2(X, y, k=1000):
    """Select features using chi-squared test"""
    
    from sklearn.feature_selection import chi2, SelectKBest
    
    # Chi-squared feature selection
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    return selected_indices, feature_scores[selected_indices]
```

**Recursive Feature Elimination**:
```python
def recursive_feature_elimination(X, y, n_features=500):
    """Use RFE to select most important features"""
    
    from sklearn.feature_selection import RFE
    
    # Base estimator
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    
    # RFE selector
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=50  # Remove 50 features at each step
    )
    
    # Fit RFE
    rfe.fit(X, y)
    
    return {
        'selected_features': rfe.get_support(indices=True),
        'feature_ranking': rfe.ranking_,
        'n_features_selected': rfe.n_features_
    }
```

## 9. Clustering and Pattern Recognition Algorithms

### 9.1 Personality Clustering Analysis

**K-Means Clustering for Personality Groups**:
```python
from sklearn.cluster import KMeans

def cluster_personality_responses(X, n_clusters=4):
    """Cluster responses to find personality patterns"""
    
    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(X)
    
    # Analyze clusters
    cluster_analysis = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        cluster_analysis[cluster_id] = {
            'size': np.sum(cluster_mask),
            'center': cluster_center,
            'inertia': np.sum((X[cluster_mask] - cluster_center) ** 2)
        }
    
    return cluster_labels, cluster_analysis
```

**Hierarchical Clustering for Personality Relationships**:
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def hierarchical_personality_analysis(X, personality_labels):
    """Analyze personality type relationships using hierarchical clustering"""
    
    # Calculate average feature vectors for each personality type
    personality_centers = {}
    unique_types = np.unique(personality_labels)
    
    for ptype in unique_types:
        mask = personality_labels == ptype
        center = np.mean(X[mask], axis=0)
        personality_centers[ptype] = center
    
    # Create distance matrix between personality types
    centers_matrix = np.array(list(personality_centers.values()))
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(centers_matrix, method='ward')
    
    # Create dendrogram
    dendrogram_data = dendrogram(
        linkage_matrix,
        labels=list(personality_centers.keys()),
        no_plot=True
    )
    
    return {
        'linkage_matrix': linkage_matrix,
        'dendrogram': dendrogram_data,
        'personality_centers': personality_centers
    }
```

## 10. Advanced Statistical Algorithms

### 10.1 Statistical Significance Testing

**Chi-Square Test for Feature Independence**:
```python
from scipy.stats import chi2_contingency

def test_feature_personality_independence(feature_matrix, personality_labels, feature_names):
    """Test independence between features and personality types"""
    
    independence_results = {}
    
    for i, feature_name in enumerate(feature_names):
        # Create contingency table
        feature_values = feature_matrix[:, i].toarray().flatten()
        
        # Binarize feature values (present/absent)
        binary_features = (feature_values > 0).astype(int)
        
        # Create contingency table
        contingency_table = pd.crosstab(binary_features, personality_labels)
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        independence_results[feature_name] = {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < 0.05
        }
    
    return independence_results
```

**ANOVA for Continuous Features**:
```python
from scipy.stats import f_oneway

def anova_feature_analysis(feature_matrix, personality_labels):
    """ANOVA analysis for feature importance across personality types"""
    
    anova_results = {}
    n_features = feature_matrix.shape[1]
    
    for feature_idx in range(n_features):
        feature_values = feature_matrix[:, feature_idx].toarray().flatten()
        
        # Group feature values by personality type
        groups = {}
        for i, ptype in enumerate(personality_labels):
            if ptype not in groups:
                groups[ptype] = []
            groups[ptype].append(feature_values[i])
        
        # Perform ANOVA
        group_values = list(groups.values())
        f_statistic, p_value = f_oneway(*group_values)
        
        anova_results[feature_idx] = {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
    return anova_results
```

### 10.2 Dimensionality Reduction Algorithms

**Principal Component Analysis (PCA)**:
```python
from sklearn.decomposition import PCA

def apply_pca_analysis(X, n_components=50):
    """Apply PCA for dimensionality reduction and analysis"""
    
    # Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    
    # Analyze components
    component_analysis = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'n_components': n_components
    }
    
    # Find optimal number of components (95% variance)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumvar >= 0.95) + 1
    
    component_analysis['optimal_components'] = optimal_components
    
    return X_pca, component_analysis
```

**t-SNE for Visualization**:
```python
from sklearn.manifold import TSNE

def tsne_personality_visualization(X, personality_labels):
    """Create t-SNE visualization of personality clusters"""
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        n_iter=1000
    )
    
    X_tsne = tsne.fit_transform(X.toarray())
    
    # Create visualization data
    visualization_data = {
        'coordinates': X_tsne,
        'labels': personality_labels,
        'unique_types': np.unique(personality_labels)
    }
    
    return visualization_data
```

This comprehensive documentation covers all the algorithms, mathematical foundations, and implementation details used in the personality detection project. Each algorithm is explained with both theoretical background and practical implementation code.