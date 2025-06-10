import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import os

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for larger main header and navigation
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem !important;
        font-weight: bold !important;
        color: #1565C0 !important;
        margin-bottom: 1.5rem !important;
        padding-left: 1rem !important;
    }
    /* Style for horizontal radio buttons */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 0.5rem !important;
    }
    div[data-testid="stRadio"] > div > div {
        min-width: 60px !important;
        padding: 0.2rem 0.5rem !important;
        font-size: 0.8rem !important;
        background-color: #f0f2f6 !important;
        border-radius: 4px !important;
        margin: 0 !important;
        text-align: center !important;
    }
    div[data-testid="stRadio"] > div > div[data-checked="true"] {
        background-color: #1565C0 !important;
        color: white !important;
    }
    /* Hide the radio button circles */
    div[data-testid="stRadio"] > div > div > div {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state for navigation if not exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# ================== Page Functions ====================
# ================== Page 1: Home ====================
def main_home():
    # Main header
    st.markdown('<div class="main-header">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    # Page header
    st.markdown('<div class="header-container"><h1>Home</h1></div>', unsafe_allow_html=True)

    # Welcome section at the top
    st.markdown("""
    <div class='section-container' style='padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;'>
        <h3 style='color: #1565C0; margin-top: 0;'>Welcome to Sentiment Analysis Dashboard</h3>
        <p style='font-size: 1.1em;'>This website created by Nayla Anandhita, allows users to explore public sentiment about Prabowo's government through social media posts on platform X. Using classical machine learning models, it analyzes and classifies sentiments as positive, negative, or neutral, offering insights into public opinion trends.</p>
    </div>
    """, unsafe_allow_html=True)

    # Wordcloud section (description only)
    st.markdown('''
        <div class='section-container' style='padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;'>
            <h3 style='color: #1565C0; margin-top: 0;'>Word Cloud Analysis</h3>
            <p style='font-size: 1.1em; margin-bottom: 1rem;'>Below is the word cloud visualization of the most frequent words in tweets about Prabowo's Government, categorized by sentiment. This visualization helps us understand the key topics and themes discussed in each sentiment category, with different colors representing different sentiments (negative, neutral, and positive).</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Display wordcloud image with specific dimensions
    try:
        # Create a container for centering
        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:
            st.image(
                "assets/wordcloud.png",
                width=600,  
                use_container_width=False
            )
    except Exception as e:
        st.error(f"Error loading wordcloud image: {str(e)}")
        # Add more detailed debugging information
        image_path = os.path.join(os.getcwd(), "assets", "wordcloud.png")
        st.info(f"""
        Debugging information:
        - Current working directory: {os.getcwd()}
        - Full image path: {image_path}
        - File exists: {os.path.exists(image_path)}
        - File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'} bytes
        """)

    # --- Search Sentiment filter in sidebar ---
    st.sidebar.header('Search Sentiment')
    sentiment_options = ['All', 'Negative', 'Neutral', 'Positive']
    selected_sentiment = st.sidebar.selectbox('Filter by Sentiment', sentiment_options, index=0)

    # Load actual data from Excel file
    data_path = r'C:/Users/Nayla/OneDrive/Documents/THESIS/streamlit/sampled_sentiment_data_balanced.xlsx'
    try:
        actual_data = pd.read_excel(data_path)
        # Use 'full_text' and 'sentiment' columns
        if 'full_text' in actual_data.columns and 'sentiment' in actual_data.columns:
            display_data = actual_data[['full_text', 'sentiment']].copy()
            display_data.columns = ['Tweet', 'Sentiment']
            if selected_sentiment != 'All':
                filtered_df = display_data[display_data['Sentiment'].str.lower() == selected_sentiment.lower()].head(5)
            else:
                # Show 5 random samples for each sentiment, then combine and shuffle
                neg = display_data[display_data['Sentiment'].str.lower() == 'negative'].sample(n=5, random_state=1) if len(display_data[display_data['Sentiment'].str.lower() == 'negative']) >= 5 else display_data[display_data['Sentiment'].str.lower() == 'negative']
                neu = display_data[display_data['Sentiment'].str.lower() == 'neutral'].sample(n=5, random_state=2) if len(display_data[display_data['Sentiment'].str.lower() == 'neutral']) >= 5 else display_data[display_data['Sentiment'].str.lower() == 'neutral']
                pos = display_data[display_data['Sentiment'].str.lower() == 'positive'].sample(n=5, random_state=3) if len(display_data[display_data['Sentiment'].str.lower() == 'positive']) >= 5 else display_data[display_data['Sentiment'].str.lower() == 'positive']
                filtered_df = pd.concat([neg, neu, pos]).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            st.warning('Could not find "full_text" and "sentiment" columns in the data.')
            filtered_df = pd.DataFrame()
    except Exception as e:
        st.warning(f'Could not load actual data: {e}')
        filtered_df = pd.DataFrame({'Tweet': [], 'Sentiment': []})

    st.markdown("""
    <div class='section-container' style='padding: 20px; border-radius: 10px; margin-bottom: 1rem;'>
        <h4 style='color: #1565C0; margin-top: 0;'>Example Tweets for Selected Sentiment</h4>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    # Info cards row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class='section-container' style='padding: 20px; border-radius: 10px; display: flex; flex-direction: column; justify-content: center;'>
                <h4 style='color: #1565C0; margin-top: 0;'>Raw Data Distribution</h4>
                <p style='font-size: 1.1em;'>A total of 12,585 tweets were collected from social media X (formerly Twitter) using data crawling between October 20, 2024, and March 31, 2025. These tweets, related to the keyword 'Prabowo's Government,' serve as the primary data source for sentiment analysis.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='section-container' style='padding: 20px; border-radius: 10px; display: flex; flex-direction: column; justify-content: center;'>
                <h4 style='color: #1565C0; margin-top: 0;'>Balanced Data Distribution</h4>
                <p style='font-size: 1.1em;'>The tweet data was balanced using stratified undersampling, with the positive class as the reference. This ensured equal sample sizes across sentiment classes while preserving their internal distribution, allowing for fair comparison in the analysis.</p>
            </div>
        """, unsafe_allow_html=True)

    # Bar charts row 
    sentiment_data = {
        'Sentiment': ['negative', 'neutral', 'positive'],
        'Count': [6189, 4668, 1727]
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    fig = px.bar(sentiment_df, x='Sentiment', y='Count',
                 labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
                 title="Sentiment Distribution - Raw Data",
                 color='Sentiment',
                 color_discrete_map={
                     'positive': '#4CAF50',
                     'negative': '#F44336',
                     'neutral': '#FFC107'
                 },
                 height=400)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_color='#1565C0',
        font=dict(size=12)
    )
    balanced_data = {
        'Sentiment': ['negative', 'neutral', 'positive'],
        'Count': [1727, 1727, 1727]
    }
    balanced_df = pd.DataFrame(balanced_data)
    fig_balanced = px.bar(balanced_df, x='Sentiment', y='Count',
                         labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
                         title="Sentiment Distribution - Balanced Data",
                         color='Sentiment',
                         color_discrete_map={
                             'positive': '#4CAF50',
                             'negative': '#F44336',
                             'neutral': '#FFC107'
                         },
                         height=400)
    fig_balanced.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_color='#1565C0',
        font=dict(size=12)
    )
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.plotly_chart(fig_balanced, use_container_width=True)

# ================== Page 2: Method ====================
def main_method():
    # Main header
    st.markdown('<div class="main-header">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    # Page header
    st.markdown('<div class="header-container"><h1>Methodology</h1></div>', unsafe_allow_html=True)

    # Create two columns for the intro sections
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-container" style="min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <h3 style='color: #1565C0; margin-top: 0;'>Classical Machine Learning</h3>
            <p style='font-size: 1.1em;'>This approach uses traditional machine learning algorithms to train computer models in learning patterns from data and making predictions based on that data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-container" style="min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <h3 style='color: #1565C0; margin-top: 0;'>What is Supervised Learning?</h3>
            <p style='font-size: 1.1em;'>Supervised Learning is a method where the model learns from labeled data. Each training example consists of an input object (typically a vector) and a desired output value (also called the supervisory signal).</p>
        </div>
        """, unsafe_allow_html=True)

    # Algorithms section 
    st.markdown("""
    <div class='section-container' style='text-align: center;'>
        <h3 style='color: #1565C0; margin-top: 0; text-align: center;'>Machine Learning Algorithms Used</h3>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for algorithm cards
    col1, col2 = st.columns(2)

    with col1:
        # KNN Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>K-Nearest Neighbors (KNN)</h4>
        <p>Classifies data based on proximity to other data points.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Distance Formula (Euclidean):</p>
            <p style='margin: 5px 0; font-family: monospace;'>d(x,y) = √(Σ(xᵢ - yᵢ)²)</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # SVM Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Support Vector Machine (SVM)</h4>
        <p>Separates data into different classes using hyperplanes.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Decision Function:</p>
            <p style='margin: 5px 0; font-family: monospace;'>f(x) = sign(Σ αᵢyᵢK(xᵢ,x) + b)</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Logistic Regression Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Logistic Regression</h4>
        <p>Used for binary classification tasks.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Sigmoid Function:</p>
            <p style='margin: 5px 0; font-family: monospace;'>σ(x) = 1/(1 + e^(-x))</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Naive Bayes Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Naive Bayes</h4>
        <p>Based on Bayes' theorem, used for probability-based classification.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Bayes Theorem:</p>
            <p style='margin: 5px 0; font-family: monospace;'>P(y|x) = P(x|y)P(y)/P(x)</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Decision Tree Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Decision Tree</h4>
        <p>Creates a tree-like model of decisions and their possible consequences.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Information Gain:</p>
            <p style='margin: 5px 0; font-family: monospace;'>IG = H(parent) - Σ(pᵢ × H(childᵢ))</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Random Forest Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Random Forest</h4>
        <p>An ensemble algorithm that uses multiple decision trees.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Final Prediction:</p>
            <p style='margin: 5px 0; font-family: monospace;'>y = mode({T₁(x), T₂(x), ..., Tₙ(x)})</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Hard Voting Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Hard Voting Classifier</h4>
        <p>Majority voting among base classifiers.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Prediction:</p>
            <p style='margin: 5px 0; font-family: monospace;'>y = argmax(Σ wᵢ × I(hᵢ(x) = c))</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Soft Voting Card
        st.markdown("""
        <div class='section-container'>
        <h4 style='color: #1565C0; margin-top: 0;'>Soft Voting Classifier</h4>
        <p>Weighted average of probability estimates.</p>
        <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0; font-family: monospace;'>Prediction:</p>
            <p style='margin: 5px 0; font-family: monospace;'>y = argmax(Σ wᵢ × P(hᵢ(x) = c))</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ================== Page 3: Analysis ====================
def main_analysis():
    # Main header
    st.markdown('<div class="main-header">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    # Page header
    st.markdown('<div class="header-container"><h1>Analysis</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;'>
        <h3 style='color: #1565C0; margin-top: 0;'>Upload Your Vectorized Data</h3>
        <p style='font-size: 1.1em;'>Upload your Excel file containing TF-IDF vectorized data. The file should contain the TF-IDF features and a 'label_encoded' column with the sentiment labels (0: negative, 1: neutral, 2: positive).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download button for the template file
    template_file = r"C:\Users\Nayla\OneDrive\Documents\THESIS\Draft Thesis\excel hasil run\cleaned_data.xlsx"
    st.sidebar.download_button(
        label="Download Template Excel for Analysis",
        data=open(template_file, "rb").read(),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_excel(uploaded_file)
            st.success("File successfully uploaded!")
            
            # Show data preview
            st.markdown("""
                <div class="section-container">
                    <h3 style='color: #1565C0; margin-top: 0;'>Data Preview</h3>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head())

            # Prepare features and labels
            X = df.drop('label_encoded', axis=1)
            Y = df['label_encoded']
            
            # Train-test split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            st.success("Data preparation completed!")

            # Model training and evaluation
            st.markdown("""
                <div class="section-container">
                    <h3 style='color: #1565C0; margin-top: 0;'>Model Training and Evaluation</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Initialize models
            models = {
                'KNN': KNeighborsClassifier(),
                'SVM': SVC(probability=True),
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'Naive Bayes': MultinomialNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }

            # Add a tab for best model summary
            tab_names = list(models.keys()) + ['Ensemble Methods', 'Best Model']
            tabs = st.tabs(tab_names)
            
            # Store predictions and scores for finding best model
            predictions = {}
            model_scores = {}
            best_model_info = {}
            
            # Store training and testing accuracy for each model
            training_accuracies = {}
            testing_accuracies = {}
            
            # Train and evaluate each model
            for i, (model_name, model) in enumerate(models.items()):
                with tabs[i]:
                    st.markdown(f"### {model_name} Results")
                    
                    # Grid search for hyperparameter tuning
                    with st.spinner(f"Training {model_name}..."):
                        if model_name == 'KNN':
                            param_grid = {
                                'n_neighbors': list(range(1, 11)),
                                'weights': ['uniform', 'distance'],
                                'metric': ['minkowski', 'manhattan'],
                                'p': [1, 2]
                            }
                        elif model_name == 'SVM':
                            param_grid = {
                                'C': [0.1, 1, 10],
                                'kernel': ['linear', 'rbf', 'poly'],
                                'gamma': ['scale', 'auto']
                            }
                        elif model_name == 'Logistic Regression':
                            param_grid = {
                                'C': [0.01, 0.1, 1, 10],
                                'penalty': ['l2'],
                                'solver': ['lbfgs', 'liblinear'],
                                'max_iter': [1000]
                            }
                        elif model_name == 'Random Forest':
                            param_grid = {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5],
                                'min_samples_leaf': [1, 2]
                            }
                        elif model_name == 'Naive Bayes':
                            param_grid = {
                                'alpha': [0.1, 0.5, 1.0]
                            }
                        elif model_name == 'Decision Tree':
                            param_grid = {
                                'max_depth': [None, 10, 20, 30],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4]
                            }

                        grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
                        grid_search.fit(X_train, Y_train)
                        
                        best_model = grid_search.best_estimator_
                        y_train_pred = best_model.predict(X_train)
                        y_test_pred = best_model.predict(X_test)
                        predictions[model_name] = y_test_pred
                        
                        # Store model score
                        model_scores[model_name] = {
                            'accuracy': accuracy_score(Y_test, y_test_pred),
                            'params': grid_search.best_params_
                        }
                        best_model_info[model_name] = {
                            'train': classification_report(Y_train, y_train_pred, output_dict=True),
                            'test': classification_report(Y_test, y_test_pred, output_dict=True),
                            'params': grid_search.best_params_,
                            'accuracy': accuracy_score(Y_test, y_test_pred)
                        }

                        # Display results in table format
                        # (Remove Model Evaluation Results card)
                        # Create tables for results
                        train_results = pd.DataFrame(classification_report(Y_train, y_train_pred, output_dict=True)).T
                        test_results = pd.DataFrame(classification_report(Y_test, y_test_pred, output_dict=True)).T
                        colA, colB, colC = st.columns(3)
                        with colA:
                            st.markdown("#### Training Results")
                            st.dataframe(train_results)
                        with colB:
                            st.markdown("#### Testing Results")
                            st.dataframe(test_results)
                        with colC:
                            st.markdown("#### Best Hyperparameters")
                            best_params_df = pd.DataFrame([grid_search.best_params_]).T
                            best_params_df.columns = ['Value']
                            st.dataframe(best_params_df)
                        # Add interpretation below the tables
                        train_acc = accuracy_score(Y_train, y_train_pred)
                        test_acc = accuracy_score(Y_test, y_test_pred)
                        st.markdown(f"""
                        <div style='margin-top: 1rem; margin-bottom: 1rem; background: #f8f9fa; border-radius: 8px; padding: 12px;'>
                        <b>Interpretation:</b> <br>
                        The <b>{model_name}</b> model achieved a training accuracy of <b>{train_acc:.3f}</b> and a testing accuracy of <b>{test_acc:.3f}</b> with the selected hyperparameters: <b>{grid_search.best_params_}</b>. This provides insight into the model's ability to generalize and the impact of hyperparameter tuning on its performance.
                        </div>
                        """, unsafe_allow_html=True)
                        # Display confusion matrix as a colored table with explanation side by side
                        labels = [0, 1, 2]
                        cm = confusion_matrix(Y_test, y_test_pred, labels=labels)
                        cm_df = pd.DataFrame(cm, index=['Negative', 'Neutral', 'Positive'], columns=['Negative', 'Neutral', 'Positive'])
                        cm_styled = cm_df.style.background_gradient(cmap='Blues')
                        colD, colE = st.columns([2, 1])
                        with colD:
                            st.markdown("#### Confusion Matrix (Table)")
                            st.dataframe(cm_styled, use_container_width=True)
                        with colE:
                            st.markdown(f"<div style='margin-bottom: 1rem; color: #444;'>The confusion matrix beside shows the number of correct and incorrect predictions for each sentiment class made by the {model_name} model. The diagonal values represent correct predictions, while off-diagonal values indicate misclassifications.</div>", unsafe_allow_html=True)
                        # Store training and testing accuracy for each model
                        training_accuracies[model_name] = accuracy_score(Y_train, y_train_pred)
                        testing_accuracies[model_name] = accuracy_score(Y_test, y_test_pred)

            # Ensemble methods tab
            with tabs[-2]:
                st.markdown("### Ensemble Methods Results")
                st.markdown("""
                <div class='section-container' style='padding: 16px; border-radius: 8px; margin-bottom: 1rem; background: #E3F2FD;'>
                <b>In this section, we implement ensemble learning using two techniques: Hard Voting and Soft Voting. Both methods combine predictions from multiple classifiers to improve overall performance and stability.</b><br><br>
                For this ensemble, we selected <b>K-Nearest Neighbors (KNN)</b> and <b>Support Vector Machine (SVM)</b> as base estimators. These models were chosen based on their strong performance in both training and testing phases, demonstrating reliable accuracy and generalization capability.<br><br>
                <b>Hard Voting</b> takes the majority class label predicted by the individual models.<br>
                <b>Soft Voting</b> considers the predicted probabilities of each class and selects the class with the highest average probability.<br><br>
                By leveraging the complementary strengths of KNN and SVM, the ensemble aims to reduce variance and improve predictive robustness for sentiment classification.
                </div>
                """, unsafe_allow_html=True)
                # Hard Voting (no header)
                voting_hard = VotingClassifier(
                    estimators=[('knn', models['KNN']), ('svm', models['SVM'])],
                    voting='hard',
                    weights=[20, 30]
                )
                voting_hard.fit(X_train, Y_train)
                y_train_pred_hard = voting_hard.predict(X_train)
                y_test_pred_hard = voting_hard.predict(X_test)
                predictions['Hard Voting'] = y_test_pred_hard
                acc_table_hard = pd.DataFrame({
                    'Set': ['Training', 'Testing'],
                    'Accuracy': [accuracy_score(Y_train, y_train_pred_hard), accuracy_score(Y_test, y_test_pred_hard)]
                })
                # Soft Voting (no header)
                voting_soft = VotingClassifier(
                    estimators=[('knn', models['KNN']), ('svm', models['SVM'])],
                    voting='soft',
                    weights=[20, 30]
                )
                voting_soft.fit(X_train, Y_train)
                y_train_pred_soft = voting_soft.predict(X_train)
                y_test_pred_soft = voting_soft.predict(X_test)
                predictions['Soft Voting'] = y_test_pred_soft
                acc_table_soft = pd.DataFrame({
                    'Set': ['Training', 'Testing'],
                    'Accuracy': [accuracy_score(Y_train, y_train_pred_soft), accuracy_score(Y_test, y_test_pred_soft)]
                })
                # Show accuracy tables side by side
                acc_col1, acc_col2 = st.columns(2)
                with acc_col1:
                    st.dataframe(acc_table_hard, use_container_width=True)
                with acc_col2:
                    st.dataframe(acc_table_soft, use_container_width=True)
                # Show confusion matrices side by side
                cm_hard = confusion_matrix(Y_test, y_test_pred_hard, labels=[0, 1, 2])
                cm_hard_df = pd.DataFrame(cm_hard, index=['Negative', 'Neutral', 'Positive'], columns=['Negative', 'Neutral', 'Positive'])
                cm_hard_styled = cm_hard_df.style.background_gradient(cmap='Blues')
                cm_soft = confusion_matrix(Y_test, y_test_pred_soft, labels=[0, 1, 2])
                cm_soft_df = pd.DataFrame(cm_soft, index=['Negative', 'Neutral', 'Positive'], columns=['Negative', 'Neutral', 'Positive'])
                cm_soft_styled = cm_soft_df.style.background_gradient(cmap='Blues')
                cm_col1, cm_col2 = st.columns(2)
                with cm_col1:
                    st.dataframe(cm_hard_styled, use_container_width=True)
                with cm_col2:
                    st.dataframe(cm_soft_styled, use_container_width=True)
                # Combined explanation below both confusion matrices, full width
                st.markdown("""
                <div style='margin-top: 1rem; margin-bottom: 1rem; background: #f8f9fa; border-radius: 8px; padding: 18px 24px; text-align: left; width: 100%;'>
                <b>Interpretation:</b> <br>
                The confusion matrices above show the number of correct and incorrect predictions for each sentiment class made by the Hard Voting and Soft Voting ensembles. The diagonal values represent correct predictions, while off-diagonal values indicate misclassifications. By comparing both, you can assess the impact of ensemble strategy on classification performance.
                </div>
                """, unsafe_allow_html=True)

            # Best Model tab
            with tabs[-1]:
                st.markdown("### Best Model from All Methods")
                # Move grouped bar chart here (at the top)
                import plotly.graph_objects as go
                model_names = list(training_accuracies.keys())
                fig = go.Figure(data=[
                    go.Bar(name='Training Accuracy', x=model_names, y=[training_accuracies[m] for m in model_names]),
                    go.Bar(name='Testing Accuracy', x=model_names, y=[testing_accuracies[m] for m in model_names])
                ])
                fig.update_layout(
                    barmode='group',
                    title="Model Training and Testing Accuracy Comparison",
                    xaxis_title="Model",
                    yaxis_title="Accuracy",
                    title_font_color='#1565C0',
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
                # Find best model from all (including ensemble)
                all_testing_accuracies = {**testing_accuracies}
                best_model_name = max(all_testing_accuracies.items(), key=lambda x: x[1])[0]
                best_model_score = all_testing_accuracies[best_model_name]
                st.markdown(f"""
                    <div class="best-model-container">
                        <h3 style='color: #1565C0; margin-top: 0;'>Best Performing Model</h3>
                        <p style='font-size: 1.1em;'>The <b>{best_model_name}</b> model achieved the highest accuracy of <b>{best_model_score:.4f}</b> on the test set.</p>
                        <p style='font-size: 1.1em;'>Demonstrating its effectiveness in sentiment classification (positive, negative, neutral) concerning discourse around the Prabowo administration. This performance suggests that SVM is well-suited for handling polarized political text when supported by appropriate preprocessing and feature extraction techniques.</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")

# ================== Page 4: Tutorial ====================
def main_tutorial():
    # Main header
    st.markdown('<div class="main-header">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    # Page header
    st.markdown('<div class="header-container"><h1>Tutorial</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section-container' style='padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;'>
        <h3 style='color: #1565C0; margin-top: 0;'>How to Use This Application</h3>
        <p style='font-size: 1.1em;'>A quick guide to navigate and use the Sentiment Analysis Dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    # First row: Home and Method pages side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='section-container' style='padding: 20px; margin-top: 20px; height: 100%;'>
            <h4 style='color: #1565C0;'>1. Home Page</h4>
            <ul>
                <li>View dashboard overview and word cloud visualization</li>
                <li>Filter tweets by sentiment using the sidebar</li>
                <li>Explore example tweets and data distribution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='section-container' style='padding: 20px; margin-top: 20px; height: 100%;'>
            <h4 style='color: #1565C0;'>2. Method Page</h4>
            <ul>
                <li>Learn about classical machine learning approach</li>
                <li>Explore implemented algorithms and their formulas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Second row: Analysis and Important Notes side by side
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class='section-container' style='padding: 20px; margin-top: 20px; height: 100%;'>
            <h4 style='color: #1565C0;'>3. Analysis Page</h4>
            <ul>
                <li>Upload your vectorized data using the template</li>
                <li>View model training results and performance metrics</li>
                <li>Compare different models and ensemble methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='section-container' style='padding: 20px; margin-top: 20px; height: 100%; background-color: #E3F2FD;'>
            <h4 style='color: #1565C0;'>Important Notes</h4>
            <ul>
                <li>Use the provided template for data upload</li>
                <li>Ensure data is preprocessed and vectorized</li>
                <li>Allow time for analysis to complete</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ================== Main App ====================
def main():
    # Create navigation in sidebar
    with st.sidebar:
        st.markdown('<div style="font-size: 1.1rem; font-weight: bold; color: #1565C0; margin-bottom: 0.5rem;">Navigation</div>', unsafe_allow_html=True)
        
        # Add 'Select page' label above the radio buttons
        pages = ["Home", "Method", "Analysis", "Tutorial"]
        selected_page = st.radio(
            "Select page",
            pages,
            index=pages.index(st.session_state.current_page),
            horizontal=True
        )
        
        # Update session state if page changes
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
    
    # Display content based on current page
    if st.session_state.current_page == "Home":
        main_home()
    elif st.session_state.current_page == "Method":
        main_method()
    elif st.session_state.current_page == "Analysis":
        main_analysis()
    elif st.session_state.current_page == "Tutorial":
        main_tutorial()

if __name__ == "__main__":
    main()
