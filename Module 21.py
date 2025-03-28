
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import gradio as gr

# Define the sms_classification function
def sms_classification(df):
    # Set the features and target variable
    X = df['text_message']
    y = df['label']

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Build and fit the model pipeline
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC()),
    ])
    text_clf.fit(X_train, y_train)

    # Return the trained model
    return text_clf

# Load the dataset into a DataFrame
sms_text_df = pd.read_csv('SMSSpamCollection.csv', names=['label', 'text_message'], sep='\t')

# Call the sms_classification function and store the result in text_clf
text_clf = sms_classification(sms_text_df)

# Define the sms_prediction function
def sms_prediction(text, model):
    # Make a prediction
    prediction = model.predict([text])[0]
    
    # Check if the message is ham or spam
    if prediction == 'ham':
        return f'The text message: "{text}", is not spam.'
    else:
        return f'The text message: "{text}", is spam.'

# Define the Gradio interface
interface = gr.Interface(
    fn=lambda text: sms_prediction(text, text_clf),
    inputs=gr.Textbox(label="Enter your text message here:"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="SMS Spam Detector"
)

# Launch the application
interface.launch(share=True)
