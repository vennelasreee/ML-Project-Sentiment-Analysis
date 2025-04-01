# ML-Project-Sentiment-Analysis
Sentiment Analysis Model
This project is a Sentiment Analysis system that predicts the sentiment (positive or negative) of given text input using Machine Learning. It utilizes scikit-learn for model training and evaluation, with the best model stored in a .pkl file for easy deployment.

 Project Structure

 Sentiment-Analysis
 ┣  Datasets              # Folder containing datasets  
 ┣  best_model.pkl        # Trained sentiment analysis model  
 ┣  label_encoder.pkl     # Encoded labels for sentiment classes  
 ┣  vectorizer.pkl        # TF-IDF or CountVectorizer for text processing  
 ┣  train.py              # Script to train the model  
 ┣  predict.py            # Script to predict sentiment from user input  
 ┣  requirements.txt      # Required Python libraries  
 ┣  .gitignore            # Files to ignore in GitHub repo  
 ┗  README.md             # Documentation  

Features
1.Trains a sentiment analysis model using machine learning algorithms.
2.Processes text data using vectorization techniques (TF-IDF or CountVectorizer).
3.Saves and loads the best model for predictions using .pkl files.
4.Easy-to-use prediction script (predict.py) to analyze new text data.

Installation & Setup
1. Clone the repository:
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis

2. Install dependencies:
pip install -r requirements.txt

3. Train the model:
python train.py

4. Make predictions:
python predict.py

How It Works
1. Training (train.py):

Loads the dataset.
Preprocesses text using TF-IDF/CountVectorizer.
Trains a machine learning model (e.g., Naïve Bayes, SVM, etc.).
Saves the best model (best_model.pkl).

2. Prediction (predict.py):

Loads the trained model and vectorizer.
Takes user input (text).
Predicts and displays sentiment (Positive/Negative).

Technologies Used
1. Python 
2. Scikit-learn 
3. Pandas 
4. Numpy 
5. Joblib/Pickle 

License
This project is open-source and available under the MIT License.
