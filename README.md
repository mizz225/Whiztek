This is the ML NLP assignment done as a part of hiring process.

## Objective
To build a minimal sentiment classifier for cryptocurrency-related Reddit comments, accessible via a basic REST API

The coding challenge has been explained thoroughly in the file "coding_challenge_doc_v1.docx"

The crypto reddit comments data which has been used in this assignment is located by the name "crypto_currency_sentiment_dataset.csv"


## Code Explanations :

--- > modelTrain_xgBoost.ipynb :

The notebook is designed to build an XGBoost-based sentiment classifier for cryptocurrency-related Reddit comments, ensuring high accuracy and fair evaluation. It follows a structured workflow, including data preprocessing, feature extraction using SBERT (Sentence-BERT), dimensionality reduction via PCA, and model training with XGBoost, optimized using GridSearchCV.

The dataset consists of Reddit comments labeled as Positive or Negative. Since raw text data is noisy, preprocessing is performed to remove URLs, usernames, and excessive whitespace, ensuring consistency. Sentiment labels are then mapped to numeric values (1 for Positive, 0 for Negative) for machine learning compatibility.

For feature extraction, SBERT (MiniLM-l6-v2) is used instead of traditional methods like CountVectorizer, TF-IDF or Word2Vec. This is because SBERT provides better contextual understanding of the text, preserving sentence-level meaning rather than relying on isolated word frequencies or static embeddings. The MiniLM-l6-v2 variant is chosen specifically because it is lightweight yet powerful, striking a balance between efficiency and accuracy.

Since SBERT embeddings are high-dimensional, Principal Component Analysis (PCA) is applied to reduce the feature size. Through trial and error, 32 dimensions were found to be optimal—any lower led to performance degradation, while any higher increased computation time, especially during XGBoost GridSearch tuning. Thus, 32 dimensions were chosen as the sweet spot.

The dataset is split into training and testing sets using stratified sampling to maintain class distribution. This ensures that the model learns equally from both positive and negative samples, preventing bias. The classifier is trained using XGBoost, which was found to outperform other models like Logistic Regression, Random Forest, and Naïve Bayes in multiple trials. XGBoost achieved an initial accuracy of >90%, meeting the target requirement.

To further enhance performance, GridSearchCV was used to fine-tune XGBoost hyperparameters, pushing the accuracy to 92.9%. RandomizedSearchCV was also tested but failed to yield optimal parameters, as expected, making GridSearchCV the preferred choice.

For model evaluation, precision, recall, and F1-score were used with the ‘weighted’ parameter to ensure a balanced assessment across classes. Using ‘binary’ instead of ‘weighted’ slightly improved the score (~93.2%), but it favored class ‘1’ (Positive) over ‘0’ (Negative), leading to an unfair evaluation. Therefore, the ‘weighted’ metric was chosen for a fair class representation.

Additionally, Stratified K-Fold Cross-Validation (StratifiedKFold-CV) was used to evaluate model robustness. This ensures that every subset of data has a proportionate representation of each class, preventing the model from overfitting to a skewed distribution. The overall methodology ensures a well-balanced, high-performing sentiment classifier optimized for accuracy, fairness, and efficiency.



--- > modelEval_xgBoost.ipynb :

The code file has the steps to load the saved model files (in train code), to ensure the evaluation metric values like Precision, Recall, F1-score are as expected i.e, >90.



--- > modelPredict_xgBoost.ipynb :

The code file simply loads the saved model files, makes sentiment prediction on any user text



--- > modelTrain_logisticRegression.ipynb :

This code file has similar workflow as xgBoost-ModelTrain code, but has been adjusted to train a Logistic Regression model. Adjusted PCA dims to 64, to be able to increase the model accuracy to 91.2 from 89.4

Though this method is not as robust as XgBoost, I have just provided the code for reference.



--- > modelEval_logisticRegression.ipynb && modelPredict_logisticRegression.ipynb :

The codes work similar to xgBoost counterparts



--- > finbert_Eval.ipynb :

Here I have used FinBert model, a BERT model pre-trained on large corpus of financial text, which predicts the text as 1 of 3 classes (Positive, Negative, Neutral). Adjusted the code to yield binary prediction as per our dataset, but need further work on this.

This code is just for reference



--- > experiments.ipynb :

The code has all the other methods which has been tried out like TF-IDF, Word2Vec, RandomizedSearchCV, Random forest, Naive-Bayes etc.,



--- > fastapi_xgBoost.py && fastapi_logisticRegression.py :

Both the codes are built with index html files, to enable the user to provide the comment on UI and get the sentiment response on UI itself.

Steps to run :
1) python fastapi_xgBoost.py
2) Launch "http://127.0.0.1:8001"
3) Provide comment on text box
4) Predicted sentiment is displayed

Alternatively, if using windows computer, pass the below command to yield Predicted sentiment through simple REST API call :
--  Invoke-WebRequest -Uri "http://127.0.0.1:8001/predict" -Method POST -Headers @{"Content-Type"="application/x-www-form-urlencoded"} -Body "comment=LUNA is expected to drop"

Or if using Linux/Mac , use below command :
-- curl -X POST 'http://127.0.0.1:8001/predict' -H 'Content-Type: application/x-www-form-urlencoded' -d 'comment=LUNA is expected to drop'


## Important Point :

1) To be able to run any file of this repo in your system, ensure to create a python virtual environment using the requirements file "categ_env_TF_cpu_reqs.txt" with python 3.9.8.
2) Saved model files are present in "saved_files" folder, Feel free to load them and check Evaluation (or) Prediction using respective files as mentioned above in a code editor like VSCode / Jupyter
