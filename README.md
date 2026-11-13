Student Performance Prediction Using Machine Learning

THis is our final project for CSE 4309 - Fundamental Of Machine Learning

For this project, we wanted to see if we could prodict whether a student might pass or fail based on the information of previous grades, study time, absence and so on.
We used a student performance dataset from the UCI Machine Learning Rep and compared 3 ML algorigth to see how well all these methods performed.

Dataset:
We used the student performance dataset from UCI Machine LEarning Repo.

    Dataset link: https://archive.ics.uci.edu/dataset/320/student+performance

    We worked with the Portuguese course dataset "student-por.csv" which contain information about 649 students.

    The dataset Contained:
    1. Student age and family background
    2. Study time
    3. Absensce
    4. Previous grades
    5. First ad second period grades (G1 and G2)
    6. Final Grade (G3)

    The constraints we used:
    G3 .>= 10 means PASS (1)
    G3 < 10 means FAIL (0)

What we did:
We started by exploring the dataset to understand the student information and how the grades were distributed.

    After that, we prepared the data for training. Since the dataset contains both numerical and categorical information, we used scaling and one-hot encoding.

    We divided the dataset into 80% training data and 20% testing data.

    We trained and compared three machine learning algorithms:

    1. Logistic Regression
    2. Decision Tree
    3. Random Forest

Model Evaluation:
We used five-fold cross-validation on the training data to compare the models.

    Since most students in the dataset passed, we did not want to depend only on accuracy. We also looked at precision, recall, F1-score, and balanced accuracy, especially for students who failed.

    Random Forest with previous grades had the highest cross-validation Fail F1-score, so we selected it as our final model.

Project Files: 1. eda.py - Explores the dataset and generates graphs. 2. train.py - Trains and compares the machine learning models. 3. feature_importance.py - Checks which features were important for the 4. selected model. 5. app.py - Streamlit application for making predictions. 6. data/student-por.csv - Dataset used in the project. 7. models/best_model.joblib - Saved trained model. 8. outputs/ - Contains graphs and model evaluation results. 9. requirements.txt - Python libraries needed to run the project.

Run this Project: <All the python code is from my linux machine>

Step 1: Create a virtual Environment <python3 -m venv .venv>
Step 2: Activate on your system <source .venv/bin/activate>
Step 3: Install all the required libraries from the requirements.txt
<python -, pip install -r requrements.txt>
Step 5: Run the data analysis <python eda.py>
Step 6: Train and evaluate the model. <python train.py>
Step 7: Generate he feature imp results <python feature_importance.py>
Step 8: Open the streamlit application <python -m streamlit run app.py>

Streamlit Application:
We created a simple application where users can enter student information and be able to predict the student will pass or fail. It also disoplays the estimated probabilities for both outcomes.

    The application uses the Random Forests model saved during Training.

What we Learned:
The main thing that we noticed is that previous makes the huge difference in the prediction results. Model using G1 and G2 genreally performed better than models without them.

    We also learned that accuracy alone is not always the best way to compare models.

    Limitations:
        We used only one datset. So the mdoels may not perform the same way for the students from other schools.
