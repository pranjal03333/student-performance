STUDENT PASS/FAIL PREDICTION - STREAMLIT APP
=============================================

FILES INCLUDED
--------------
app.py                         Streamlit interface
student_pass_fail_model.pkl    Your supplied trained model (not retrained or replaced)
student_exam_performance.csv   Your supplied dataset
requirements.txt               Python package requirements
README.txt                     This instruction file
run_app.bat                    Windows shortcut to launch the app

HOW TO RUN ON WINDOWS
---------------------
1. Install Python 3.10 or 3.11 (recommended for compatibility).
2. Extract this ZIP file into a folder.
3. Open Command Prompt in the extracted folder.
4. (Recommended) create a virtual environment:

       py -m venv .venv
       .venv\Scripts\activate

5. Install dependencies:

       python -m pip install --upgrade pip
       pip install -r requirements.txt

6. Start Streamlit:

       streamlit run app.py

7. Open the local URL printed in the terminal, usually:
       http://localhost:8501

You can also double-click run_app.bat after completing steps 1-5.

IMPORTANT NOTES
---------------
- Keep app.py, the pickle, and the CSV in the same folder.
- The app uses the feature list from your original IPython notebook.
- The supplied pickle was saved from a scikit-learn version associated with 1.6.1.
  requirements.txt pins scikit-learn to 1.6.1 to reduce pickle compatibility issues.
- The app does not retrain, rewrite, or replace your model.
- Streamlit reruns when a widget changes, so the displayed prediction updates automatically.
- A small change in a feature does not guarantee that the predicted class will flip.
  Probabilities are displayed to three decimal places where supported by the model.
- Only load pickle/joblib files that you trust. Loading untrusted pickle files can execute code.

TROUBLESHOOTING
---------------
If 'streamlit' is not recognized:
       python -m streamlit run app.py

If you see a scikit-learn pickle compatibility error:
       pip install --force-reinstall scikit-learn==1.6.1

If a package is missing:
       pip install -r requirements.txt
