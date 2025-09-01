from flask import Flask, request, render_template, session
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import uuid
import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

MAX_ROWS = 50
MAX_COLS = 5


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 0.5 * 1024 * 1024  # 0.5 MB
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'your_secret_key_here'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return render_template('index.html', error="File too large. Max size is 0.5MB.", success=False, summary=None, use_default=True)

@app.route('/reset')
def reset():
    filepath = session.pop('file_path', None)
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Could not delete uploaded file: {e}")
    return render_template('index.html', success=False, error=None, summary=None, use_default=True)



@app.route('/', methods=['GET', 'POST'])
def index():
    # Clear session file_path if file no longer exists
    filepath = session.get('file_path')
    if filepath and not os.path.exists(filepath):
        session.pop('file_path', None)
        filepath = None
    if request.method == 'POST':

# Check if a new file was uploaded
        file = request.files.get('file')

        if file and file.filename != '':
            # Ensure uploads folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Generate a unique filename to avoid collisions
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # Save the file persistently
            file.save(filepath)

            # Store the filepath in the session
            session['file_path'] = filepath

        else:
            # Use the previously uploaded file
            filepath = session.get('file_path')

            if not filepath or not os.path.exists(filepath):
                return render_template('index.html', error="No file selected or uploaded.", success=False, summary=None, use_default=True)
        # Read CSV
        try:
            df = pd.read_csv(filepath, delimiter = ";")
            
            if df.shape[0] > MAX_ROWS:
                return render_template('index.html', error=f"Too many rows (max {MAX_ROWS})",use_default=True)
            if df.shape[1] > MAX_COLS:
                return render_template('index.html', error=f"Too many columns (max {MAX_COLS})",use_default=True)
            
            df = df.replace(",", ".", regex = True).astype(float)
            
            if 'y' not in df.columns:
                return render_template('index.html', error="Missing 'y' column (dependent variable).",use_default=True)

            # ✅ Extract form settings
            use_default = 'use_default' in request.form
            constant = 'constant' in request.form
            standardize = 'standardize' in request.form
            log_y = 'log_y' in request.form
            run_vif = 'run_vif' in request.form
            residual_plot = 'residual_plot' in request.form


            # Separate predictors and target 
            X = df.drop(columns='y') 
            y = df['y'] 

            if not use_default:
                if log_y:
                    y = np.log(y)
                if standardize:
                    X = (X - X.mean()) / X.std()
                if constant:
                    X = sm.add_constant(X) 

            if use_default:
                X = sm.add_constant(X)

            # Fit model 
            model = sm.OLS(y, X).fit() 
            # Extract regression results 
            summary_text = model.summary().as_text() 
            # Render results to HTML 
            return render_template('index.html', success=True, summary=summary_text, use_default=use_default)
            
        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {e}", success=False, summary=None, use_default=True)
            

    return render_template('index.html', success=False, error=None, summary=None, use_default=True)

if __name__ == '__main__':

    app.run(debug=True)

