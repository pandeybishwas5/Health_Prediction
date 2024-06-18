import numpy as np
from flask import Flask, flash, request, jsonify, render_template, session, redirect, url_for
import pickle
from flask_sqlalchemy import SQLAlchemy
import secrets
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate

app = Flask(__name__)

# Generate and set the secret key
app.secret_key = secrets.token_hex(16)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:biswas123@localhost:5432/healthPrediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load models
model = pickle.load(open('models/rfc_model_diabetes.pkl', 'rb'))
diab = pickle.load(open('models/diabetes.pkl', 'rb'))
heartmodel = pickle.load(open('models/heartmodel.pkl', 'rb'))
livermodel = pickle.load(open('models/livermodel.pkl', 'rb'))
breastmodel = pickle.load(open('models/breastmodel.pkl', 'rb'))


bcrypt = Bcrypt(app)
# Define User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)


# Define routes
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predictionSystem')
def predictionSystem():
    return render_template('predictionSystem.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/breast')
def breast():
    return render_template('breast.html')

@app.route('/departments')
def departments():
    return render_template('departments.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/diabetesAnalytics')
def diabetesAnalytics():
    return render_template('diabetesAnalytics.html')

@app.route('/heartAnalytics')
def heartAnalytics():
    return render_template('heartAnalytics.html')

@app.route('/liverAnalytics')
def liverAnalytics():
    return render_template('liverAnalytics.html')

@app.route('/breastAnalytics')
def breastAnalytics():
    return render_template('breastAnalytics.html')

@app.route('/loginOrg', methods=['GET', 'POST'])
def loginOrg():
    if request.method == 'POST':
        username = request.form['orgname']
        password = request.form['password']
        
        # Query user by username
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session['orgname'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('predictionSystem'))  # Redirect to dashboard or any other page after login
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('loginOrg'))  # Redirect back to the sign-in page if login fails
    else:
        return render_template('loginOrg.html')

@app.route('/signupOrg', methods=['GET', 'POST'])
def signupOrg():
    if request.method == 'POST':
        username = request.form['orgname']
        password = request.form['password']
        email = request.form['email']
        
        
        # Check if the user already exists
        user = User.query.filter_by(username=username).first()
        existing_user = User.query.filter_by(email=email).first()
        if user:
            flash('Username already exists', 'error')
            return redirect(url_for('signupOrg'))
        
        if existing_user:
            flash('Email address already in use. Please choose a different email.', 'error')
            return redirect(url_for('signupOrg'))

        new_user = User(username=username, password=password, email=email)
        db.session.add(new_user)
        db.session.commit()
        flash('Sign up successful!', 'success')
        return redirect(url_for('loginOrg'))
    else:
        return render_template('signupOrg.html')

@app.route('/')
def BeforeLoginHome():
    return render_template('BeforeLoginHome.html')

@app.route('/BeforeLoginContact')
def BeforeLoginContact():
    return render_template('BeforeLoginContact.html')

@app.route('/BeforeLoginAbout')
def BeforeLoginAbout():
    return render_template('BeforeLoginAbout.html')

@app.route('/Logout')
def Logout():
    session.pop('loggedin', None)
    session.pop('orgid', None)
    session.pop('orgname', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('loginOrg'))

# DIABETES
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = diab.predict(final_features)
    output = int(round(prediction[0], 2))
    if output == 0:
        return render_template('diabetesResult-.html', prediction_text='The individual does not show the probability of having diabetes ( {} ) with 80.0347 % accuracy.'.format(output))
    else:
        return render_template('diabetesResult+.html', prediction_text='The individual shows the probability of having diabetes ( {} ) with 80.0347 % accuracy.'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = diab.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

# HEART
@app.route('/predictheart', methods=['POST'])
def predictheart():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = heartmodel.predict(final_features)
    output = int(round(prediction[0], 2))
    if output == 0:
        return render_template('heartResult-.html', prediction_text='The individual does not show the probability of having heart disease ( {} ) with 83.007 % accuracy.'.format(output))
    else:
        return render_template('heartResult+.html', prediction_text='The individual shows the probability of having heart disease ( {} ) with 83.007 % accuracy.'.format(output))

@app.route('/predict_apiheart', methods=['POST'])
def predict_apiheart():
    data = request.get_json(force=True)
    prediction = heartmodel.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

# LIVER
@app.route('/predictliver', methods=['POST'])
def predictliver():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = livermodel.predict(final_features)
    output = int(round(prediction[0], 2))
    if output == 0:
        return render_template('liverResult-.html', prediction_text='The individual does not show the probability of having liver disease ( {} ) with 71.026 % accuracy.'.format(output))
    else:
        return render_template('liverResult+.html', prediction_text='The individual shows the probability of having liver disease ( {} ) with 71.026 % accuracy.'.format(output))

@app.route('/predict_apiliver', methods=['POST'])
def predict_apiliver():
    data = request.get_json(force=True)
    prediction = livermodel.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

# BREAST
@app.route('/predictbreast', methods=['POST'])
def predictbreast():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = breastmodel.predict(final_features)
    output = int(round(prediction[0], 2))
    if output == 0:
        return render_template('breastResult+.html', prediction_text='The individual has Benign Cancer ( {} ) with 91.01 % accuracy.'.format(output))
    else:
        return render_template('breastResult-.html', prediction_text='The individual has Malignant Cancer ( {} ) with 91.01 % accuracy.'.format(output))

@app.route('/predict_apibreast', methods=['POST'])
def predict_apibreast():
    data = request.get_json(force=True)
    prediction = breastmodel.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
