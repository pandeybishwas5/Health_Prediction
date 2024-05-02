import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
diab = pickle.load(open('diab.pkl', 'rb'))
heartmodel = pickle.load(open('heartmodel.pkl', 'rb'))
livermodel = pickle.load(open('livermodel.pkl', 'rb'))
breastmodel = pickle.load(open('breastmodel.pkl', 'rb'))

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

@app.route('/loginOrg')
def loginOrg():
    return render_template('loginOrg.html')

@app.route('/signupOrg')
def signupOrg():
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
    return render_template('loginOrg.html')



#DIABETES
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = diab.predict(final_features)
    output = int(round(prediction[0],2))
    if(output==0):
        return render_template('diabetesResult-.html', prediction_text='The individual does not show the probability of having diabetes ( {} ) with 80.0347 % accuracy.'.format(output))
    else:
        return render_template('diabetesResult+.html', prediction_text='The individual shows the probability of having diabetes ( {} ) with 80.0347 % accuracy.'.format(output))
    
    
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = diab.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


#HEART
@app.route('/predictheart',methods=['POST'])
def predictheart():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = heartmodel.predict(final_features)

    output = int(round(prediction[0],2))
    if(output==0):
        return render_template('heartResult-.html', prediction_text='The individual does not show the probability of having heart disease ( {} ) with 83.007 % accuracy..'.format(output))
    else:
        return render_template('heartResult+.html', prediction_text='The individual shows the probability of having heart disease ( {} ) with 83.007 % accuracy.'.format(output))

@app.route('/predict_apiheart',methods=['POST'])
def predict_apiheart():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = heartmodel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

#LIVER
@app.route('/predictliver',methods=['POST'])
def predictliver():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = livermodel.predict(final_features)

    output = int(round(prediction[0],2))
    if(output==0):
        return render_template('liverResult-.html', prediction_text='The individual does not show the probability of having liver disease ( {} ) with 71.026 % accuracy.'.format(output))
    else:
        return render_template('liverResult+.html', prediction_text='The individual shows the probability of having liver disease ( {} ) with 71.026 % accuracy.'.format(output))

@app.route('/predict_apiliver',methods=['POST'])
def predict_apiliver():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = livermodel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



#BREAST
@app.route('/predictbreast',methods=['POST'])
def predictbreast():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = breastmodel.predict(final_features)
#  benign = 0, malignant = 1
    output = int(round(prediction[0],2))
    if(output==0):
        return render_template('breastResult+.html', prediction_text='The individual has Benign Cancer ( {} ) with 91.01 % accuracy.'.format(output))
    else:
        return render_template('breastResult-.html', prediction_text='The individual has Malignant Cancer ( {} ) with 91.01 % accuracy.'.format(output))

@app.route('/predict_apibreast',methods=['POST'])
def predict_apibreast():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = breastmodel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
