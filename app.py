import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,send_file   #flask ; host my model , render_template : redirect to home page for inout to get ouput
import pickle
from werkzeug.utils import secure_filename
import os

#data = pd.read_csv("E:\\Project\\incident\\uploads\\test.csv")

app = Flask(__name__)  #initailization
#app.run("localhost", "9999", debug=True)
model = pickle.load(open('model.pkl', 'rb'))  #read mode of model

UPLOAD_FOLDER = './uploads'
DOWNLOAD_FOLDER = './results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')      #indirect to home file

@app.route('/success', methods = ['POST'])  
def success():      
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(os.path.join("uploads",r'temp.csv'))
        #test.to_csv(os.path.join(temp_path,r'temp.csv'))
        return render_template("index.html", name = f.filename)  
        
@app.route('/predict',methods=['POST'])      #look at from tag of index page , click button to run this api code
def predict():
    '''
    For rendering results on HTML GUI
    '''     
    data = pd.read_csv(app.config['UPLOAD_FOLDER'] + "/" + 'temp.csv')
    data.drop(['S.No','created_at','updated_at','problem_ID','change_request'], axis=1 , inplace =True)
    
    #zero imputaion
    for col in data.columns:
        data.loc[data[col] == '?', col] = '0'
    for col in data.columns:
        data.loc[data[col] == '-100', col] = '0'
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    data['ID'] = le.fit_transform(data['ID'])
    data['ID_status'] = le.fit_transform(data['ID_status'])
    data['ID_caller'] = le.fit_transform(data['ID_caller'])
    data['opened_by'] = le.fit_transform(data['opened_by'])
    data['opened_time'] = le.fit_transform(data['opened_time'])
    data['Created_by'] = le.fit_transform(data['Created_by'])
    data['updated_by'] = le.fit_transform(data['updated_by'])
    data['type_contact'] = le.fit_transform(data['type_contact'])
    data['location'] = le.fit_transform(data['location'])
    data['category_ID'] = le.fit_transform(data['category_ID'])
    data['user_symptom'] = le.fit_transform(data['user_symptom'])
    data['Support_group'] = le.fit_transform(data['Support_group'])
    data['active'] = le.fit_transform(data['active'])
    data['Doc_knowledge'] = le.fit_transform(data['Doc_knowledge'])
    data['confirmation_check'] = le.fit_transform(data['confirmation_check'])
    data['support_incharge'] = le.fit_transform(data['support_incharge'])
    data['notify'] = le.fit_transform(data['notify'])

    model = pickle.load(open('model.pkl', 'rb'))
    pred = model.predict(data)
    pred = pd.DataFrame(pred)
    df=pd.concat([data.ID,pred],axis=1)
    df.columns
    df = df.rename(columns={0: "pred"})
    pred1 = {0.0:'2 - Medium',
            1.0:'3 - Low',
            2.0:'1 - High',
    }
    df = df.replace({"pred":pred1}) 
    #df['pred'] = df.pred1.map(pred1)
    #final_pred_file = pd.DataFrame(pred)
    #df=pd.concat([data.ID,final_pred_file],axis=1)
    df.columns=['Id','prediction1']
    df.to_csv(os.path.join(app.config['DOWNLOAD_FOLDER'], 'result.csv'), index=False)
    return render_template('index.html',  tables=[df.to_html(classes='data', header="true")])    
    
    
@app.route('/download', methods=['GET'])
def download_file():
    return send_file(app.config['DOWNLOAD_FOLDER']+'/'+'result.csv',as_attachment=True, attachment_filename='result.csv',mimetype='application/x-csv')   

if __name__ == "__main__":
    app.run(debug=True , port=8888)