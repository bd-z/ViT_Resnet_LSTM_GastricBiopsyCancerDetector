from flask import Flask
from flask import render_template
from flask import Flask,  render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
basedir = r"C:\Users/zhang\rnn_tsf\vit\transfer learning\python scripts"
os.chdir(basedir)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='This is a VIT(Vision Transformer) based network to predict whether cancer exists')

@app.route('/upload2',methods=['GET'])
def upload_images():
    return render_template("upload.html")

@app.route('/getImg/',methods=['GET','POST'])
def getImg():	
    imgData = request.files["image"]    
    path = basedir +"/static/upload/"   
    imgName = imgData.filename    
    file_path = path + imgName    
    imgData.save(file_path)    
    url = file_path     
    print(f"Image URL: {url}")    
    print(f"imgName: {imgName}")         
    return render_template("upload_ok.html", url=url)

@app.route('/prediction/',methods=['GET','POST'])
def prediction():
    url=request.args.get('url')   
    path=[url]   
    result = get_prediction(path)  
    vit_result = get_prediction_vit(path)
    return render_template('prediction.html',
                            pred = str(result),
                            pred_vit = str(vit_result))

@app.route('/submit-comment', methods=['GET','POST'])
def submit_comment():
    user_comment = request.form['userComment']    
    print(user_comment)
    result = get_prediction_lstmNLP(user_comment)    
    if result > 0.6:
        feedback = "We are delighted to provide valuable predictions."
    elif result > 0.4 and result <= 0.6:
        feedback = "We promise to continue our efforts to improve our services."
    else:
        feedback ="It's regrettable to hear that."       
    return render_template('user_feedback.html',
                            feedback = feedback)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080 )
    app.run(debug=True, port=5000)