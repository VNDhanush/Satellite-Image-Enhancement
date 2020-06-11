from flask import *  
import uuid
import os
from gandeblur import *
from denoise import *
from denoise import denoisee
from keras import backend as K
import logging
import shutil  

logger = logging.getLogger()
logger.disabled = False
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER=os.path.join('static','uploads') 
app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER



@app.route('/')  
def upload():  
    return render_template("index.html")
@app.route('/den')
def den():
    return render_template("denoise.html")
@app.route('/process', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['formfile']
        f_ext=f.filename.split('.')[1]
        if(f_ext in ALLOWED_EXTENSIONS):

            #print(type(f))
            unique_filename = str(uuid.uuid4())
            unique_file=unique_filename+'.'+f_ext
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],unique_file))
            K.clear_session()
            deblur2('generator.h5',os.path.join(app.config['UPLOAD_FOLDER'],unique_file),'',unique_file)
            before_pic=os.path.join(app.config['UPLOAD_FOLDER'],unique_file)
            after_pic=os.path.join(app.config['UPLOAD_FOLDER'],"generated_"+unique_file)
            return render_template("responseblur.html", name = f.filename, before = before_pic, after = after_pic)
        else:
            return """File format has to be one of the following: 
            <br/> 1 .png 
            <br/> 2 .jpg 
            <br/> 3 .jpeg 
            <br/> The file uploaded was in <strong>{}</strong> format""".format(f_ext)

@app.route('/denoise', methods = ['POST'])
def denois():

    if request.method == 'POST':
        f=request.files['formfile']
        f_ext=f.filename.split('.')[1]
        if(f_ext in ALLOWED_EXTENSIONS):
            #print(type(f))
            unique_filename = str(uuid.uuid4())
            unique_file=unique_filename+'.'+f_ext
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],unique_file))
            denoisee('denoise_model.h5',os.path.join(app.config['UPLOAD_FOLDER'],'',unique_file,))
            before_pic=os.path.join(app.config['UPLOAD_FOLDER'],unique_file)
            after_pic=os.path.join(app.config['UPLOAD_FOLDER'],"denoised_"+unique_file)
            return render_template("success.html", name = f.filename, before = before_pic, after = after_pic)

if __name__ == '__main__':  
    app.run(host ='0.0.0.0',debug = True)  