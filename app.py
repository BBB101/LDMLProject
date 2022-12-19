import base64
import io
from PIL import Image
from flask import *
import LDML

app = Flask(__name__)
app.secret_key = "fe64b47d64e7b3f1a287382122a04eb15c778483cfa18705"


@app.route('/')
def index():
    im = Image.open('./static/preview-img.jpg')
    data = io.BytesIO()
    im.save(data, "JPEG") # Jinja Variable Transfer
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html', img_data=encoded_img_data.decode('utf-8'), response_data="")


@app.route('/ConvertImage', methods=["POST", "GET"])
def ImageConverter():
    print("HALLO")
    DigOrLet = request.form.get("DigLet")
    print(DigOrLet)
    im = Image.open(request.files.get("images"))
    png = False
    if request.files.get("images") is None:
        im = Image.open('./static/preview-img.jpg')
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        return render_template('index.html', img_data=encoded_img_data.decode('utf-8'), response_data="No Image Entered")

    if not request.files.get("images").name.__contains__(".jpg") or not request.files.get("images").name.__contains__(".jpeg"):
        im = Image.open(request.files.get("images"))
        rgb_im = im.convert("RGB")
        rgb_im.save("Temp.jpg")
        png = True
    if DigOrLet is None:
        print("LETTERS")
        if not png:
            Cresponse = LDML.getPrediction(request.files.get("images"), 'my_Lmodel', False)
        else:
            Cresponse = LDML.getPrediction("Temp.jpg", 'my_Lmodel', False)
    else:
        print("DIGITS")
        if not png:
            Cresponse = LDML.getPrediction(request.files.get("images"), 'my_Dmodel', True)
        else:
            Cresponse = LDML.getPrediction("Temp.jpg", 'my_Dmodel', True)
    data = io.BytesIO()
    if not png:
        im.save(data, "JPEG") # Jinja Variable Transfer
    else:
        im = Image.open("Temp.jpg")
        im.save(data, "JPEG") # Jinja Variable Transfer
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html', img_data=encoded_img_data.decode('utf-8'), response_data=Cresponse)


if __name__ == '__main__':
    # key is 'secret'
    # Files Created Using winpty openssl req -x509 -new -nodes -key myCA.key -sha256 -days 1825 -out myCA.pem
    context = (r'cert.pem', r'key.pem') # Encrypts Using TLS/SSL/RSA
    app.run(debug=True, port=5000, threaded=True, ssl_context=context)

