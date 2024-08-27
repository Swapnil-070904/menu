import boto3
import base64
from gtts import gTTS
import pandas as pd,numpy as np
import io,os,random,cv2
from PIL import Image, ImageDraw,ImageFilter
import smtplib,subprocess,re
from twilio.rest import Client
from geopy.geocoders import Nominatim 
from flask import Flask, request, render_template, redirect, url_for, flash,jsonify,send_file

app = Flask(__name__)
app.secret_key ='231d61aacdc033ea781601c07e4415dd'

# Email configuration
FROM_EMAIL =os.getenv('email')
FROM_PASSWORD = os.getenv('app_pwd')
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
account_sid = os.getenv('acc_sid')
auth_token = os.getenv('auth_token')
number = '+15739282798'
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_action', methods=['POST'])
def handle_action():
    action = request.form.get('action')

    if action == 'email':
        return render_template('form.html')
    elif action == 'call':
        return render_template('call.html')
    elif action == 'Sms':
        return render_template('Sms.html')
    elif action == 'docker':
        return render_template('docker.html')
    elif action == 'wth':
        return render_template('wth.html')
    elif action == 'geocord':
        return render_template('geocord.html')
    elif action == 'tta':
        return render_template('texttoaudio.html')
    elif action == 'data':
        return render_template('dataprocess.html')
    elif action == 'random_image':
        return render_template('random_image.html')
    elif action == 'image_filter':
        return render_template('image_filter.html')
    elif action == 'live':
        return render_template('livestream.html')
    else:
        return 'Unknown action!'
# --------------------------------------------------------------------------------------------------------------------
@app.route('/send_email', methods=['POST'])
def send_email():
    to_email = request.form['to']
    subject = request.form['subject']
    body = request.form['body']

    email_text = f"""\
From: {FROM_EMAIL}
To: {to_email}
Subject: {subject}

{body}
"""

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(FROM_EMAIL, FROM_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, email_text)
        server.quit()
        return('Email sent successfully!')
    except Exception as e:
        flash(f'Failed to send email: {e}')
    
    return render_template('form.html')
# ------------------------------------------------------------------------------------------------------------------------
@app.route("/call", methods=['post'])
def call():
    no= '+91'+ request.form['to']
    account_sid = os.getenv('acc_sid')
    auth_token = os.getenv('auth_token')
    client = Client(account_sid, auth_token)
    call = client.calls.create(
            from_=number,

            to=no,

            url="http://demo.twilio.com/docs/voice.xml",

        )

    return(call.sid)
# ------------------------------------------------------------------------------------------------------------------------------
@app.route("/Sms", methods=['post'])
def Sms():
    msg=request.form['msg']
    no=request.form['to']
    client = Client(account_sid, auth_token)
    message = client.messages.create(
    body=msg,

	    from_=number,

	    to=no

	)
    return(f"Message sent with SID: {message.sid}")
# ------------------------------------------------------------------------------------------------------------------------------
@app.route("/wth", methods=['post'])
def wth():
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body=request.form['body'],
    to='whatsapp:+91'+request.form['to']
)
    return (f"Message sent with SID: {message.sid}")
# -----------------------------------------------------------------------------------------------------------------------------------
@app.route('/get_geo_coordinates', methods=['POST'])
def get_geo_coordinates():
    address = request.form.get('address')

    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(address)

    if location:
        coordinates = {
            'latitude': location.latitude,
            'longitude': location.longitude,
            'address': location.address
        }
    else:
        coordinates = {'error': 'Location not found'}

    return jsonify(coordinates)
# --------------------------------------------------------------------------------------------------------------------------------
@app.route('/stringtoaudio', methods=['POST'])
def string_to_audio():
    data = request.json
    text = data.get('text', '')
    if text:
        audio_buffer = io.BytesIO()
        tts = gTTS(text, lang='en')
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return send_file(
            audio_buffer,
            as_attachment=True,
            mimetype='audio/mpeg',
            download_name='output.mp3'
        )
    else:
        return jsonify({'error': 'No text provided'}), 400
# ----------------------------------------------------------------------------------------------------------------------------
@app.route('/process_data', methods=['POST'])
def process_data():
    file = request.files['file']
    if not file:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})

    try:
        # Read the file into a DataFrame
        df = pd.read_csv(file)
        
        # Perform data processing
        summary = {
            'mean': df.mean().to_dict(),
            'median': df.median().to_dict(),
            'std_dev': df.std().to_dict()
        }
        
        return jsonify({'status': 'success', 'summary': summary})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
# --------------------------------------------------------------------------------------------------------------------------------
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        width = int(request.form['width'])
        height = int(request.form['height'])

        # Create an image with random colors
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        for x in range(width):
            for y in range(height):
                draw.point((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return f"Error generating image: {str(e)}", 400@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        width = int(request.form['width'])
        height = int(request.form['height'])

        # Create an image with random colors
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        for x in range(width):
            for y in range(height):
                draw.point((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return f"Error generating image: {str(e)}", 400
# ------------------------------------------------------------------------------------------------------------------------
def apply_filter(image, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'edge':
        return cv2.Canny(image, 100, 200)
    else:
        return image

@app.route('/filter', methods=['POST'])
def apply_filter_route():
    file = request.files['file']
    filter_type = request.form.get('filter')

    if not file:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})

    try:
        # Read the image file
        img = Image.open(file.stream)
        img = np.array(img)

        # Apply the chosen filter
        filtered_img = apply_filter(img, filter_type)

        # Convert back to PIL Image
        filtered_pil = Image.fromarray(filtered_img)
        buffer = io.BytesIO()
        filtered_pil.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        return jsonify({
            'status': 'success',
            'filtered_image': 'data:image/png;base64,' + base64.b64encode(img_data).decode('utf-8')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    # ---------------------------------------------------------------------------------------------------------------------------
# Create an AWS Rekognition client
rekognition = boto3.client('rekognition',region_name='ap-south-1')

@app.route('/liveStream', methods=['POST'])
def liveStream():
    # Receive the live stream from the frontend
    stream = request.get_json()

    # Process the live stream using OpenCV
    cap = cv2.VideoCapture(stream)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the live stream using AWS Rekognition
        response = rekognition.detect_labels(Image={'Bytes': frame.tobytes()})
        labels = response['Labels']

        # Return the detected labels to the frontend
        return jsonify({'labels': [label['Name'] for label in labels]})

# ------------------------------------------------------DOCKER----------------------------------------------
@app.route("/pull", methods=['post'])
def pull():
    img=request.form['docker']
    cmd=f'docker pull {img}'
    status,output = subprocess.getstatusoutput(cmd)
    if status == 0:
        image_name = output.split('/')[-1]
        return image_name
    else:
        return("image downloded failed")

@app.route("/images", methods=['post'])
def get_images():
    status,output = subprocess.getstatusoutput('docker images')
    if status == 0:
        img = re.sub(r'(SIZE|MB|kB)', r'\1\n', output)
        return img
    else:
        return("image downloded failed")

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')

                                                                                                       
