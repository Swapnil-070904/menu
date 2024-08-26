from gtts import gTTS # type: ignore
import io
import smtplib,subprocess,re
from twilio.rest import Client
from geopy.geocoders import Nominatim # type: ignore
from flask import Flask, request, render_template, redirect, url_for, flash,jsonify,send_file

app = Flask(__name__)
app.secret_key ='231d61aacdc033ea781601c07e4415dd'

# Email configuration
FROM_EMAIL = "ak6805002@gmail.com"
FROM_PASSWORD = "eamd vvqn bhqz dxpn"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
account_sid = 'AC7ace71fd257c6ea8db0b126215487662'
auth_token = '9871251fb007510d65ff52afe6027b66'
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

    else:
        return 'Unknown action!'

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
@app.route("/call", methods=['post'])
def call():
    no= request.form['to']
    account_sid = "AC7ace71fd257c6ea8db0b126215487662"
    auth_token = "9871251fb007510d65ff52afe6027b66"
    client = Client(account_sid, auth_token)
    call = client.calls.create(
            from_=number,

            to=no,

            url="http://demo.twilio.com/docs/voice.xml",

        )

    return(call.sid)
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
@app.route("/wth", methods=['post'])
def wth():
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body=request.form['body'],
    to='whatsapp:+91'+request.form['to']
)
    return (f"Message sent with SID: {message.sid}")

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

                                                                                                       
