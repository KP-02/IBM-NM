import cv2
import pickle
import cvzone
import numpy as np
from flask import Flask, render_template, Response,request, redirect,session
import ibm_db
import re

app = Flask(__name__)
app.secret_key = 'secret key'
# Database connection parameters
db_params = {
    "database": "bludb",
    "hostname": "21fecfd8-47b7-4937-840d-d791d0218660.bs2io90l08kqb1od8lcg.databases.appdomain.cloud",
    "port": 31864,
    "protocol": "TCPIP",
    "uid": "pdx64708",
    "pwd": "ovv5YbWRqJCNdWTu;",
    "ssl": "SSL",
    "security": "SSL",
    "sslservercertificate": "DigiCertGlobalRootCA.crt"
}

conn = None
print("Connecting to the database...")

try:
    conn = ibm_db.connect(
        f"DATABASE={db_params['database']};"
        f"HOSTNAME={db_params['hostname']};"
        f"PORT={db_params['port']};"
        f"PROTOCOL={db_params['protocol']};"
        f"UID={db_params['uid']};"
        f"PWD={db_params['pwd']};"
        f"SSL={db_params['ssl']};"
        f"SECURITY={db_params['security']};"
        f"SSLServerCertificate={db_params['sslservercertificate']};",
        "", ""
    )
    print("Connected to the database successfully!")
except Exception as e:
    print(f"Error connecting to the database: {str(e)}")

# Video feed
cap = None
posList = []
width, height = 107, 48


def initialize_video():
    global cap, posList
    cap = cv2.VideoCapture('..\carPark.mp4')

    with open('..\ModelBuilding\CarParkPos', 'rb') as f:
        posList = pickle.load(f)


def check_parking_space(imgPro, img):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))


def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        check_parking_space(imgDilate, img)

        # Encode frame as JPEG image
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user exists and the password is correct
        sql = "SELECT * FROM REGISTER WHERE name = ? AND password = ?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, username)
        ibm_db.bind_param(stmt, 2, password)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)

        if account:
            session['username'] = username  # Store the username in the session
            return redirect('/select')
        else:
            error_message = 'Invalid username or password'
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        sql = "SELECT * FROM REGISTER WHERE name = ?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, name)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        print(account)
        if account:
            return render_template('login.html', error=True)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        else:
            insert_sql = "INSERT INTO REGISTER (name, email, password) VALUES (?, ?, ?)"
            prep_stmt = ibm_db.prepare(conn, insert_sql)
            ibm_db.bind_param(prep_stmt, 1, name)
            ibm_db.bind_param(prep_stmt, 2, email)
            ibm_db.bind_param(prep_stmt, 3, password)
            ibm_db.execute(prep_stmt)
            return render_template('login.html', success=True)
    return render_template('signup.html', msg=msg)

@app.route('/select')
def select():
    return render_template('select.html')


@app.route('/predict')
def predict():
    initialize_video()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
