import winsound
import numpy as np
import cv2
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from tkinter import filedialog, Tk, Button, Label
from PIL import Image, ImageTk
from flask import Flask, render_template,request
from ultralytics import YOLO
import cv2
import os
import numpy as np
import cv2
import os
from twilio.rest import Client
# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def send_email_alert(image_path):
    try:
        sender_email = "ms4400073@gmail.com"
        sender_password = "szsj zkvx wlbx jeqh"
        #receiver_email = "sanatanas2580@gmail.com"
        receiver_email = "arjunkrishnakr17@gmail.com"
        subject = "Fall Detection Alert"
        body = "Hi sanat :)"

        # Create the email message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
        "Content-Disposition",
        f"attachment; filename= {os.path.basename(image_path)}",
        )
        message.attach(part)

        # Connect to the email server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())

        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Route for serving login page
@app.route('/', methods=["GET", "POST"])
def login():
    if request.method=="POST":
        if 'video' not in request.files:
             print('No file part')
        file = request.files['video']
        if file.filename == '':
            print('No selected file')
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('File uploaded successfully')
        selected_model=None
        selected_model = request.form.get("selected_model")
        print(selected_model)
        print("helloo")
        flag=False
        if selected_model=="webcam":
            print("sanat")
            flag=True
        if selected_model=="model1" or selected_model=="webcam":
            MODEL_PATH = "yolov4-coco"
            MIN_CONF = 0.3
            NMS_THRESH = 0.3

            # RESIZE_WIDTH=1280
            RESIZE_WIDTH=1280
            RESIZE_HEIGHT=720

            labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            weightsPath = os.path.sep.join([MODEL_PATH, "yolov4.weights"])
            configPath = os.path.sep.join([MODEL_PATH, "yolov4.cfg"])
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            src = cv2.cuda_GpuMat()
            ln = net.getLayerNames()
            ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
            print("Accessing video stream...")

            output_directory = "model1_fall_detected"

            # Create the directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            def detect_people(frame, net, ln, personIdx=0):
                (H, W) = frame.shape[:2]
                results = []
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                layerOutputs = net.forward(ln)
                boxes = []
                centroids = []
                confidences = []
                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if classID == personIdx and confidence > MIN_CONF:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            boxes.append([x, y, int(width), int(height)])
                            centroids.append((centerX, centerY))
                            confidences.append(float(confidence))
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                        results.append(r)
                return results


            def send_email_alert(image_path):
                try:
                    sender_email = "ms4400073@gmail.com"
                    sender_password = "szsj zkvx wlbx jeqh"
                    receiver_email = "sanatanas2580@gmail.com"
                    #receiver_email = "arjunkrishnakr17@gmail.com"
                    subject = "Fall Detection Alert"
                    body = "model1-detection"

                    # Create the email message
                    message = MIMEMultipart()
                    message['From'] = sender_email
                    message['To'] = receiver_email
                    message['Subject'] = subject
                    message.attach(MIMEText(body, 'plain'))

                    with open(image_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(image_path)}",
                    )
                    message.attach(part)

                    # Connect to the email server and send the email
                    with smtplib.SMTP('smtp.gmail.com', 587) as server:
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, receiver_email, message.as_string())

                    print("Email sent successfully!")
                except Exception as e:
                    print(f"Error sending email: {e}")

            def save_fall_detected_frame(frame, output_directory, frame_count):
                # Construct the filename for the saved frame
                filename = os.path.join(output_directory, f"fall_detected_frame_{frame_count}.jpg")
                # Save the frame as an image file
                cv2.imwrite(filename, frame)
                print(f"Fall detected frame saved as: {filename}")
                return filename        


            def process_video(net,ln):
                j = 0
                previous_cY = 0
                frame_count = 0
                while True:
                    (grabbed, frame) = vs.read()
                    if not grabbed:
                        break
                    #Fwidth = 700
                    #Fheight = frame.shape[0]
                    #dim = (Fwidth, Fheight)
                    frame = cv2.resize(frame, (RESIZE_WIDTH,RESIZE_HEIGHT))
                    results = detect_people(frame, net, ln)
                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)
                        x, y = startX, startY
                        w = endX - startX
                        h = endY - startY
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)
                        print(h, " ", w)
                        print(j)
                        if h < (0.75 * w) :
                            j += 1
                            fall_speed = cY - previous_cY
                            if j > 2 and fall_speed > 5 :
                                print("snant")
                                cv2.putText(frame, 'FALL Detected', (int(x + 0.5 * w), y - 5), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255),
                                            2)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                                image_path=save_fall_detected_frame(frame, output_directory, frame_count)
                                frame_count += 1
                                # Send email alert
                                send_email_alert(image_path)

                        if h >= (0.75 * w):
                            j = 0
                        print(j)
                        previous_cY = cY

                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                vs.release()
                cv2.destroyAllWindows()

            #video_path = "testvideo5.webm"
            if flag:
                print("hii")
                vs = cv2.VideoCapture(0)
                for _ in range(30):
                    vs.read()
                process_video(net,net.getUnconnectedOutLayersNames())
            else:
                video_path = "uploads/"f"{filename}"    
                vs = cv2.VideoCapture(video_path)
                
            #vs = cv2.VideoCapture(0)
            
            
        if selected_model=="model2":
            print("helloo")
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

            # model = YOLO("./runs/detect/train/weights/best.pt")
            model = YOLO("fall_det_1.pt")

            video_path = "uploads/"f"{filename}"
            cap = cv2.VideoCapture(video_path)
            output_directory = "model2_fall_detected"

            # Create the directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            def save_fall_detected_frame(frame, output_directory, frame_count):
                # Construct the filename for the saved frame
                filename = os.path.join(output_directory, f"fall_detected_frame_{frame_count}.jpg")
                # Save the frame as an image file
                cv2.imwrite(filename, frame)
                print(f"Fall detected frame saved as: {filename}")
                return filename  
            frame_count = 0 
            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = model.track(frame, persist=True, conf=0.5)
                    print("sant")
                    #print(str(results[0]))
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    """if "Fall-Detected" in str(results[0]):
                    # Save the annotated frame
                        save_fall_detected_frame(annotated_frame, output_directory, frame_count)
                        frame_count += 1"""

                    # Display the annotated frame
                    for result in results:
                        if result.boxes:
                            red_boxes_present = True
                            print("arjun")
                            image_path=save_fall_detected_frame(annotated_frame, output_directory, frame_count)
                            frame_count+=1
                            break
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    
                    break
            def send_email_alert(image_path):
                try:
                    sender_email = "ms4400073@gmail.com"
                    sender_password = "szsj zkvx wlbx jeqh"
                    receiver_email = "arjunkrishnakr17@gmail.com"
                    #receiver_email = "arjunkrishnakr17@gmail.com"
                    subject = "Fall Detection Alert"
                    body = "model2-detection"

                    # Create the email message
                    message = MIMEMultipart()
                    message['From'] = sender_email
                    message['To'] = receiver_email
                    message['Subject'] = subject
                    message.attach(MIMEText(body, 'plain'))

                    with open(image_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(image_path)}",
                    )
                    message.attach(part)

                    # Connect to the email server and send the email
                    with smtplib.SMTP('smtp.gmail.com', 587) as server:
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, receiver_email, message.as_string())

                    print("Email sent successfully!")
                except Exception as e:
                    print(f"Error sending email: {e}")
    
            send_email_alert(image_path)    

            # Release the video capture object and close the display window
            cap.release()
            cv2.destroyAllWindows()

        if selected_model=="model3":
            MODEL_PATH = "yolov4-coco"
            display = 1
            MIN_CONF = 0.3
            NMS_THRESH = 0.3
            labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            weightsPath = os.path.sep.join([MODEL_PATH, "yolov4.weights"])
            configPath = os.path.sep.join([MODEL_PATH, "yolov4.cfg"])
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            src = cv2.cuda_GpuMat()
            ln = net.getLayerNames()
            ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
            print("Accessing video stream...")
            output_directory = "model3_fall_detected"

            # Create the directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            def save_fall_detected_frame(frame, output_directory, frame_count):
                # Construct the filename for the saved frame
                filename = os.path.join(output_directory, f"fall_detected_frame_{frame_count}.jpg")
                # Save the frame as an image file
                cv2.imwrite(filename, frame)
                print(f"Fall detected frame saved as: {filename}")
                return filename

            video_path = "uploads/"f"{filename}"
            vs = cv2.VideoCapture(video_path)
            writer = None

            def detect_people(frame, net, ln, personIdx=0):
                (H, W) = frame.shape[:2]
                results = []
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                layerOutputs = net.forward(ln)
                boxes = []
                centroids = []
                confidences = []
                BoxCoordinates = []
                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if classID == personIdx and confidence > MIN_CONF:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            boxes.append([x, y, int(width), int(height)])
                            centroids.append((centerX, centerY))
                            confidences.append(float(confidence))
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                        results.append(r)
                return results
            def send_email_alert(image_path):
                try:
                    sender_email = "ms4400073@gmail.com"
                    sender_password = "szsj zkvx wlbx jeqh"
                    receiver_email = "arjunkrishnakr17@gmail.com"
                    #receiver_email = "arjunkrishnakr17@gmail.com"
                    subject = "Fall Detection Alert"
                    body = "model3-detection"

                    # Create the email message
                    message = MIMEMultipart()
                    message['From'] = sender_email
                    message['To'] = receiver_email
                    message['Subject'] = subject
                    message.attach(MIMEText(body, 'plain'))

                    with open(image_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(image_path)}",
                    )
                    message.attach(part)

                    # Connect to the email server and send the email
                    with smtplib.SMTP('smtp.gmail.com', 587) as server:
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, receiver_email, message.as_string())

                    print("Email sent successfully!")
                except Exception as e:
                    print(f"Error sending email: {e}")

            def save_fall_detected_frame(frame, output_directory, frame_count):
                # Construct the filename for the saved frame
                    filename = os.path.join(output_directory, f"fall_detected_frame_{frame_count}.jpg")
                    # Save the frame as an image file
                    cv2.imwrite(filename, frame)
                    print(f"Fall detected frame saved as: {filename}")
                    return filename          
            frame_count=0
            j = 0
            while True:
                (grabbed, frame) = vs.read()
                if not grabbed:
                    break
                Fwidth = 700
                Fheight = frame.shape[0]
                dim = (Fwidth, Fheight)
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
                for (i, (prob, bbox, centroid)) in enumerate(results):
                    (startX, startY, endX, endY) = bbox
                    (cX, cY) = centroid
                    color = (0, 255, 0)
                    x, y = startX, startY
                    w = endX - startX
                    h = endY - startY
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.circle(frame, (cX, cY), 5, color, 1)
                    print(h, " ", w)
                    if h < (0.75 * w):
                        j += 1
                    # Conditions for fall detection contour Area
                    if j > 2:
                        cv2.putText(frame, 'FALLEN', (int(x + 0.5 * w), y - 5), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        image_path=save_fall_detected_frame(frame, output_directory, frame_count)
                        frame_count += 1
                        # Send email alert
                        #send_email_alert(image_path)
                        # Play multiple beep sounds when fall detected
                        for _ in range(3):
                            winsound.Beep(1000, 500)  # Adjust frequency (Hz) and duration (ms) as needed
                    if h >= (0.75 * w):
                        j = 0
                    print(j)

                  

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            send_email_alert(image_path)    
            vs.release()
            cv2.destroyAllWindows()







    return render_template('login.html')

# Route for sending email alert
@app.route('/send_email_alert', methods=['GET'])
def send_email_alert_endpoint():
    send_email_alert()
    return '', 200

# Route for processing video from webcam
@app.route('/process_webcam_video', methods=['GET'])
def process_webcam_video_endpoint():
    process_webcam_video()
    return '', 200

# Route for processing uploaded video
@app.route('/process_uploaded_video', methods=['GET'])
def process_uploaded_video_endpoint():
    process_uploaded_video()
    return '', 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run the Flask app