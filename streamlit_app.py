#Import for streamlit
import streamlit as st
import av
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


#Import for handling image
import cv2
from cvzone.FaceDetectionModule import FaceDetector


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#Main intelligence of the file, class to launch a webcam, detect faces, then detect emotion and output probability for each emotion
def app_emotion_detection():
    class EmotionPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            # Sign detector
            self.face_detector = FaceDetector(    )




        def find_faces(self, image):

            image_face, faces = self.face_detector.findFaces(image)
            # loop over all faces and print them on the video + apply prediction
            for face in faces:
                  print(face)
            #     bbox = face["bbox"]

            #     rectangle = cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20),
            #                               (bbox[0] + bbox[2] + 20,
            #                                bbox[1] + bbox[3] + 20),
            #                               (255, 0, 255), 2)
                #load model
                #model = retrieve_model()

                # prediction
                # prediction = model.predict(
                #     np.expand_dims(tf.image.resize(
                #         (rectangle), [64, 64]),
                #         axis=0) / 255.0)

                # prediction_max = np.argmax(prediction)
                # pred = mapping[prediction_max]
                # #check on terminal the prediction
                # print(pred)

                # #draw emotion on images
                # cv2.putText(image_face, pred, (bbox[0] + 130, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                #                 2, (255, 0, 255), 2)
            return faces, image_face

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            faces, annotated_image = self.find_faces(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


############################ Sidebar + launching #################################################

object_detection_page = "Emotion video detector"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        object_detection_page,
    ],
)
st.subheader(app_mode)
if app_mode == object_detection_page:
    app_emotion_detection()
