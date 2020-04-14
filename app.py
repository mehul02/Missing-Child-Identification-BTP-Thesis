import tkinter as tk
from tkinter import ttk, Tk
from tkinter.filedialog import askopenfilename

import cv2
import sys, os

import pylab

%pylab qt

from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from keras.models import load_model

import mtcnn

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

root = tk.Tk

file_path = "D:/Downloads/Compressed/facematch-master/facematch-master/images/test.jpg"

"""
facenet_model = load_model(
    "D:/Downloads/BTP/Models/model/facenet_keras.h5", compile=False
)
"""

face_array_ = [
    (1.0, 2.0, 1.5, 1.5, 3, 0.0),
    (1.0, 2.0, 1.5, 1.5, 3, 0.0),
    (1.0, 2.0, 1.5, 1.5, 3, 0.0),
    (1.0, 2.0, 1.5, 1.5, 3, 0.0),
]

details = {}
details["Ananya"] = {"Address": "123 hazira Gwalior M.P.", "Phone": "8989855234"}
details["Anuj"] = {"Address": "437 Teachers Colony Udaipur Raj.", "Phone": "8989575241"}
details["Mehul"] = {"Address": "42 Sector 5 Udaipur Raj.", "Phone": "7285828278"}
details["Gopal"] = {"Address": "232 hazira Gwalior M.P.", "Phone": "7828272472"}
details["Kiran"] = {"Address": "42 Sector 5 Udaipur Raj.", "Phone": "7285828278"}
details["Lovely"] = {"Address": "34 Sector 5 Udaipur Raj.", "Phone": "9898989898"}
details["Mayank"] = {"Address": "4 Sector 5 Udaipur Raj.", "Phone": "9859829454"}
details["Praveen"] = {"Address": "123 hazira Gwalior M.P.", "Phone": "9897821213"}
details["Rakesh"] = {"Address": "23 Sector 8 Udaipur Raj.", "Phone": "9897521232"}
details["Riya"] = {"Address": "333 Sector 1 Vapi Guj.", "Phone": "9232521454"}
details["Saumya"] = {"Address": "545 City Center Gwalior M.P.", "Phone": "9562123354"}
details["Saurabh"] = {"Address": "677 Padav Gwalior M.P.", "Phone": "9874121233"}
details["Suman"] = {"Address": "222 Sector 51 Udaipur Raj..", "Phone": "8979846415"}
details["Suresh"] = {"Address": "2 Sector 52 Kanpur U.p.", "Phone": "9854212154"}


def popmsg_notfind():
    NORM_FONT = ("Verdana", 15)
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text="No match found!", font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay!", command=lambda: popup.destroy())
    B1.pack()
    popup.mainloop()


def show_img(path):
    plt.imshow(path)
    plt.show()


class BtpAPP(root):
    def __init__(self, *args, **kwargs):

        root.__init__(self, *args, **kwargs)

        root.iconbitmap(self, default="F:/Tkinter/BTP/icon.ico")
        root.wm_title(self, "Missing Child Identification")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.config(background="#E5E4E2")
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def select_file(self):
        global file_path
        file_path = askopenfilename()
        self.show_frame(PageOne)

    def add_child(self):
        window = tk.Tk()
        label = ttk.Label(window, text="Name ")
        label.grid(row=0, column=0)
        my_entry = ttk.Entry(window, width=50)
        my_entry.grid(row=0, column=1)

        label1 = ttk.Label(window, text="Address ")
        label1.grid(row=2, column=0)
        my_entry1 = ttk.Entry(window, width=50)
        my_entry1.grid(row=2, column=1)

        label2 = ttk.Label(window, text="Phone No ")
        label2.grid(row=4, column=0)
        my_entry2 = ttk.Entry(window, width=50)
        my_entry2.grid(row=4, column=1)

        def select_file():
            global file_path
            file_path = askopenfilename()

        button1 = ttk.Button(
            window, text="Select the image of the child", command=lambda: select_file()
        )
        button1.grid(row=6, column=1)

        def save_details():
            name = my_entry.get()
            addr = my_entry1.get()
            phone = my_entry2.get()
            d = {}
            d[name] = {"Address": addr, "Phone No:": phone}
            with open("D:/Downloads/BTP/details.txt", "a") as myfile:
                myfile.write(str(d) + " " + file_path)
                myfile.write("\n")
            window.destroy()

        button2 = ttk.Button(window, text="Save", command=lambda: save_details())
        button2.grid(row=8, column=1)

    def Machine_learning(self):
        detector = MTCNN()

        img = cv2.imread(file_path)
        img = cv2.resize(img, (500, 600))

        result = detector.detect_faces(img)

        bounding_box = result[0]["box"]

        keypoints = result[0]["keypoints"]

        cv2.rectangle(
            img,
            (bounding_box[0], bounding_box[1]),
            (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            (0, 255, 0),
            2,
        )

        cv2.circle(img, (keypoints["left_eye"]), 3, (0, 255, 0), 3)
        cv2.circle(img, (keypoints["right_eye"]), 3, (0, 255, 0), 3)
        cv2.circle(img, (keypoints["nose"]), 3, (0, 255, 0), 3)
        cv2.circle(img, (keypoints["mouth_left"]), 3, (0, 255, 0), 3)
        cv2.circle(img, (keypoints["mouth_right"]), 3, (0, 255, 0), 3)

        cv2.imshow("Face", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # show_frame(PageTwo)

    def Classification(self):
        def extract_face(filename, required_size=(160, 160)):
            # load image from file
            image = Image.open(filename)
            # convert to RGB, if needed
            image = image.convert("RGB")
            # convert to array
            pixels = np.asarray(image)
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            results = detector.detect_faces(pixels)
            # extract the bounding box from the first face
            if len(results) == 0:
                return []
            x1, y1, width, height = results[0]["box"]
            # deal with negative pixel index
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = np.asarray(image)
            return face_array

        data = np.load("D:/Downloads/BTP/face_dataset_mo.npz")

        # Invariant Points
        invar = []
        for face in face_array_:
            invar.append(
                np.sqrt(np.square(face[0] - face[3]) + np.square(face[1] - face[4]))
            )

        trainX, trainy, testX, testy = (
            data["arr_0"],
            data["arr_1"],
            data["arr_2"],
            data["arr_3"],
        )
        # facenet_model = load_model("D:/Downloads/BTP/Models/model/facenet_keras.h5")

        embeddings = np.load("D:/Downloads/BTP/face_embeddings_mo.npz")
        emdTrainX, trainy, emdTestX, testy = (
            embeddings["arr_0"],
            embeddings["arr_1"],
            embeddings["arr_2"],
            embeddings["arr_3"],
        )

        def get_embedding(model, face):
            # scale pixel values
            face = face.astype("float32")
            # standardization
            mean, std = face.mean(), face.std()
            face = (face - mean) / std
            # transfer face into one sample (3 dimension to 4 dimension)
            sample = np.expand_dims(face, axis=0)
            # make prediction to get embedding
            yhat = model.predict(sample)
            return yhat[0]

        # normalize input vectors
        in_encoder = Normalizer()
        emdTrainX_norm = in_encoder.transform(emdTrainX)
        emdTestX_norm = in_encoder.transform(emdTestX)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy_enc = out_encoder.transform(trainy)
        testy_enc = out_encoder.transform(testy)

        # prefix sum
        pref = {}
        curr = 0
        pref[curr] = 0
        for i in range(len(trainy) - 1):
            if trainy[i] != trainy[i + 1]:
                curr += 1
                pref[curr] = i + 1

        # fit model
        model = SVC(kernel="linear", probability=True)
        # model = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
        # model = KNeighborsClassifier(n_neighbors=5,leaf_size=30)
        model.fit(emdTrainX_norm, trainy_enc)

        test_img_path = file_path

        curr_face = extract_face(test_img_path)

        embd = get_embedding(facenet_model, curr_face)
        embd = np.reshape(embd, (-1, len(embd)))
        embd = in_encoder.transform(embd)
        embd = np.asarray(embd)

        yhat_class = model.predict(embd)
        yhat_prob = model.predict_proba(embd)

        predict_names = out_encoder.inverse_transform(yhat_class)

        class_index = yhat_class[0]

        dist = np.sqrt(
            np.sum(np.square(np.subtract(embd, emdTrainX_norm[pref[yhat_class[0]]])))
        )

        threshold = 1.10

        if dist > threshold:
            popmsg_notfind()  # todo
        else:
            # show_img(trainX[pref[yhat_class[0]]])
            plt.imshow(trainX[pref[yhat_class[0]]])
            title_x = str(
                str(predict_names[0])
                + ", Address: "
                + str(details[str(predict_names[0])]["Address"])
                + ", Phone No: "
                + str(details[str(predict_names[0])]["Phone"])
            )
            plt.title(title_x)
            plt.text(
                x, y, s, fontdict=None, withdash=cbook.deprecation._deprecated_parameter
            )
            plt.show()
            # cv2.imshow("Face", trainX[pref[yhat_class[0]]])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def take_photo(self):
        cap = cv2.VideoCapture(0)
        width, height = 800, 600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        root_ = Tk()
        root_.bind("<Escape>", lambda e: root.quit())
        lmain = tk.Label(root_)
        lmain.pack()

        def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = PIL.Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(10, show_frame)

        show_frame()
        root_.mainloop()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Start Page")
        label.pack(pady=5, padx=30)

        button1 = ttk.Button(
            self,
            text="Check if the child is lost!",
            command=lambda: controller.select_file(),
        )
        button1.pack()

        button2 = ttk.Button(
            self,
            text="Add a child to the database",
            command=lambda: controller.add_child(),
        )
        button2.pack(pady=15)

        """
        button3 = ttk.Button(
            self,
            text="Take an image from the camera",
            command=lambda: controller.take_photo(),
        )
        button3.pack(pady=15)
        """


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Select an option")
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(
            self, text="Find Coordinates", command=lambda: controller.Machine_learning()
        )
        button1.pack(pady=0)

        button2 = ttk.Button(
            self, text="Find Match", command=lambda: controller.Classification()
        )
        button2.pack(pady=15)

        button4 = ttk.Button(
            self,
            text="Select a different Image",
            command=lambda: controller.show_frame(StartPage),
        )
        button4.pack(pady=10)


# tk.Tk().configure(background='#C2DFFF')
app = BtpAPP()
app.configure(background="#C2DFFF")
app.geometry("400x600")
app.mainloop()
