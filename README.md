# Missing-Child-Identification-BTP-Thesis
During the Btech thesis project we developed a deep learning model to identify missing children using Face Recognition, where the photograph of any child can be used to check in the lost children database. We used MTCNN algorithm and pre-trained FaceNet model by google and trained the dataset over XGBoost classifier.
The following steps are perfomed by the model:

Step 1: Given an input image the image is resized and the with the help of mtcnn algorithm the bounding boxes are identified and the overlapping boxes are removed with NMS.
![](images/img1.PNG)

Step 2: The facenet Model trained on millions of images gives us a 128D vector which represents a face.
![](images/img2.PNG)

An application using TKinter was made to give the work a UI.

![](images/img3.PNG)

If you want to add a child to the database you can fill the form.

![](images/img6.PNG)

Otherwise if you want to search you get this UI

![](images/img4.PNG)

Then you can find the coordinates as well as other information.

![](images/img5.PNG)
![](images/img7.png)
