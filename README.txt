CrossBandCooccurence - 4.2.1

1. Make sure you have installed the required dependencies  
    TensorFlow
    Keras
    OpenCV
    NumPy
    Matplotlib

2. Set the appropriate values for train_dir, img_width, and img_height variables according to your directory structure and image size.

3. Data structure: Fake images in one folder and real images in another folder.

3. Run the CrossbandCooccurence.py

FourierTransform - 4.2.2

1. Make sure you have installed the required dependencies -  
TensorFlow
    Keras
    OpenCV
    NumPy
    Matplotlib

2. Set the appropriate values for base_path, train_dir, img_width, and img_height variables according to your directory structure and image size.

3. Data structure: Fake images in one folder and real images in another folder.

3. If you have a GPU available and want to utilize it, uncomment the code that sets the GPU device.

SVM - 4.4

Note:
Go to the dlib website (http://dlib.net/files/).
Download the file "shape_predictor_68_face_landmarks.dat.bz2" and extract it using a decompression software (e.g., 7-Zip).
Once extracted, you will find the "shape_predictor_68_face_landmarks.dat" file, which you can use in your code.

1. Download a 10000 images of gan from the google drive link - https://drive.google.com/drive/folders/14lm8VRN1pr4g_KVe6_LvyDX1PObst6d4 and save the entire 10000 images under a single folder ( note avoid any subfolders)
2. Download a 10000 images of real images from the Celeb-A dataset and save the images under a single folder.
3. give the path of gan image folder in "folder_path" variable of orig_new.py rountine and download the file "shape_predictor_68_face_landmarks.dat" and give the path of that file  for the function (dlib.shape_predictor) in the same routine. Then execute the orig_new.py routine that will generate a feature vectors and save it as a .npy file 
4.  give the path of real image folder in "folder_path" variable of orig_new.py rountine and download the file "shape_predictor_68_face_landmarks.dat" and give the path of that file  for the function (dlib.shape_predictor) in the same routine. Then execute the orig_new.py routine that will generate a feature vectors and save it as a .npy file 
5. Then execute the gan_svm.py after specifying the path of features corresponding to real images for real_features variable's load function. similarly specify the path of generated feature vector file of the gan images to the gan_features variable's load function.

To detect the gan generated image for instagram id:
After the above execution, the trained model is saved as svm_model.joblib. in detect_insta_gan.py routine specify the saved path of svm  model file for the svm_model variable's load function. give a instagram user id and password of a specific account for username and password variable. after execution you could specify the specific public account instagram id so that this routine determines whether the profile picture of that account is gan generated or not 

Freuqnecy Analysis - 4.3

1. Make sure you have installed the required dependencies -  
TensorFlow
Keras
OpenCV
NumPy
Pandas
Matplotlib
imageio

2. Replace dataset_path in freq_analysis.py code
3. Combine both real and GAN image into a single folder. Assumption .png (GAN images) and .jpg (Real images)
4. Execute freq_analysis.py




