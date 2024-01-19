# Detecting GAN Generated Images
<p align="center">
  <img src="https://github.com/rakesh09111996/Detecting-GAN-generated-Images/blob/b5076a88aa223a68f6e20b5c61791691491460d1/Intro.PNG" alt="Facial Landmark points detected for given image">
</p>

# Abstract
Generative adversarial Networks (GANs) has become  prevalent in producing realistic images, videos and other forms of data for malicious purpose such as spreading fake  news in social media. In order to detect and stop the  exploitation of this technology, it is now crucial to detect  GAN generated images. GAN generated images often have  some artifacts different from real images. So, In this  project, we explored four different approaches for  detecting GAN generated images: Analysing by exploiting  inherent differences in the co-occurrence patterns of  different image bands, a Fast Fourier Transform (FFT) is  applied on each of the RGB channels of the input images  and then deep learning technique applied, analyzing frequency space to identify artifacts, and using a support  vector machine (SVM) classifier to distinguish between real and fake images using the landmark location of facial features. We used dataset of GAN generated images provided by Professor Fil Menzer and a dataset of real images from CELEB-A HQ which contains images of celebrities. The performance of all approaches was evaluated on the testing dataset and some real world images from Instagram public profiles. Our models were found to be effective in detecting style GAN generated images. In the future, we aim to generalize our approaches to detect GAN generated images by other architecture

# Dataset
## GAN Generated Images
For GAN generated images, one of the professor at IU have generated images via GAN model and shared link which contains 10,000 images with a resolution of  1024x1024 pixels. There were total of 3,00,000 images,  out of which we choose 10,000 images for training  purposes in all four of our approaches
## Real Images
For real images, we used CelebA dataset which is now a well-liked dataset for training machine learning and deep learning models for variety of face related tasks. It has over 2,02,599 images of celebrity with resolution of 1024x1024 pixels. The images in CelebA are centeredand cropped such that faces are around the same size and orientation. For training purpose of our four approaches we choose 10,000 images from the CelebA dataset.

We used dataset of GAN generated images provided by Professor Fil Menzer and a dataset of real images from CELEB-A HQ which contains images of celebrities. The performance of all approaches was evaluated on the testing 
dataset and some real world images from Instagram public profiles. Our models were found to be effective in detecting style GAN generated images. In the future, we aim 
to generalize our approaches to detect GAN generated images by other architecture.

# Models
## Eye Coordinate Analysis
In this method, we investigate the eye coordinates of the GAN generated images to identify inconsistency. The lack of physical limitations in the GAN models is what causing this discrepancy. The eye positions of GAN generated images are in same position.
 
## RGB Analysis
### Using cross band co-occurrences 
This approach can effectively differentiate between GAN-generated images and real images by exploiting the inherent differences in the co-occurrence patterns of different image bands by analyzing the cross-band co-occurrences in GAN-generated face images. 
### Using FFT
The goal of this approach is to classify images as "Real" or "Fake GAN generated" using a deep learning model. To achieve this, a Fast Fourier Transform (FFT)is applied on each of the RGB channels of the input images. The transformed data is then shifted to the center using FFT shift and normalized. Finally, the original RGB channels are replaced with the transformed and normalized channels, and the images are fed into a deep neural network for classification

## Frequency Analysis
In this method we investigate the frequency domain of the GAN generated images to identify artifacts. To analyze the images in frequency domain, we use the discrete cosine transform (DCT). The DCT is similar to Discrete Fourier Transform (DFT) where sit sums cosine functions that oscillate at various frequencies to represent a finitesequence of data points. The DCT is used because it effectively compresses the energy in image signal and can be separated for effective implementations. Since, StyleGAN generated images exhibit artifacts in frequency spectrum, the aim is to analyze if there is any common occurrence of pattern in Style GAN images by training the DCT transformed images using convolutional neural network

## Face feature Analysis
In this method we use the facial landmark positions to expose GAN synthesized images. The approach is based on the observation that the facial components configuration produced by GAN models differs from that of real faces since there aren’t any global limitations.  This approach is more detailed and comprehensive that uses 32 facial points instead of 2 points. A SVM classifier is trained on the facial landmark locations to classify GAN synthesized faces from the real face.
<p align="center">
  <img src="https://github.com/rakesh09111996/Detecting-GAN-generated-Images/blob/335fbf187666e3054fa3f366189601e67e5c5349/Facial_landmarks.png" alt="Facial Landmark points detected for given image">
</p>

The feature vectors and labels of train and test set are split into 80:20 ratio using train_test_split function from the sklearn model selection. A grid search is used to perform exhaustive search over grid parameters to find the optimal hyperparameters for the SVM classifier. 5-fold cross validation object is used to ensure balance between the training and validation sets. The SVM classifier is trained with the feature vectors and corresponding labels using the method of the grid search object. The model is then tested with profile images of Instagram public account. Using Instagram API, user profile information is fetched including user profile picture by providing Instagram user ID. The landmark facial key points are detected using the dlib library. The predicted landmark are combined and then normalized to [0,1]x[0,1] region. The profile picture is determined GAN or not using the normalized feature vectors by the pre trained SVM model.

# Conclusion
To summarize, four distinct methods were utilized to identify images generated by StyleGAN. The SVM classifier, which relied on facial feature landmark positioning, yielded the least accurate results. On the other hand, frequency analysis approaches provided the most accurate results, with crossband cooccurrence analysis methods coming in a close second
