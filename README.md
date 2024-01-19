# rsantha-kmandak-arsivak-hasugum-finalProject

Generative adversarial Networks (GANs) has become prevalent in producing realistic images, videos and other forms of data for malicious purpose such as spreading fake news
in social media. In order to detect and stop the exploitation of this technology, it is now crucial to detect GAN generated images. GAN generated images often have some 
artifacts different from real images. So, In this project, we explored four different approaches for detecting GAN generated images: 


1. Analyzing RGB features using a convolutional neural network (CNN) model
2. Fourier Transform analysis on the frequency space
2. Analyzing frequency space to identify artifacts 
3. Using a support vector machine (SVM) classifier to distinguish between real and fake images using the landmark location of facial features

We used dataset of GAN generated images provided by Professor Fil Menzer and a dataset of real images from CELEB-A HQ which contains images of celebrities. The performance of all approaches was evaluated on the testing 
dataset and some real world images from Instagram public profiles. Our models were found to be effective in detecting style GAN generated images. In the future, we aim 
to generalize our approaches to detect GAN generated images by other architecture.
