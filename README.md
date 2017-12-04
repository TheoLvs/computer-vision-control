# Behavior detection and control via Computer Vision
![](https://www.stemmer-imaging.se/media/uploads/products/software/CVB/CVB-Optical-Flow-App1-I0.jpg)

The idea is to control your computer by **detecting specific movements and behaviors** on your laptop webcam by associating motions to actions.<br>
This is a personal project whose goal is to fully implement and deploy a Deep Learning algorithm in a live setting.<br>
But also to compare the complexity between a Machine Learning implementation and a hand crafted one using classical image processing methods such as those from OpenCV. 

##### Goals : 
- Build a project from scratch to production
- Real understanding of the hyperparameters
- Reproduce teachable machines
- Make a model that runs on CPU and a small laptop and deployable to work at 60FPS with a webcam

### Possible actions to implement
- Control Spotify

***
## References
### References for the hand-crafted implementation
- [Hand tracking recognition](http://sa-cybernetics.github.io/blog/2013/08/12/hand-tracking-and-recognition-with-opencv/)
- [Finger tracking tutorial](https://picoledelimao.github.io/blog/2015/11/15/fingertip-detection-on-opencv/)
- [Detecting Hands and Recognizing Activities in Complex Egocentric Interactions](http://vision.soic.indiana.edu/papers/egohands2015iccv.pdf)
- [Handmap blog](https://handmap.github.io/dlib-classifier-for-object-detection/)
- [Scikit image documentation](http://scikit-image.org/docs/dev/auto_examples/)

### References for the Machine Learning implementation
- [Deep Learning for Integrated Hand Detection and Pose Estimation](http://vision.unipv.it/CV/materiale2016-17/4th%20Choice/0257.pdf)
- [Deep Learning Based Hand Detectionin Cluttered Environment Using Skin Segmentation](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w11/Roy_Deep_Learning_Based_ICCV_2017_paper.pdf)
- [Real Time Full Hand tracking](https://www.youtube.com/watch?v=OLL_F4LV0YM) - and the [github repo](https://github.com/timctho/convolutional-pose-machines-tensorflow)
- [Nice tutorial on stackoverflow](https://stackoverflow.com/questions/44491350/deep-learning-for-hand-detection)
- [Paper on real time pose estimation](https://arxiv.org/abs/1611.08050) - [Repo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) and [Chainer implementation](https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation)
- [Tutorial for Tensorflow implementation](https://towardsdatascience.com/how-to-build-a-real-time-hand-detector-using-neural-networks-ssd-on-tensorflow-d6bac0e4b2ce)


### Datasets 
- [Database for hand gesture recognition](http://sun.aei.polsl.pl/~mkawulok/gestures/)
- [Hand Dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/)
- [VIVA hand detection benchmark](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/)
- [EgoHands](http://vision.soic.indiana.edu/projects/egohands/)
- [Mujah dataset](https://www.mutah.edu.jo/biometrix/hand-images-databases.html)
- [Google dataset](https://sites.google.com/view/11khands)
