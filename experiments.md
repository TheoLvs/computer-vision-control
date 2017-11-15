# Experiments Journal


## Hand Detection Experiments
![](http://www.intorobotics.com/wp-content/uploads/2013/11/pdoj-vfkd90skjvf0dvdfv0df000000004.png)

### With Deep Learning
- 200 images in each class (2 class) captured from the webcam with Canny edges detection (99% accuracy with a shallow network on train and validation set) — overfitting to the validation set(size 320*240)
- 350 images in each class, size 320*240, Canny edges detection — overfitting totally as well
- same 350 images without Canny edges detection, just black and white, a little preprocessing only via rescaling (/255) — no result in production, still a very shallow network
- Same 350 images with and without canny edges detection, little preprocessing, but with a convnet
- Data augmentation (zoom, shift, rotation) to build a new dataset with 3500 images in each class but with a size of 160*120
- Very shallow neural network on the 3500 images, too much bias cannot learn (struggles to reach 60% in train accuracy)
- A deeper shallow neural networks (more hidden units, 3 or 4 layers) : ok for the bias but now a variance problem (easily reach in 50 epochs a 85% accuracy for the training set, but only 55% for the validation set)
- Added dropout helped reduce the variance (85% versus 75% between training and validation), but still unable to detect anything in a production live setting, the model is still working fast enough to work at 60FPS though even if the model size is now 60Mo. No preprocessing other than black and white and rescaling was a little better than with canny edges detection
- CNN architecture trained from scratch on the edges version reached easily 90% and 86% on the training and validation sets. It worked a little bit more in production, but still unexpectedly overfits to a detail in the background
- CNN architecture using a pretrained base model on Image Net and then added a fully connected neural network on top of it. I didn't go too far with this option, even if it could have been the best, because calculating the embedding with the pre trained base was too slow (300ms for a single sample), thus obviously not usable at 60FPS 
- CNN architecture on the black and white version only, works really well on the 6000 images, (90% accuracy for the training and validation set). But unfortunately it worked less in a live production setting than the canny edges detection. Too much sensitive to the background
- New attempt by creating a dataset with background reduction algorithm, without data augmentation. CNN and vanilla neural networks give good accuracy for the training and validation set (90%). But don't work well in a live setting


### With classical CV algorithms
- Convert to HSV color scale and filter to keep only the skin (ideally remove the background)
- Apply a median blur to remove some noise
- Dilate the remaining points to fill the hand shape
- Find the contour



***
## Face ID
![](http://globbsecurity.fr/wp-content/uploads/2017/09/face-id-3d-scanning-points.jpg)

##### Face detection and bounding box
- Face detection and cropping with OpenCV Cascade classifiers


##### Face recognition algorithm
- First attempt with 440 images of 6 persons, me amongst them with 80 images : fully connected neural network (2 hidden layers (512,256) trained with Stochastic Gradient Descent for 150 epochs) - 90% as training accuracy (no validation accuracy because too small dataset) - work already quite well in production, high probability for me, but still a lot of false positives