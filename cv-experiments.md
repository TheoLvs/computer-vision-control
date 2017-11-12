# Computer vision experiments


## Detection with classical computer vision algorithms
- Convert to HSV color scale and filter to keep only the skin (ideally remove the background)
- Apply a median blur to remove some noise
- Dilate the remaining points to fill the hand shape
- Find the contour