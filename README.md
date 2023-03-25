This is a new method of matching how close an element is with another by utilizing all values in the space (the whole image). This is different from the previous approach where only the closest neighbors were considered

The approach involves the following steps
### 1: Converting image representation into a polar coordinates
This is achieved by using each element as the center of the image and finding the difference between the element and each of the others in terms of euclidean distance and difference in absolute angle. Recall that angle in this case is just how vertical or horizontal the line is

Essentially, we will end up with a similar number of images as the number of elements in the original image. Each of these image will be from the perspective of each item that constituted the original image

### 2: Modifying the angle and distance values to their ranks in the arrangement
Here, angles and distances are converted to ranks. The smallest distance is ranked 1, while the last is ranked last. Same thing is done with angle difference. Items whose angle difference between themselves and the referent element are given rank 1 with respect to angle

### 3: Comparing elements
A comparison of elements is just a comparison between the two images obtained above. Comparison is done on an element to element basis. Element here, is an "element" in the image created from the perspective of an element
