# **Finding Lane Lines on the Road** 

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted an input image to grayscale, then I applied Gaussian smoothing to the grayscale image. Next, I found edges in the resulting image using Canny edge detector. Next, I defined a region-of-interest (ROI) and extracted edges inside this ROI. Finally, I applied Hough transform to detect lines.

In order to draw a single line for each lane, I modified the draw_lines() function by classifying each line segment as either belonging to left lane, right lane or as an outlier. Then for each lane, I compute the median of the line parameters (slope and intercept) to get the final result.

### 2. Identify potential shortcomings with your current pipeline.

One potential shortcoming would be what would happen when there are bends in the road, resulting in curvy lane lines. In such a situation, fitting a single line to find a lane is not ideal. Another shortcoming is when there are shadows or residual lane markings on the road, outlier line segments can be present inside the region-of-interest.

### 3. Suggest possible improvements to your pipeline.

One possible improvement would be to detect lanes in a temporally consistent manner. Another possible improvement would be to automatically determine the upper extent of the left and right lanes. This could be done by calculating the intersection point of the two lines.