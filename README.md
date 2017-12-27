## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/tests/camera/undistort/undistorted_test1.jpg "undistorted Image 1"
[image2]: ./test_images/test1.jpg "Original Test1"
[image3]: ./output_images/tests/camera/undistort/undistorted_test3.jpg "undistorted Image 3"
[image4]: ./test_images/test3.jpg "Original Test3"
[image5]: ./output_images/tests/threshold/binary/edge_neg_test1.jpg "edge_neg_test1.jpg" 
[image6]: ./output_images/tests/threshold/binary/edge_pos_test1.jpg "edge_pos_test1.jpg"
[image7]: ./output_images/tests/threshold/binary/white_loose_test1.jpg "white_loose_test1.jpg" 
[image8]: ./output_images/tests/threshold/binary/white_tight_test1.jpg "white_tight_test1.jpg"
[image9]: ./output_images/Calibration.jpg "calibration_undistored"

[image10]: ./output_images/tests/transformation/birdseye/get_trans/debug_trans_undistored_original_image.jpg  "Original Image with lines"
[image11]: ./output_images/tests/transformation/birdseye/get_trans/debug_trans_undistored_warp_image.jpg "Birds eye with lines"
[image12]: ./output_images/tests/threshold/binary/yellow_edge_neg_test1.jpg "yellow_edge_neg_test1.jpg"
[image13]: ./output_images/tests/threshold/binary/yellow_edge_pos_test1.jpg "yellow_edge_pos_test1.jpg"
[image14]: ./output_images/tests/threshold/binary/yellow_test1.jpg "yellow_test1.jpg"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

My code in located in the Advanced-Lane-Lines folder it is a visual studio solution. The code for camera_calibration and image undistortion is located in the camera folder and is a method in the ImageProcessor class found in the image_processor.py file.
I use a pretty stand way to calibrate my camera. One thing I do is if I have claibrate the camrea before I just check if the calibration data is there and I load this data if not I recalibrate and save the calibration data. 

My calibration code is added below. 

``` python
def save_to_file(self, mtx, dist, cal_file_name):
         dist_pickle = {}
         dist_pickle["mtx"] = mtx
         dist_pickle["dist"] = dist
         pickle.dump( dist_pickle, open( cal_file_name, "wb" ) )

def calibrate_camera (self, nx=9, ny=6, re_cal = False):
        """
        Returns the camera matrix, distortion coefficients only. The rotation and translation vectors 
        are calculated but not returned. The first time this is run it it will calcutate the values 
        and save it to a file. The next time it is run with re_cal = false it will look for the file
        and return the values from the previous calibation.
        
        To re_calibrate even if a file exists use re_cal = True
        """
        #The file where the calibration data us persisted       
        cal_file_name = "data/wide_dist_pickle.p"
        
        if (os.path.exists(cal_file_name) and re_cal == False):
            # file exists
            with open(cal_file_name, mode='rb') as f:
                calibration_data = pickle.load(f)
        
                self.mtx, self.dist = calibration_data["mtx"], calibration_data["dist"]
        
        else:
            # Criteria for termination of the iterative process of corner refinement.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.1)
        
            #Prepare known cordinates for a chess board with 9x6 object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((ny*nx,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d points in real world space
            imgpoints = [] # 2d points in image plane.
        
            # Make a list of calibration images
            images = glob.glob('camera_cal/*.jpg')
        
            # Step through the list and search for chessboard corners
            for idx, fname in enumerate(images):
                img = mpimg.imread(fname, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
                # If found, add object points, image points
                if (ret == True):
                    objpoints.append(objp)
                    #Once we find the corners, we can increase their accuracy using this code below.
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)
            img_size = (img.shape[1], img.shape[0])
            # Do camera calibration given object points and image points
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
            # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
            self.save_to_file(self.mtx, self.dist, cal_file_name)
```
Once my camera is calibrate I then get the transforamtion matrix data that I will later use to change the image perspective. I'll provide more details about this in later sections. 


I use the distortion matix and coefficents to undistored the image.

The code I used to undistored images is listed below:

```python 
	   
    def undistort_image(self, img): 
        """
        Use this method to to undistored images 
        """
        #img_size = (img.shape[1], img.shape[0])
        
        # undistort the image
        
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
```
Here are a few examples: 
![alt text][image9]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
##### Original Image
![alt text][image2]

How i perfomr distortion correction is described in section 1. However I describe it briefly here. 
1. I calauclate the distortion matrix and coeffecients retured by the cv2.calibrateCamera function. 
1. I use these numbers to correct the image distortion.

Here is an example of the same image after correcting for image distortion:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
In order to decided which transforms I should combine in my final binary image, I ran a numebr of tests using the follwoing methods and inspected the output of each:

1. Absolute Threashold
1. Binary Threashold
1. Magnitude Thershold 
1. Direction Thershold 
1. S-Chennel Threshold
1. L-Chennel Threshold
1. H-Chennel Threshold
1. LAB B-Channel Threshold

What if found was the following:
1. In-order to find just the lane lines and ignore all the noise in the image i had to split the image into 7 color channels. 
1.1. 4 channels where obtained using sobel gradient along x-direction. I kept the poisitve and negative signs.
1.1. The S gradient allows us to find the yellow egdes
1.1. The V gradient allows us to find the white and yellow edges.
1. The other channels are obtain buy appying a color threshold.
1.1. H color threshold allows us to pick yellow 
1.1. V threshold for picking the white color

In the section below i will walk through each of the 7 channels to show why they were finally slected to bet the thresholded image:

##### Thresholding 
I performed soble x on the V & S channels and this here is the original image and the threshold output for follow:
###### V-Channel - white and yellow lines (Positive)
![alt text][image1] ![alt text][image6]
###### V-Channel - white and yellow lines (Negative)
![alt text][image1] ![alt text][image5]
###### S-Channel - yellow lines(Positive)
![alt text][image1] ![alt text][image13]
###### S-Channel - yellow lines(Negative)
![alt text][image1] ![alt text][image12]

The thresholded images provide a nice view of the line in the x-direcction even though there is still noise in the image. Later i can deal with the good line in the picuture bu applying a region of interest mask

###### V-Channel - white (tight) white lines that is between 175 and 255 + s-channel + 20
![alt text][image1] ![alt text][image8]
###### V-Channel - - white lines (loose) white lines that is between 175 and 255 + s-channel 
![alt text][image1] ![alt text][image7]
###### H-Channel - yellow lines that is between 175 and 255
![alt text][image1] ![alt text][image14]

Since i look at each image with the following 7 leyers you cna see fom the oupt visuals that i have 7 layers of information to determine if the line I'm loking at is a shite or yellow line in my region of interest

Here is the code that gets the binary image
```python
def split_channels(img) :
        """
        returns a total of 7 channels : 
        4 edge channels : all color edges (including the signs), yellow edges (including the signs) 
        3 color channels : yellow and white (2 different thresholds are used for white) 
        """
        binary = {}  
        
        # thresholding parameters for various color channels and Sobel x-gradients
        h_thresh=(15, 35)
        s_thresh=(75, 255) 
        v_thresh=(175,255)
        vx_thresh = (20, 120)
        sx_thresh=(10, 100)

        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        # Sobel x for v-channel
        sobelx_pos = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        sobelx_neg = np.copy(sobelx_pos)
        sobelx_pos[sobelx_pos<=0] = 0
        sobelx_neg[sobelx_neg>0] = 0
        sobelx_neg = np.absolute(sobelx_neg)
        scaled_sobel_pos = np.uint8(255*sobelx_pos/np.max(sobelx_pos))
        scaled_sobel_neg = np.uint8(255*sobelx_neg/np.max(sobelx_neg))
        vxbinary_pos = np.zeros_like(v_channel)
        vxbinary_pos[(scaled_sobel_pos >= vx_thresh[0]) & (scaled_sobel_pos <= vx_thresh[1])] = 1
        binary['edge_pos'] = vxbinary_pos
        vxbinary_neg = np.zeros_like(v_channel)
        vxbinary_neg[(scaled_sobel_neg >= vx_thresh[0]) & (scaled_sobel_neg <= vx_thresh[1])] = 1
        binary['edge_neg'] = vxbinary_neg

        # Sobel x for s-channel
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))
        sxbinary_pos = np.zeros_like(s_channel)
        sxbinary_neg = np.zeros_like(s_channel)
        sxbinary_pos[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) 
                     & (scaled_sobel_pos >= vx_thresh[0]-10) & (scaled_sobel_pos <= vx_thresh[1])]=1
        sxbinary_neg[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) 
                     & (scaled_sobel_neg >= vx_thresh[0]-10) & (scaled_sobel_neg <= vx_thresh[1])]=1           
        binary['yellow_edge_pos'] = sxbinary_pos
        binary['yellow_edge_neg'] = sxbinary_neg

        # color thresholds for selecting white lines
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= v_thresh[0]+s_channel+20) & (v_channel <= v_thresh[1])] = 1
        binary['white_tight'] = np.copy(v_binary)
        v_binary[v_channel >= v_thresh[0]+s_channel] = 1
        binary['white_loose'] = v_binary

        # color threshold for selecting yellow lines
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]) & (s_channel >= s_thresh[0])] = 1
        binary['yellow'] = h_binary

        return binary
```


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My ```python transformation.py ``` class have the prespective tranformation code.

WhenI create an instance of the Transformation class  i do the follwoing:
1. Calibrtate my camera so i can get my camera matrix and distortion coeffecients
1. Calculate the transformation and inverse matrix
Here is the class init code

``` python 
def __init__(self, cal, plot, image):
        self.calibrate_camera(9, 6, cal)
        #image = plt.imread(image)
        self.get_transformation_matrix(image, plot=plot)
```
1. Once i have the M and M_Inv vlaues calculated i can use the following code to chnage the prespective:
``` python 
def transform_perspective(self, image):
        x = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return x

    def inverse_transform_perspective(self, image):
        x = cv2.warpPerspective(image, self.M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return x
```

Here is an example of an image that has been transformedto a birdseye view:
![alt text][image10] ![alt text][image11]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

My ```python LaneLines.py ``` class have the the lane detection code. I'll describe it here:

One key part of my approach to finding lane line is to look at the right left side of the image individually. i do this as there can be instances where i only detect an image on one side of the image. if this happens i know which side i found the line and then i can use means to polt points for the side where i didn't find the line.



![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
