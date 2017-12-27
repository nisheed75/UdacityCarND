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
[image15]: ./output_images/tests/lane_lines/detect/final_file1.jpg "Final output"
[image16]: ./output_images/tests/lane_lines/detect/fimg_A_file1.jpg "img_A_file1.jpg"
[image17]: ./output_images/tests/lane_lines/detect/img_B_file1.jpg "img_B_file1.jpg"
[image18]: ./output_images/tests/lane_lines/detect/img_C_file1.jpg "img_C_file1.jpg"
[image19]: ./output_images/test1.jpg "test1.jpg"
[image20]: ./output_images/test2.jpg "test2.jpg"
[image21]: ./output_images/test3.jpg "test3.jpg"
[image22]: ./output_images/test4.jpg "test4.jpg"
[image23]: ./output_images/test5.jpg "test5.jpg"
[image24]: ./output_images/test6.jpg "test6.jpg"
[image25]: ./output_images/test7.jpg "test7.jpg"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

My code in located in the Advanced-Lane-Lines folder it is a visual studio solution. The code for camera_calibration and image undistortion is located in the camera folder and is a method in the ImageProcessor class found in the image_processor.py file.
I use a pretty stand way to calibrate my camera. One thing I do is if I have calibrated the camera before I just check if the calibration data is there and I load this data if not I recalibrate and save the calibration data. 

My calibration code below. 

``` python
def save_to_file(self, mtx, dist, cal_file_name):
         dist_pickle = {}
         dist_pickle["mtx"] = mtx
         dist_pickle["dist"] = dist
         pickle.dump( dist_pickle, open( cal_file_name, "wb" ) )

def calibrate_camera (self, nx=9, ny=6, re_cal = False):
        """
        Returns the camera matrix, distortion coefficients only. The rotation and translation vectors 
        are calculated but not returned. The first time this is run it it will calculate the values and save it to a file. The next time it is run with re_cal = false it will look for the file and return the values from the previous calibration.
        
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


I use the distortion matrix and coefficients to undistorted the image.

The code I used to undistorted images is listed below:

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

How I perform distortion correction is described in section 1. However, I describe it briefly here. 
1. I calculate the distortion matrix and coefficients returned by the cv2.calibrateCamera function. 
1. I use these numbers to correct the image distortion.

Here is an example of the same image after correcting for image distortion:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
To decide which transforms I should combine in my final binary image, I ran some tests using the following methods and inspected the output of each:

1. Absolute Threshold
1. Binary Threshold
1. Magnitude Threshold 
1. Direction Threshold 
1. S-Channel Threshold
1. L-Channel Threshold
1. H-Channel Threshold
1. LAB B-Channel Threshold

What if found was the following:
1. In-order to find just the lane lines and ignore all the noise in the image I had to split the image into seven color channels. 
1.1. 4 channels were obtained using Sobel gradient along the x-direction. I kept the positive and negative signs.
1.1. The S gradient allows us to find the yellow edges
1.1. The V gradient allows us to find the white and yellow edges.
1. The other channels are obtained by applying a color threshold.
1.1. H color threshold allows us to pick yellow 
1.1. V threshold for picking the white color

In the section below I will walk through each of the 7 channels to show why they were finally slected to bet the thresholded image:

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

The thresholded images provide a nice view of the line in the x-direction even though there is still noise in the image. Later I can deal with the good line in the picture bu applying a region of interest mask

###### V-Channel - white (tight) white lines that is between 175 and 255 + s-channel + 20
![alt text][image1] ![alt text][image8]
###### V-Channel - - white lines (loose) white lines that is between 175 and 255 + s-channel 
![alt text][image1] ![alt text][image7]
###### H-Channel - yellow lines that is between 175 and 255
![alt text][image1] ![alt text][image14]

Since I look at each image with the following 7 layers, you can see from the ouput visuals that I have 7 layers of information to determine if the line I'm looking at is a shite or yellow line in my region of interest

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


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provided an example of a transformed image.

My ```python transformation.py ``` class have the perspective transformation code.

WhenI create an instance of the Transformation class  I do the following:
1. Calibrate my camera so I can get my camera matrix and distortion coefficients
1. Calculate the transformation and inverse matrix
Here is the class init code

``` python 
def __init__(self, cal, plot, image):
        self.calibrate_camera(9, 6, cal)
        #image = plt.imread(image)
        self.get_transformation_matrix(image, plot=plot)
```
1. Once I have the M and M_Inv vlaues calculated I can use the following code to chnage the prespective:
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

My ```python LaneLines.py ``` class have the lane detection code. I'll describe it here:

One key part of my approach to finding lane line is to look at the right left side of the image individually. I do this as there can be instances where I only detect an image on one side of the image. If this happens, I know which side I found the line and then I can use means to plot points for the side where I didn't find the line.

To find lane line, I use the sliding window approach where I take a frame and break it up into 15 windows. Here is an example of how my code does this:
1. I use a Line object to keep track of the lines I have detected and the x and y values of the line detected, etc.
1. Taking the image I first split the image in half, and I take a window that is 48 pixels high and start looking at the 7 channels that I have and find the line I'm interested in.
![alt text][image16] ![alt text][image17]
1. If I found lines the next window is going to be tighter around the place I found the last line. As you see in the about image, the second window in the image on the left is a small centered around the line. 
1. if I don't find a line I expand the window and i search a larger window for lines I'm interested in.
1. using point I found I can plot the best fit line. See image below.

![alt text][image18]
1. the last image shows the result after I finished analysis all the windows in the image.
![alt text][image15]

Here is the main code that searches for the lines in the image
``` python 
def find_lines_in_windows(self, image, nb_windows=15, visualize=True, debug=False):
         # get channels and warp them
        
        self.binary = Thresholding.split_channels(image)
        self.binary = {k: self.transformation.transform_perspective(v) for k, v in self.binary.items()}

        # group A consists of all line edges and white color 
        group_A = np.dstack((self.binary['edge_pos'], self.binary['edge_neg'], self.binary['white_loose']))
        # group B consists of yellow edges and yellow color
        group_B = np.dstack((self.binary['yellow_edge_pos'], self.binary['yellow_edge_neg'], self.binary['yellow']))
        
        if visualize :
            out_img_A = np.copy(group_A)*255
            out_img_B = np.copy(group_B)*255
            out_img_C = np.zeros_like(out_img_A)
        

        
        #Set the number of windows and and the width & height for each window
        height, width = group_A.shape[:2]
        num_windows = nb_windows
        num_rows = height
        self.dims = (width,height)

        window_height = np.int(height / num_windows)
        window_width = 50
        
        midpoint = np.int(width/2)
        # window with +/- margin
        self.margin = {'left' : np.int(0.5*midpoint), 
                       'right': np.int(0.5*midpoint)}
        self.min_pixels = 100
        self.lane_gap = 600

        # center of current left and right windows
        self.x_current = {'left' : np.int(0.5*midpoint), 
                          'right': np.int(1.5*midpoint)}
        # center of left and right windows last found
        self.x_last_found = {'left' : np.int(0.5*midpoint), 
                             'right': np.int(1.5*midpoint)}
        # center of previous left and right windows
        x_prev = {'left' : None, 
                  'right': None}
        
        momentum    = {'left' :0, 
                       'right':0}
        last_update = {'left' :0, 
                       'right':0}
        self.found  = {'left' :False, 
                       'right':False}
        
        self.nonzero_x, self.nonzero_y = self.get_nonzero_pixels() 
         # good pixels
        self.good_pixels_x = {'left' : [], 'right' : []} 
        self.good_pixels_y = {'left' : [], 'right' : []} 
        
        # Step through the windows one by one
        for window in range(num_windows):
                
            # final window refinement with updated centers and margins
            Y1 = height - (window+1)*window_height
            Y2 = height - window*window_height
            X1 = {side : self.x_current[side]-self.margin[side] for side in ['left','right']} 
            X2 = {side : self.x_current[side]+self.margin[side] for side in ['left','right']} 
            if debug :
                print("-----",window, X1, X2, Y1, Y2)
               
            found, good_pixels_x, good_pixels_y = self.window_analysis(X1,Y1,X2,Y2)
            if not self.check_lanes(min_lane_gap=350, img_range=(-50,width+50)) : 
                break
                 
            for i,side in enumerate(['left','right']) :
                # Add good pixels to list
                if found[side] :
                    self.good_pixels_x[side].append(good_pixels_x[side])
                    self.good_pixels_y[side].append(good_pixels_y[side])
                    self.x_last_found[side] = self.x_current[side]
                
                # Draw the windows on the visualization image
                if visualize :
                    cv2.rectangle(out_img_A,(X1[side],Y1) ,(X2[side],Y2) ,(0,255,i*255), 2)  
                    cv2.rectangle(out_img_B,(X1[side],Y1) ,(X2[side],Y2) ,(0,255,i*255), 2) 
                    # Draw good pixels 
                    out_img_C[good_pixels_y[side], good_pixels_x[side],i] = 255 
        
        for side in ['left','right'] :
            if self.good_pixels_x[side] :
                self.found[side] = True
                self.good_pixels_x[side] = np.concatenate(self.good_pixels_x[side])
                self.good_pixels_y[side] = np.concatenate(self.good_pixels_y[side])
            else :
                self.good_pixels_x[side] = None
                self.good_pixels_y[side] = None
        if visualize :
            return out_img_A.astype(np.uint8), out_img_B.astype(np.uint8), out_img_C.astype(np.uint8)
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code to calculate the radius of curvature is in the ``` python Line.py ``` class I've added is below for your benefit:
it uses some key constants to turn the values into meters

``` python

 # Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def calc_R(self, fit) :
        y=img_dim[1]
        self.radius_of_curvature = ((ym_per_pix**2 + xm_per_pix**2*(2*fit[0]*y + fit[1])**2)**1.5)/(2
                                    *xm_per_pix*ym_per_pix*fit[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

My pipleine in the ``` python LaneLines.py``` class. Here is the code below:
``` python 
 def get_image_with_lanes(self, image,  line, fail, nb_windows=15, visualize=True, debug=False):
        global counter
        self.counter +=1
        self.line = line
        self.fail = fail
        self.img = image
        ym_per_pix = 30/720 # meters per pixel in y dimension

        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        height, width = image.shape[:2]
        img_dim = (width, height)

        img = self.transformation.undistort_image(image)
        
        
        imgA = np.zeros_like(img)
        imgB = np.zeros_like(img)
        imgC = np.zeros_like(img)
        main_img = np.zeros_like(img).astype(np.uint8) #blank image like img

        imgA,imgB,imgC = self.find_lines_in_windows(image, nb_windows, visualize, debug) # find the lines using a window search 
        
        fit = {'left':None, 'right':None}    
        sides = ['left','right']
    
        for side in sides :
            if not self.found[side] :
                self.fail[side]+=1
                self.line[side].detected=False
            else :
                pixels_x, pixels_y = self.good_pixels_x, self.good_pixels_y
                self.line[side].add_line(pixels_x[side], pixels_y[side])
                self.line[side].detected=True
            
        if self.line['left'].check_diverging_curves(self.line['right']) or self.line['left'].fit_ratio(self.line['right'])>10 \
                or (not 400*xm_per_pix<self.line['left'].base_gap(self.line['right'])<750*xm_per_pix) :
        
            for side in sides :
                if self.line[side].delta_xfitted() > 1000 or self.line[side].res > 55: 
                    self.fail[side] += 1
                else :
                    self.line[side].update()
        else :
            for side in sides :
                if self.line[side].res > 55  :
                    self.fail[side] +=1
                elif self.line[side].detected : 
                    self.fail[side]=0
                    self.line[side].update()
            
        for side in sides :  
            fit[side] = self.line[side].avg_fit
            pts = np.array(np.vstack((self.line[side].avg_xfitted, self.line[side].yfitted)).T, dtype=np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(imgC,[pts],False,(255,255,0), thickness=5)
            
            pts = np.array(np.vstack((self.line[side].current_xfitted, self.line[side].yfitted)).T, dtype=np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(imgC,[pts],False,(0,255,255), thickness=2)
            self.line[side].calc_R(self.line[side].avg_fit)
            self.line[side].calc_base_dist(self.line[side].avg_fit)

        R_avg = (self.line['left'].radius_of_curvature + self.line['right'].radius_of_curvature)/2
        base_gap = (self.line['left'].base_gap(self.line['right']))
        center_pos = (self.line['left'].line_base_pos + self.line['right'].line_base_pos)/2

        main_img = self.plot_lane(fit) #This is where we plot the lane lines on the image
        filename = "file{}.jpg".format(self.counter)
        if debug:         
            img_A = cv2.resize(imgA,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
            hA,wA = img_A.shape[:2]
            img_B = cv2.resize(imgB,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
            hB,wB = img_B.shape[:2]
            img_C = cv2.resize(imgC,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
            text_A = np.zeros((int(hA/4), wA,3))
            h_text = text_A.shape[0]    
            text_B = np.zeros((h_text, wA,3))
            text_C = np.zeros((h_text, wA,3))
        else:   
            text_A = np.zeros((int(height/4), width,3))
            h_text = text_A.shape[0] 
            text_B = np.zeros((h_text, width,3))
            text_C = np.zeros((h_text, width,3))
        
        text_A = text_A.astype(np.uint8)
        text_B = text_B.astype(np.uint8)
        text_C = text_C.astype(np.uint8)

        for i in range(1,3) :
            text_A[:,:,i] = 255
            text_B[:,:,i] = 255
            text_C[:,:,i] = 255
        
        
        font = cv2.FM_8POINT
        cv2.putText(text_A,'Threshold',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(text_B,'Threshold (yellow)',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(text_C,'Best fit',(10,h_text-20), font,1,(0,0,255),3,cv2.LINE_AA)
        
        if debug: 
            img_combined_right = np.vstack((text_A, img_A, text_B, img_B, text_C, img_C))
            main_text = np.zeros((3*h_text+3*hA-height,width,3)).astype(np.uint8)
        else:
            main_text = np.zeros((int(0.8*h_text),width,3)).astype(np.uint8)

        h_main_text, w_main_text = main_text.shape[:2]
        cv2.putText(main_text,'Radius of curvature : {:5.2f} m'.format(abs(R_avg)),
                    (10,35), font, 1,(255,255,255),3,cv2.LINE_AA)
        shift = "left" if center_pos>0 else "right"
        cv2.putText(main_text,'Vehicle is {:6.2f} m {:5} of center'.format(abs(center_pos), shift),
                    (10,80), font, 1,(255,255,255),3,cv2.LINE_AA)
        if self.line['left'].avg_fit[0]>0.0001 and  self.line['right'].avg_fit[0]>0.0001 :
            cv2.putText(main_text,'Right curve ahead',
                    (10,135), font, 1,(100,100,25),3,cv2.LINE_AA)
        elif self.line['left'].avg_fit[0]<-0.0001 and  self.line['right'].avg_fit[0]<-0.0001 :
            cv2.putText(main_text,'Left curve ahead',
                    (10,135), font, 1,(100,100,25),3,cv2.LINE_AA)
        img_combined_left = np.vstack((main_img, main_text))
        
        
        if debug:
            final = np.hstack((img_combined_left, img_combined_right))
            self.plot_image("img_A", filename, img_A)
            self.plot_image("img_B", filename, img_B)
            self.plot_image("img_C", filename, img_C)
        else: 
            final = img_combined_left
        if visualize:
            self.plot_image("main_img", filename, main_img)
            self.plot_image("final", filename, final)
 
        return final, self.line, self.fail
```

#### My final output images for the 7 test images provided

The follwoing images shows ho my pipeline was able to plot the lanelines on the test imaages
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was a challenging project. I took about 4 to 5 attempts at finding a solution to plot these lane lines. The challenges I faced were in the following areas:  
1. Initially, I tried to clean up the images and completely get rid of the noise. This just ended up also getting rid of faint white line markings and thus i had many images where I only had the yellow lines.
1. Shadows caused havoc and i couldn't find  a solution that addresses shadows well until I stumbled across a fellow students project that used the 7 channel approach I finally borrowed. Ak to preritj
1. His approach handled this by " The shadows, and irrelevant road markings usually have either positive gradients or negative gradient edges but not both, unlike the lane markings which have both."

There are areas that I'm currently working on to improve my project namely:
1. Improve the computation efficiency but taking the approach where I can use the best fit line to predict images for some frames instead of going through each of the 15 windows especially if the fit window finds that the line is in similar location to the previously found location. This will save computation cycles and will still be accurate.  
1. Improve my curve checking function.
1. implement convolution to select my hot pixels this why I can remove outliers.
1. Explore thresholding techniques to better refine my channels as this will make line detection better. 
