Kara Pelster

CISC442 

Programming Assignment 1

# Setup

numpy, matplotlib, opencv-contrib-python, and scikit-image are needed to run this program

# Run Program

simply run the program as you would normally with your python interpreter

# Part A

Once you run the program, a series of images will open in new windows on the screen:
1. Convolved image with a kernel
2. Reduced image
3. Expanded image
4. Gaussian pyramid
5. Laplacian pyramid
6. Reconstructed image

Exit out of the image to proceed to the next image

# Part A Image Blending

**Non-Unwarped Mosaic**

4 sets of images will display to the console one set at a time to begin non-unwarped mosaicing. 

    a. First, the two images will appear together so that you can view overlapping images. Exit out of that window to select points to begin mosaic. 
    b. The right-most image will display, and you should select two points that are stacked vertically essentially to create a line between the points. The region that you select the points should be the region that overlaps with the left most image.
    c. Press any key to exit, and the mosaic will appear. Exit out of the image to move on to the next set of images.

**Perspective Unwarped Mosaic**

4 sets of images will display to the console one set at a time to begin perspective unwarped mosaicing. 

    a. First, the two images will appear together so that you can view overlapping images. Exit out of that window to select points to begin mosaic. 
    b. The left-most image will display, and you should select exactly four points that correspond to points on the right-most image. Press any key to exit. 
    c. Next, the right-most image will display, and you should select the exact same corresponding points that you chose for the left image. The more precise the better.

The perspective transform is sensitive, so small variations in correspondance result in a wonky transform.
Press any key to exit. A perspective transform will be applied to the right image, and the mosaic will appear. Exit out of the image to move on to the next set of images.

**Affine Unwarped Mosaic**

4 sets of images will display to the console one set at a time to begin affine unwarped mosaicing. 

    a. First, the two images will appear together so that you can view overlapping images. Exit out of that window to select points to begin mosaic. 
    b. The left-most image will display, and you should select exactly three points that correspond to points on the right-most image. Press any key to exit. 
    c. Next, the right-most image will display, and you should select the exact same corresponding points that you chose for the left image. The more precise the better. Press any key to exit. 

An affine transform will be applied to the right image, and the mosaic will appear. 
Exit out of the image to move on to the next set of images.

# Part B

After viewing and exiting out of the 6 images, a new set of images will begin running that require user interaction.

**1. Affine Transformation.**

   The way that this affine transformation works is that first, a non-transformed image appears, and you select three points. 
   Next, an image with a known affine transformation appears, and you must select exactly the same corresponding points as the 
   first non-transformed image. You must select exactly three points for both images. The method MyAffine() uses these two sets of points to calculate the affine transformation matrix and compares
   the resulting affine matrix to the calculated built-in affine matrix from cv2.getAffineTranform(). The error between the two
   calculations is displayed in the console.
    
    a. A star image will appear. Click on three points on the image and then press any key to exit from the image.
    b. Next an affine transformed image will appear. Click on exactly the same three points that correspond to the first image.
        Press any key to exit.
   
**2. Four Point Perspective Transformation.**

   The way that this perspective transformation works is that first, a non-transformed image appears, and you select four points. 
   Next, an image with a known perspective transformation appears, and you must select exactly the same corresponding points as the 
   first non-transformed image. You must select exactly four points for both images. The more precisely you correspond points between images, the better 
   the transformation is. The perspective transformation algorithm is sensitive. The method getPerspectiveTranform() uses these two sets of points to calculate the perspective transformation matrix and compares
   the resulting perspective matrix to the calculated built-in perspective matrix from cv2.getPerspectiveTranform(). The error between the two
   calculations is displayed in the console.

    a. A star image will appear. Click on four points on the image and then press any key to exit from the image.
    b. Next a perspective transformed image will appear. Click on exactly the same four points that correspond to the first image.
        Press any key to exit.
   
**2. Eight Point Overconstrained Perspective Transformation.**

   The way that this perspective transformation works is that first, a non-transformed image appears, and you select four points. 
   Next, an image with a known perspective transformation appears, and you must select exactly the same corresponding points as the 
   first non-transformed image. You must select exactly eight points for both images.  The more precisely you correspond points between images, the better 
   the transformation is. The perspective transformation algorithm is sensitive.The method getPerspectiveTranform() uses these two sets of points to calculate the perspective transformation matrix and compares
   the resulting perspective matrix to the calculated built-in perspective matrix from cv2.getPerspectiveTranform(). The error between the two
   calculations is displayed in the console.
    
    a. A star image will appear. Click on eight points on the image and then press any key to exit from the image.
    b. Next a perspective transformed image will appear. Click on exactly the same eight points that correspond to the first image.
        Press any key to exit.

