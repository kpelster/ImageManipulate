"""
Kara Pelster
CISC442
Programming Assignment 1

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
import sys
from mpmath import mp

def click_event(event, x, y, flags, params):
	"""
	click event applied to open cv image. when user clicks on image, coordinates of points added to an array of points
	:param: default parameters, never defined when calling click_event
	:return: returns array of points
	"""
	# global variable points that is used for all point selecting
	global points
	font = cv2.FONT_HERSHEY_PLAIN
	if event == cv2.EVENT_LBUTTONDOWN:
		points.append([x,y])
		# for all point selecting, img and 'image' is used. must be this way or click event does not work
		cv2.putText(img, "+", (x, y), font, 1, (100, 100, 0), 3)
		# keep showing image
		cv2.imshow('image', img)
	return points

def getting_image_points(img, points):
	"""
	:param img: image to get coordinates from
	:param points: array to add points to that is used in click_event
	:return:
	"""
	cv2.imshow('image', img)
	cv2.setMouseCallback('image', click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return points

def Convolve(image, kernel):
	"""
	A1.
	Given an image and a kernel, convolve image with kernel
	Image can be grayscale or RGB

	:param image: image to convolve
	:param kernel: kernel to convolve with. a matrix
	:return: returns resulting convolved image
	"""

	# parameter dimensions
	# image
	(imH, imW) = image.shape[:2]
	# kernel
	(knH, knW) = kernel.shape[:2]

	# padding for image border
	pad = (knW-1)//2


	# add border to image
	# https: // www.geeksforgeeks.org / python - opencv - cv2 - copymakeborder - method /
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

	#empty matrix to store convolved image
	result = np.zeros((imH, imH, 3), dtype="float32")
	# print(result.shape)

	# iterate over image and apply kernel
	for column in np.arange(pad, imH + pad):
		for row in np.arange(pad, imW + pad):
			for channel in range(3):

				# print(c, r, channel)
				# roi = region of interest; centered region about coordinates
				roi = image[column - pad:column + pad + 1, row - pad:row + pad + 1, channel]

				# convolution
				convxy = (roi * kernel).sum()

				# put convolved value in result
				result[column - pad, row - pad, channel] = convxy

	# scale new image to be in typical rgb range
	result = rescale_intensity(result, in_range=(0,255))
	result = (result*255).astype("uint8")
	return result

def smoothed_image(image):
	"""
	A2.
	Given an image, apply gaussian filter and reduce its width and height by half
	:param image: image to apply gaussian filter to
	:return: return gaussian smoothed image
	"""
	# define Gaussian filter
	gaussian_filter = np.array(
		[[1, 2, 1],
		 [2, 4, 2],
		 [1, 2, 1]]
	) / 16.0
	# gaussian kernel from notes

	# apply convolution from 1 where g is the kernel
	smoothed_im = Convolve(image, gaussian_filter)

	return smoothed_im

def resize_image(image, factor):
	"""
	A. given an image and scaling factor, resize image. assumed to have already been smoothed if needed

	:param image: image to be resized
	:param factor: factor to scale image with
	:return: scaled image
	"""

	# image = smoothed_image(image)

	# new image parameters
	(imH, imW) = image.shape[:2]
	(newimH, newimW) = (int(imH*factor), int(imW*factor))

	# define new matrix to put new value in
	new_image = np.zeros((newimH, newimW, 3), dtype = "float32")

	# reduce by half
	for column in range(newimW):
		for row in range(newimH):
			new_image[column][row] = image[int(column/factor)][int(row/factor)]

	# scale new image to be in typical rgb range
	new_image = rescale_intensity(new_image, in_range=(0, 255))
	new_image = (new_image * 255).astype("uint8")

	return new_image

def Reduce(image):
	"""
	A2. Reduce image by one half. Smooths image with gaussian kernel first
	uses resize_image(image) to scale
	:param image: image to be reduced
	:return: reduced image
	"""
	#  applies gaussian smoothing kernel to image before reducing
	return resize_image(smoothed_image(image), 0.5)

def Expand(image):
	"""
	A3. Expand image by factor of 2.
	uses resize_image(image) to scale
	:param image: image to be expanded
	:return: expanded image
	"""
	return resize_image(image, 2)

def GaussianPyramid(image, num_levels):
	"""
	A4. Creates a Gaussian pyramid with a specified number of levels. Applies gaussian smoothing first
	:param image: image to create pyramid with
	:param num_levels: number of levels of pyramid
	:return: gaussian pyramid with specified number of levels
	"""
	# smooth image first with gaussian pyramid
	image = smoothed_image(image)
	# initialize pyramid
	pyramid = [image]
	for level in range(1,num_levels+1):
		# for each level reduce last level
		im = Reduce(pyramid[level-1])
		# add  image to pyramid
		pyramid.append(im)
	return pyramid

def LaplacianPyramids(image, gaussian_pyramid, num_levels):
	"""
	A5. Creates Laplacian pyramid with a specified number of levels.
	:param image: image to create gaussian pyramid
	:param gaussian_pyramid: a gaussian pyramid that has already been applied to the image
	:param num_levels: number of levels of pyramid
	:return: laplacian pyramid with specified number of levels
	"""
	#gaussian = smoothed_image(image)
	# initialize pyramid
	pyramid = [gaussian_pyramid[num_levels]]
	for level in reversed(range(num_levels)):
		(imH, imW) = gaussian_pyramid[level].shape[:2]
		# expand gaussian pyramid level
		expanded_image = Expand(gaussian_pyramid[level+1])
		# get new dimensions
		(newimH, newimW) = expanded_image.shape[:2]
		expanded_image = cv2.copyMakeBorder(expanded_image, 0, imH - newimH, 0, imW - newimW, cv2.BORDER_REPLICATE)
		# take difference of gaussian image and expanded laplacian and add to pyamid
		pyramid.append(gaussian_pyramid[level] - expanded_image)
	return pyramid

def Reconstruct(laplacian_pyramid, num_levels):
	"""
	A6. Collapses laplacian pyramid to generate original image
	:param laplacian_pyramid: laplacian pyramid to collapse
	:param num_levels: number of levels in pyramid
	:return: reconstructed image
	"""
	# initialize matrix to add levels to
	reconstructed_matrix = None
	# iterate through levels in input laplacian pyramid
	for level in range(len(laplacian_pyramid)):
		if level == 0:
			# first level of reconstructed is first level of laplacian
			reconstructed_matrix = laplacian_pyramid[level]
		else:
			# here is where we start adding back pyramids
			(imH, imW) = laplacian_pyramid[level].shape[:2]
			# double size of matrix using Expand
			expanded_image = Expand(reconstructed_matrix)
			# get new height and width
			(newimH, newimW) = expanded_image.shape[:2]
			expanded_image = cv2.copyMakeBorder(expanded_image, 0, imH - newimH, 0, imW - newimW, cv2.BORDER_REPLICATE)
			reconstructed_matrix = expanded_image + laplacian_pyramid[level]
	return reconstructed_matrix

def MyAffine(source, destination):
	"""
	B2. Calculates affine transformation matrix given a set of corresponding points
	:param source: points from first image, source points
	:param destination: points from second image, destination points
	:returns: 2x3 transformation matrix
	"""

	# creates matrices from source and destination points
	source_matrix = np.matrix(np.array([
		[ source[0][0], source[1][0], source[2][0] ],
		[ source[0][1], source[1][1], source[2][1] ],
		[ 1, 1, 1 ]
	]))
	destination_matrix = np.matrix(np.array([
		[ destination[0][0], destination[1][0], destination[2][0] ],
		[ destination[0][1], destination[1][1], destination[2][1] ],
		[ 1, 1, 1 ]
	]))

	# solve
	affine_matrix = destination_matrix * np.linalg.inv(source_matrix)
	# remove last row since it is just [0,0,1]
	affine_matrix_new = np.delete(affine_matrix, 2, axis = 0)

	return affine_matrix_new

def getPerspectiveTransform(source, destination, numpoints):

	"""
	B2. Calculates perspective transformation matrix given a set of corresponding points and number of points
	:param source: points from first image, source points
	:param destination: points from second image, destination points
	:param numpoints: number of points in source and destination
	:return: perspective transformation matrix
	"""

	# given number of points, make a matrix with dimensions 2nx9 of zeros. values will be filled in
	# 2nx9 matrix found in professors notes on homography
	homography_matrix = np.zeros((2*numpoints, 9))
	# iterate to make matrix based on number of points
	# matrix found in professor's notes
	row = 0 # row count for adding row to matrix
	for point in range (0, numpoints): #iterate through points in source and destination
			# adds two rows at a time
			homography_matrix[row ] = [source[point][0], source[point][1], 1, 0, 0, 0, -source[point][0]*destination[point][1], -source[point][1]*destination[point][0], -destination[point][0]]
			homography_matrix[row +1] = [0, 0, 0, source[point][0], source[point][1], 1, -source[point][0]*destination[point][1], -source[point][1]*destination[point][1], -destination[point][1]]
			row  = row  + 2

	#svd
	(u, s, v) = np.linalg.svd(homography_matrix)
	# extract v
	result = v[-1].reshape(3,3)
	return result

def HarrisDetector(img1, img2):
	"""
	Extra Credit. Detects edge points in images
	"""
	operatedImage1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	operatedImage2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	operatedImage1 = np.float32(operatedImage1)
	operatedImage2 = np.float32(operatedImage2)
	dest1 = cv2.cornerHarris(operatedImage1, 2, 5, 0.07)
	dest2 = cv2.cornerHarris(operatedImage2, 2, 5, 0.07)

	dest1 = np.uint8(cv2.dilate(dest1, None))
	dest2 = np.uint8(cv2.dilate(dest2, None))


	cv2.imshow('dest1', dest1)
	cv2.waitKey(0)
	# img1[dest1 > 0.01 * dest1.max()] = [0, 0, 255]
	# img2[dest2 > 0.01 * dest2.max()] = [0, 0, 255]
	print("cc of dest in hd")
	print(np.mean(np.multiply((dest1 - np.mean(dest1)), (dest2 - np.mean(dest2)))) / (np.std(dest1) * np.std(dest2)))

	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dest1)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dest2)

	return img1, img2

def CrossCorrelation(img1, img2):
	"""
	Extra credit. calculates correlation coefficient
	"""
	# Correlation coefficient
	print("cc of haris dect results")
	print(np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2)))


def Blend(img, img2, points, imcount):
	"""
	Part A. Creates mosaic of two images
	:param img: image to choose points from
	:param img2: image that has overlapping areas with img
	:param points: set of points that overlap between images. points are only from img
	:return: no return. displays image
	"""
	# points from first image
	plt.subplot(121), plt.imshow(img2[:, :, ::-1]), plt.title('Left Image')
	plt.subplot(122), plt.imshow(img[:, :, ::-1]), plt.title('Right Image\nExit to click points on right image')
	plt.show()

	# show right image for user to click points
	print("\nSelect two points with one vertically on top of the other that corresponds to a similar region on the other image")
	img_original = img.copy()
	points = getting_image_points(img, points)

	average_x = (points[0][0] + points[1][0]) / 2
	# create bit mask
	mask = np.full(img.shape, 255, dtype=np.uint8)

	# mask = np.zeros(shape = ())
	for row in range(mask.shape[0]):
		for column in range(0, int(average_x)):
				mask[row][column] = 0
				# break if out of range
				if column > average_x:
					break

	fig = plt.figure(figsize=(5, 5))
	plt.subplot(121), plt.imshow(mask[:, :, ::-1]), plt.title('Bit Mask')
	plt.show()
	fig.savefig('pr1_images/mask_'+str(imcount)+'_12.png')

	# apply bit mask to right image
	masked_img = cv2.bitwise_and(img_original, mask)
	# get rid of empty space
	cropped_img = masked_img[::, int(average_x) + 1:masked_img.shape[0]]
	# concatenate left image with bit masked right image
	fullimage = np.concatenate((img2, cropped_img), axis=1)
	fig2 = plt.figure(figsize=(5, 5))
	plt.subplot(121), plt.imshow(fullimage[:, :, ::-1]), plt.title('Mosaic No Unwarping')
	plt.show()
	fig2.savefig('pr1_images/image_'+str(imcount)+'_12.png')


if __name__ == "__main__":

	# --------------------------------------------
	# CONVOLVE IMAGE WITH KERNEL
	# --------------------------------------------
	print("--------------------------------------------\nCONVOLVE IMAGE WITH KERNEL\n--------------------------------------------\nExit window to continue.")

	# read original image
	original_im = cv2.imread('pr1_images/stairs.png')

	# convolve image
	# define kernel
	kernel = np.ones((3, 3)) / 9
	convolve = Convolve(original_im, kernel)

	# plot original and convolved images next to each other
	fig1 = plt.figure(figsize=(8, 8))
	plt.subplot(121), plt.imshow(original_im[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(convolve[:, :, ::-1]), plt.title('Convolved Image')
	plt.show()

	# --------------------------------------------
	# REDUCE IMAGE
	# --------------------------------------------
	print(
		"--------------------------------------------\nREDUCE IMAGE\n--------------------------------------------\nExit window to continue.")
	reduce = Reduce(original_im)
	fig2 = plt.figure(figsize=(8, 8))
	plt.subplot(121), plt.imshow(original_im[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(reduce[:, :, ::-1]), plt.title('Reduced Image (Note Scale)')
	plt.show()

	# --------------------------------------------
	# EXPAND IMAGE
	# --------------------------------------------
	print(
		"--------------------------------------------\nEXPAND IMAGE\n--------------------------------------------\nExit window to continue.")
	expand = Expand(original_im)
	fig3 = plt.figure(figsize=(8, 8))
	plt.subplot(121), plt.imshow(original_im[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(expand[:, :, ::-1]), plt.title('Expanded Image (Note Scale)')
	plt.show()

	# --------------------------------------------
	# GAUSSIAN PYRAMID
	# --------------------------------------------
	print(
		"--------------------------------------------\nGAUSSIAN PYRAMID\n--------------------------------------------\nExit window to continue.")
	gaussian_pyramid = GaussianPyramid(original_im, 4)
	o = 1
	fig = plt.figure(figsize=(10, 10))
	for i in gaussian_pyramid:
		plt.subplot(1, len(gaussian_pyramid), o), plt.imshow(i[:, :, ::-1]), plt.title('Gaussian ' + str(o))
		o += 1
	plt.show()

	# --------------------------------------------
	# LAPLACIAN PYRAMID
	# --------------------------------------------
	print(
		"--------------------------------------------\nLAPLACIAN PYRAMID\n--------------------------------------------\nExit window to continue.")
	laplacian_pyramid = LaplacianPyramids(original_im, gaussian_pyramid, 4)
	o = 1
	fig = plt.figure(figsize=(10, 10))
	for i in laplacian_pyramid:
		plt.subplot(1, len(gaussian_pyramid), o), plt.imshow(i[:, :, ::-1]), plt.title('Laplacian ' + str(o))
		o += 1
	plt.show()

	# --------------------------------------------
	# RECONSTRUCT
	# --------------------------------------------
	print(
		"--------------------------------------------\nRECONSTRUCT\n--------------------------------------------\nExit window to continue.")

	reconstruct = Reconstruct(laplacian_pyramid, 4)
	fig1 = plt.figure(figsize=(8, 8))
	plt.subplot(121), plt.imshow(original_im[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(reconstruct[:, :, ::-1]), plt.title('Reconstructed Image')
	plt.show()

	# --------------------------------------------
	# 3 POINT AFFINE TRANSFORMATION
	# --------------------------------------------
	print(
		"--------------------------------------------\n3 POINT AFFINE TRANSFORMATION\n--------------------------------------------\nExit window to continue.")
	# original image
	star_image = cv2.imread('pr1_images/star.png')

	# Select 3 Corresponding points on original image
	"""
	Select three points to be stored as source points to compute affine matrix.
	Next, a transformed image with a known affine transformation will display in which you select the corresponding points
	"""
	print("Click on three points. Press any key to continue.")
	points = []
	img = star_image.copy()
	# cv2.imshow('image', img)
	# cv2.setMouseCallback('image', click_event)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	points = getting_image_points(img, points)
	# set source points to resulting points from click_event
	source = (np.float32(points)).copy()

	# Select 3 Corresponding points on affine transformed image
	"""
	Select three points to be stored as destination points to compute affine matrix. Three points must match with
	original image to compute accurate transformation matrix
	"""
	print("\nClick on three corresponding points to original image. Press any key to continue.")
	affine_matrix = np.matrix([[1, -0.25, 0.25], [0, 1, 0]])  # bottom row [0, 0, 1] is omitted for cv2.warpAffine
	img = cv2.warpAffine(star_image, affine_matrix, (star_image.shape[1], star_image.shape[0]))
	# reset global points
	points = []
	points = getting_image_points(img, points)
	# set destination points to resulting points from click_event
	destination = np.float32(points).copy()

	# Compute built in affine transformation matrix with corresponding source/destination points
	print("Built-in Affine Transformation")
	affine_builtin = cv2.getAffineTransform(source, destination)
	print(affine_builtin) # print resulting affine matrix
	# Compute affine transformation matrix with corresponding source/destination points
	print("\nAffine Transformation")
	affine = MyAffine(source, destination)
	print(affine) # print resulting affine matrix
	# Compute error between builtin and my own function to compute Affine matrix
	print("\nAffine Transformation Error")
	affine_error = np.sum(np.abs(np.subtract(affine, affine_builtin, dtype=float)))
	print(affine_error)

	# Display affine transformed image
	"""
	Display affine transformed image using matrix calculated from MyAffine(image, kernel)
	"""
	print("\nDisplaying Original Image on Left and Affine Transformed Matrix on Right. Press any key to continue.")
	im_affine = cv2.warpAffine(star_image, affine, (star_image.shape[1], star_image.shape[0]))

	# Display original and transformed image
	plt.subplot(121), plt.imshow(star_image[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(im_affine[:, :, ::-1]), plt.title('Affine Transform')
	plt.show()

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# --------------------------------------------
	# 4 POINT PERSPECTIVE TRANSFORMATION
	# --------------------------------------------
	print(
		"--------------------------------------------\n4 POINT PERSPECTIVE TRANSFORMATION\n--------------------------------------------")
	# Select 4 Corresponding points on original image
	"""
	Select four points to be stored as source points to compute perspective matrix.
	Next, a transformed image with a known perspective transformation will display in which you select the corresponding points
	"""
	print("\nClick on four points. Press any key to contine.")
	points = []
	img = star_image.copy()
	points = getting_image_points(img, points)

	# set source points to resulting points from click_event
	source = (np.float32(points)).copy()

	# Select 4 Corresponding points on perspective transformed image
	"""
	Select four points to be stored as destination points to compute perspective matrix. Four points must match with
	original image to compute accurate transformation matrix
	"""
	print("\nClick on four corresponding points to original image points. Press any key to continue.")
	perspective_matrix = np.matrix(np.float32([[1, 0.5, 20],[0, 1, 20],[0, 0, 1]]))
	img = cv2.warpPerspective(star_image, perspective_matrix, (star_image.shape[1], star_image.shape[0]))
	# reset global variable points
	points = []
	points = getting_image_points(img, points)
	# set destination points to resulting points from click_event
	destination = np.float32(points).copy()

	# Compute four point perspective transformation with selected source and destination points using built in function
	print("\nBuilt-in Perspective Transform with Four Points")
	builtin_perspective_four = cv2.getPerspectiveTransform(source, destination, len(source))
	print(builtin_perspective_four)
	# Compute four point perspective transformation with selected source and destination points using function i wrote
	print('\nRequired Perspective Transform with Four Points')
	perspective_four = getPerspectiveTransform(source, destination, len(source))
	print(perspective_four)

	print("\nFour Points Error")
	perspective_error = np.sum(np.abs(np.subtract(builtin_perspective_four, perspective_four, dtype=float)))
	print(perspective_error)

	# Display perspective transformed image
	"""
	Display four point perspective transformed image using matrix calculated from getPerspectiveTransform(source, dest, n)
	"""
	print("\nDisplaying Original Image on Left and Four Point Perspective Transformed Image on Right. Press any key to continue.")
	# Transform image using computed four point transformation matrix
	four_point = cv2.warpPerspective(star_image, perspective_four,
									 (star_image.shape[1], star_image.shape[0]))

	# Display original and transformed image
	plt.subplot(121), plt.imshow(star_image[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(four_point[:, :, ::-1]), plt.title('Four Point Perspective Transform')
	plt.show()
	#
	# # --------------------------------------------
	# # 8 POINT PERSPECTIVE
	# # --------------------------------------------
	print(
		"--------------------------------------------\n8 POINT PERSPECTIVE\n--------------------------------------------")

	# Select 8 Corresponding points on original image
	"""
	Select eight points to be stored as source points to compute perspective matrix.
	Next, a transformed image with a known perspective transformation will display in which you select the corresponding points
	"""
	print("\nClick on eight points. Press any key to continue.")
	points = []
	img = star_image.copy()
	points = getting_image_points(img, points)
	source = np.float32(points).copy()

	print("\nClick on eight points that correspond to original image points. Press any key to continue.")
	img = cv2.warpPerspective(star_image, perspective_matrix, (star_image.shape[1], star_image.shape[0]))
	points = []
	points = getting_image_points(img, points)
	destination = np.float32(points)

	print("\nOverconstrained Built-in Perspective Transform with Eight Points")
	builtin_perspective_eight = cv2.getPerspectiveTransform(source[0:4], destination[0:4], 4) # only first four points because 8 does not work for the built in
	print(builtin_perspective_eight) # display resulting matrix
	# Compute eight point perspective transformation with selected source and destination points using function i wrote
	print('\nOverconstrained Required Perspective Transform with Eight Points')
	perspective_eight = getPerspectiveTransform(source, destination, len(source))
	print(perspective_eight) # display resulting matrix

	print("\nOverconstrained Eight Points Error")
	perspective_error_eight = np.sum(np.abs(np.subtract(builtin_perspective_eight, perspective_eight, dtype=float)))
	print(perspective_error_eight)

	# Display 8 point perspective transformed image
	"""
	Display eight point perspective transformed image using matrix calculated from getPerspectiveTransform(source, dest, n)
	"""
	print("\nDisplaying Original Image on Left and Eight Point Perspective Transformed Image on Right. Press any key to continue.")
	# Transform image using computed four point transformation matrix
	eight_point = cv2.warpPerspective(star_image, perspective_eight,
									  (star_image.shape[1], star_image.shape[0]))

	# Display original and transformed image
	plt.subplot(121), plt.imshow(star_image[:, :, ::-1]), plt.title('Original Image')
	plt.subplot(122), plt.imshow(eight_point[:, :, ::-1]), plt.title('Eight Point Perspective Transform')
	plt.show()
	"""
	--------------------------------------------
	IMAGE BLENDING
	--------------------------------------------
	"""

	# set of images to be used with for no unwarping, perspective unwarping, and affine unwarping
	images_set = [['pr1_images/im4_1.png', 'pr1_images/im4_2.png'], ['pr1_images/im5_1.png', 'pr1_images/im5_2.png'],
				  ['pr1_images/im6_1.png', 'pr1_images/im6_2.png'], ['pr1_images/im1_1.jpg', 'pr1_images/im1_2.jpg']]

	# --------------------------------------------
	# NO UNWARPING AND MOSAIC
	# --------------------------------------------
	print(
		"# --------------------------------------------\n# NO UNWARPING AND MOSAIC\n--------------------------------------------")

	"""
	Creates mosaic with two images without any transformations
	There are four instances of mosaicing
	"""

	count = 1;
	for set in images_set:
		points = []
		print("\nUNWARPING", count)
		img2 = cv2.imread(set[0])
		img = cv2.imread(set[1])
		Blend(img, img2, points, count)
		count +=1

	# --------------------------------------------
	# PERSPECTIVE UNWARPING AND MOSAIC
	# --------------------------------------------
	print("# --------------------------------------------\n# PERSPECTIVE UNWARPING AND MOSAIC\n--------------------------------------------")

	"""
	Creates mosaic with two images after perspective transformation right image
	There are four instances of mosaicing
	"""

	count = 1;
	for set in images_set:

		# display images together
		print("\nPERSPECTIVE UNWARPING", count)
		imageleft = cv2.imread(set[0])
		imageright = cv2.imread(set[1])

		plt.subplot(121), plt.imshow(imageleft[:, :, ::-1]), plt.title('Left Image')
		plt.subplot(122), plt.imshow(imageright[:, :, ::-1]), plt.title('Right Image')
		plt.show()

		points = []
		img = imageleft.copy()
		print("\nSelect four points from the left image to correspond with points from the right image. Press any key to continue.")
		points = getting_image_points(img, points)
		leftpoints = points.copy()

		points = []
		print("\nSelect four points from the right image to correspond with points from the left image. Press any key to continue.")
		img = imageright.copy()
		points = getting_image_points(img, points)
		rightpoints = points.copy()

		V = getPerspectiveTransform(leftpoints, rightpoints, 4)
		H = np.linalg.inv(V)

		result=cv2.warpPerspective(imageright,H,(imageleft.shape[1]+imageright.shape[1],imageleft.shape[0]))

		fig = plt.figure(figsize=(5, 5))
		result[0:imageleft.shape[0], 0:imageleft.shape[1]] = imageleft
		plt.subplot(121), plt.imshow(result[:, :, ::-1]), plt.title('Perspective Unwarping Mosaic')
		plt.show()
		fig.savefig('pr1_images/perspective_'+str(count)+'_12.png')
		count +=1

	# --------------------------------------------
	# AFFINE WARPING AND MOSAIC
	# --------------------------------------------

	print(
		"--------------------------------------------\nAFFINE WARPING AND MOSAIC\n--------------------------------------------")
	"""
	Creates mosaic with two images after affine transformation on right image
	There are four instances of mosaicing
	"""

	count = 1;
	for set in images_set:

		print("\nAFFINE UNWARPING", count)
		imageleft = cv2.imread(set[0])
		imageright = cv2.imread(set[1])

		# display images together
		plt.subplot(121), plt.imshow(imageleft[:, :, ::-1]), plt.title('Left Image')
		plt.subplot(122), plt.imshow(imageright[:, :, ::-1]), plt.title('Right Image')
		plt.show()

		# set image to image of interest
		img = imageleft.copy()
		# reset global points back
		points = []
		print("\nSelect three points from the left image to correspond with points from the right image. Press any key to continue.")
		points = getting_image_points(img, points)
		# set left points to output of points
		left_points = points.copy()

		# set image to image of interest
		img = imageright.copy()
		# reset global points back
		points = []
		print("\nSelect three points from the right image to correspond with points from the left image. Press any key to continue.")
		points = getting_image_points(img, points)
		# set right points to output of points
		right_points = points.copy()

		# compute affine matrix from left and right points
		affine_matrix = MyAffine(left_points, right_points)

		# add bottom row to affine matrix since it produces a 2x3
		new_affine = np.vstack([affine_matrix, [0,0,1]])

		# solve for affine matrix
		affine_matrix_final = np.linalg.inv(new_affine)

		# perspective warp affine matrix with image (this was in TA's jupyter example)
		result = cv2.warpPerspective(imageright, affine_matrix_final, (imageleft.shape[1] + imageright.shape[1], imageleft.shape[0]))

		# concatenate images
		result[0:imageleft.shape[0], 0:imageleft.shape[1]] = imageleft

		# used matplotlib here because cv2 was producing a very wrong image
		fig = plt.figure(figsize=(5, 5))
		plt.subplot(122), plt.imshow(result[:, :, ::-1]), plt.title('Affine Unwarped Mosaic')
		plt.show()
		fig.savefig('pr1_images/affine_'+str(count)+'_12.png')
		count +=1


	# --------------------------------------------
	# EXTRA CREDIT
	# --------------------------------------------
	# HARRIS DETECTOR EC

	# imageleft = cv2.imread('pr1_images/im1_1.jpg')
	# imageright = cv2.imread('pr1_images/im1_2.jpg')
	#
	# print("cc of images")
	# print(np.mean(np.multiply((imageleft - np.mean(imageleft)), (imageright - np.mean(imageright)))) / (np.std(imageleft) * np.std(imageright)))
	#
	#
	# harris1, harris2 = HarrisDetector(imageleft, imageright)
	#
	# CrossCorrelation(harris1, harris2)



