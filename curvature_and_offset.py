import numpy as np

def measure_curvature_pixels(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad

def offset(x_values, image):
    # Calculate position of the car from the centre
    left_x_values = x_values[0]
    right_x_values = x_values[1]
    # Get the centre of the lane using the poynomial equation
    lane_diff = abs(left_x_values[len(left_x_values) - 1] - right_x_values[len(left_x_values) - 1]) / 2
    lane_centre = lane_diff + left_x_values[len(left_x_values) - 1]
    # Get the centre of the car which is the centre of the image captured by the camera
    image_centre = image.shape[1] / 2
    # offset of the car from the lane centre in pixels
    offset_pixels = abs(lane_centre - image_centre)
    # offset of the car from the lane centre in meters
    xmeter_per_pixel = 3.7 / 700
    offset_meters = offset_pixels * xmeter_per_pixel
    return offset_meters