from moviepy.editor import VideoFileClip
import glob
import project_code as pc

images = glob.glob('camera_cal/calibration*.jpg')
mtx, dist = pc.camera_calib(images)

def video_pipeline(pipeline_image):
    # Undistort the image
    undistorted_image = pc.undistort_image(pipeline_image, mtx, dist)

    # Apply the combined thresholds on the image to get a combined binary image
    color_binary, combined_binary = pc.create_thresholded_binary_image(undistorted_image)

    after_mask = pc.create_interest_mask(combined_binary)
    transformed_perspective_image, M, Minv = pc.transform_perspective(after_mask)

    # Find the lane line from the polynomials formed
    sliding_window_img, ploty, left_fit, right_fit, left_fitx, right_fitx = pc.fit_polynomial(transformed_perspective_image)

    # In the next frame, find lane lines from the lane lines of the last frame
    result = pc.search_around_poly(sliding_window_img, left_fit, right_fit)

    # # Find the radius of curvature of the lane line
    # radius_curvature = get_radius_of_curvature(x_line_values)
    #
    # # Get radius of left and right curvature
    # left_radius_curvature = radius_curvature[0]
    # right_radius_curvature = radius_curvature[1]
    #
    # # Calculate offset in meters
    # offset_mtrs = offset()
    #
    # # Draw the lane lines on the original image by unwarping the image
    # orig_lane_image = original_lane_lines(warp_image, undistorted_image, x_line_values, MatrInv)

    # return orig_lane_image
    return result

output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(video_pipeline)#NOTE: this function expects color images!!
%time white_clip.write_videofile(output, audio=False)