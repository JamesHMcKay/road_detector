import create_training_set


def get_sub_images(
        image,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps):
    v_steps = number_of_steps * verticalDivisor
    h_steps = number_of_steps * horizontalDivisor

    height = image.shape[0]
    width = image.shape[1]
    h = (height / verticalDivisor)
    w = (width / horizontalDivisor)
    images = []
    pixels_per_step_h = int(round(height / v_steps))
    pixels_per_step_w = int(round(width / h_steps))

    for i in range(v_steps):
        for j in range(h_steps):
            lower_y = i * pixels_per_step_h
            upper_y = lower_y + h
            lower_x = j * pixels_per_step_w
            upper_x = lower_x + w
            if (upper_y < height and upper_x < width):
                subimage = image[lower_y:upper_y, lower_x:upper_x]
                images.append(subimage)
    return images


def assign_to_subimages(
        values,
        result,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps):
    index = 0
    height, width = result.shape
    v_steps = number_of_steps * verticalDivisor
    h_steps = number_of_steps * horizontalDivisor

    h = (height / verticalDivisor)
    w = (width / horizontalDivisor)
    images = []
    pixels_per_step_h = int(round(height / v_steps))
    pixels_per_step_w = int(round(width / h_steps))

    pps_h_2 = int(round(0.5 * pixels_per_step_h))
    pps_w_2 = int(round(0.5 * pixels_per_step_w))

    half_h = int(round(0.5 * h))
    half_w = int(round(0.5 * w))

    for i in range(v_steps):
        for j in range(h_steps):
            middle_y = i * pixels_per_step_h + half_h
            middle_x = j * pixels_per_step_w + half_w
            lower_y = i * pixels_per_step_h
            upper_y = lower_y + h
            lower_x = j * pixels_per_step_w
            upper_x = lower_x + w
            if (upper_y < height and upper_x < width):
                likelihood = values[index]
                like_road = likelihood[0]
                index += 1
                result[
                    middle_y - pps_w_2:middle_y + pps_w_2,
                    middle_x - pps_h_2:middle_x + pps_h_2] = like_road * 255
    return result
