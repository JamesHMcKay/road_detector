import create_training_set


def get_sub_images(
        image,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps=5):
    height = image.shape[0]
    width = image.shape[1]
    h = (height / verticalDivisor)
    w = (width / horizontalDivisor)
    images = []
    pixels_per_step_h = int(round(h / number_of_steps))
    pixels_per_step_w = int(round(w / number_of_steps))

    for i in range(number_of_steps):
        for y in range(verticalDivisor):
            for j in range(number_of_steps):
                for x in range(horizontalDivisor):
                    lower_y = h * y + i * pixels_per_step_h
                    upper_y = lower_y + h
                    lower_x = w * x + j * pixels_per_step_w
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
        number_of_steps=5):
    index = 0
    height, width = result.shape
    h = (height / verticalDivisor)
    w = (width / horizontalDivisor)
    quarter_range = int(round(0.25 * h))
    quarter_rangex = int(round(0.25 * w))
    pixels_per_step_h = int(round(h / number_of_steps))
    pixels_per_step_w = int(round(w / number_of_steps))
    pps_h_2 = int(round(0.5 * pixels_per_step_h))
    pps_w_2 = int(round(0.5 * pixels_per_step_w))
    for i in range(number_of_steps):
        for y in range(verticalDivisor):
            for j in range(number_of_steps):
                for x in range(horizontalDivisor):
                    lower_y = h * y + i * pixels_per_step_h
                    upper_y = lower_y + h
                    lower_x = w * x + j * pixels_per_step_w
                    upper_x = lower_x + w
                    if (upper_y < height and upper_x < width):
                        likelihood = values[index]
                        like_road = likelihood[0]
                        index += 1
                        middle_y = int(round(0.5*(upper_y + lower_y)))
                        middle_x = int(round(0.5*(upper_x + lower_x)))
                        result[
                            middle_y - pps_w_2:middle_y + pps_w_2,
                            middle_x-pps_h_2:middle_x + pps_h_2
                            ] = like_road * 255
    return result
