# @author Ke Ma for final project
import cv2
import numpy as np
import os


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return noisy_image

def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)

def main(photo_name):

    girl_face_filename = photo_name
    print('opening image: ', girl_face_filename)

    girl_face_image = cv2.imread(girl_face_filename, cv2.IMREAD_UNCHANGED)
    girl_face_grayscale_image = cv2.cvtColor(girl_face_image, cv2.COLOR_BGR2GRAY)

    noisy_sigma = 35
    noisy_image = add_gaussian_noise(girl_face_grayscale_image, noisy_sigma)

    print('noisy image shape: {0}, len of shape {1}'.format(girl_face_image.shape, len(noisy_image.shape)))
    print('    WxH: {0}x{1}'.format(noisy_image.shape[1], noisy_image.shape[0]))
    print('    image size: {0} bytes'.format(noisy_image.size))

    noisy_filename = 'girl_face_noise_' + str(noisy_sigma) + '.jpg'
    cv2.imwrite(photo_name, noisy_image)

if __name__ == "__main__":
    file_dir = os.getcwd()
    for root, dirs, files in os.walk(file_dir):
        file_list = files
    for photo in files:
        x = photo.split('-')
        if len(x) > 1:
            main(photo)
