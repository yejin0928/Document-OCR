# EasyOCR을 활용한 서류 분류 OCR
import cv2
import os
import natsort
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import easyocr
from difflib import SequenceMatcher
import distance
import time
import math
import jaro
import timeit

# pdf 분할 후, 이미지로 저장
def pdf_to_image(input_file_path, input_file_name):
    pages = convert_from_path(input_file_path + input_file_name,
                              poppler_path="poppler_path")
    for i, page in enumerate(pages):
        page.save(split_file_path + input_file_name + "_" + str(i + 1) + ".jpg", "JPEG")


# 제목의 1/2 영역 Blur
def x_cord_contour(contour):
    M = cv2.moments(contour)
    return (int(M['m10'] / M['m00']))


def makeSquare(not_square):
    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]

    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2

        if (height > width):
            pad = int((height - width) / 2)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = int((width - height) / 2)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    return doublesize_square


def resize_to_pixel(dimensions, image):
    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    return ReSizedImg


def blur(dir_path, split_path, blur_path):
    upsize_files = os.listdir(dir_path + split_path)
    sort_upsize_files = natsort.natsorted(upsize_files)

    for i in range(0, len(sort_upsize_files)):
        result_x, result_y, result_w, result_h = 0, 0, 0, 0
        current_file = sort_upsize_files[i]
        image = cv2.imread(split_path + current_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]
        contours = sorted(filtered_contours, key=x_cord_contour, reverse=False)
        full_number = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if 45 >= w >= 35 and 40 >= h >= 35:
                if 650 >= x >= 580 and 400 >= y >= 360:
                    roi = blurred[y:y + h, x:x + w]
                    ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
                    squared = makeSquare(roi)
                    final = resize_to_pixel(20, squared)
                    final_array = final.reshape((1, 400))
                    final_array = final_array.astype(np.float32)
                    result_x = x
                    result_y = y
                    result_w = w
                    result_h = h

        ksize = 30
        b_x, b_y, b_w, b_h = result_x, result_y, result_w, result_h
        if b_x != 0 and b_y != 0 and b_w != 0 and b_h != 0:
            roii = image[int(b_y + 1.13 * b_h):int(b_y + 3 * b_h), int(b_x - b_w):int(b_x + 9 * b_w)]
            roii = cv2.blur(roii, (ksize, ksize))
            image[int(b_y + 1.13 * b_h):int(b_y + 3 * b_h), int(b_x - b_w):int(b_x + 9 * b_w)] = roii
        cv2.imwrite(blur_path + current_file[0:len(current_file) - 4] + ".jpg", image)


# 이미지 세로로 4분할 중 최상단 저장
def crop(dir_path, split_path, crop_path):
    split_files = os.listdir(dir_path + split_path)
    sort_split_files = natsort.natsorted(split_files)

    for i in range(0, len(sort_split_files)):
        current_file = sort_split_files[i]
        image1 = Image.open(split_path + current_file)
        width, height = image1.size
        croppedImage = image1.crop((0, 0, width, height / 4))
        croppedImage.save(crop_path + current_file[0:len(current_file) - 4] + ".jpg")


# 이미지 2배 확대
def upsize_image(dir_path, crop_path, upsize_path):
    crop_files = os.listdir(dir_path + crop_path)
    sort_crop_files = natsort.natsorted(crop_files)

    for i in range(0, len(sort_crop_files)):
        current_file = sort_crop_files[i]
        input_img = cv2.imread(crop_path + current_file, cv2.IMREAD_COLOR)
        height, width, channel = input_img.shape
        upsized_image = cv2.pyrUp(input_img, dstsize=(width * 2, height * 2),
                                  borderType=cv2.BORDER_DEFAULT)
        cv2.imwrite(upsize_path + current_file[0:len(current_file) - 4] + ".jpg", upsized_image)

# 이미지 선명하게
def filter2D(dir_path, upsize_path, filter2D_path):
    upsize_files = os.listdir(dir_path + upsize_path)
    sort_upsize_files = natsort.natsorted(upsize_files)

    for i in range(0, len(sort_upsize_files)):
        current_file = sort_upsize_files[i]
        image = cv2.imread(upsize_path + current_file, cv2.IMREAD_GRAYSCALE)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image_sharp = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(filter2D_path + current_file[0:len(current_file) - 4] + ".jpg", image_sharp)

# EasyOCR
def easyOCR(dir_path, filter2D_path, easy_path, doc_list):
    filter2D_files = os.listdir(dir_path + filter2D_path)
    sort_filter2D_files = natsort.natsorted(filter2D_files)
    reader = easyocr.Reader(["ko"])
    for i in range(0, len(sort_filter2D_files)):
        result_string = ""
        current_file = sort_filter2D_files[i]
        result = reader.readtext(filter2D_path + current_file)
        f = open(easy_path+"EasyOCR-result.txt","a")
        for j in result:
            result_string += j[1]
        result_string = result_string.replace(" ", "")
        closed_string = []
        flag = 0
        f.write("________________________________________\n")
        f.write(current_file+"에 대한 결과\n")
        for d in doc_list:
            closed_string.append(jaro.jaro_winkler_metric(result_string,d))
            if d in result_string:
                final_result = d
                flag = 1
                f.write(d + " : 100% 일치\n")
        if flag == 0:
            max_num = max(closed_string)
            max_num_list =[t for t, v in enumerate(closed_string) if v == max_num]
            fin_= []
            for l in range(0, len(max_num_list)):
                fin_.append(doc_list[max_num_list[l]])
            f.write("\n")
            f.write("\n")
            f.write("가장 일치하는 값:"+', '.join(fin_)+"\n")
        f.close()
# main문

# 전체 파일 디렉토리
dir_path = ""
# 테스트 파일 명
input_file_path = ""
# split file path
split_file_path = ""
# cropped image path
cropped_file_path = ""
# upsized image folder
upsized_file_path = ""
# filter2D image path
filter2D_file_path = ""
# erosion image path
erosion_file_path = ""
# dnn image path
dnn_file_path = ""
# Resolution path
resolution_file_path = ""
# 최종 image 파일 path
destination_file_path = ""
# EasyOCR txt 기존 테스트(이미지 보정x)
easyocr_before_txt_path = ""
# EasyOCR txt 테스트1(blur 없는 보정)
easyocr_test1_txt_path = ""
# EasyOCR txt 테스트2(blur 있는 보정)
easyocr_test2_txt_path = ""

document_list = []
# blur result
blur_file_path = ""

# 동작 과정
# 1. pdf를 이미지로 분할 -> 2. blur 3. 1/4 crop image-> 4. 2X upsize image -> 5. filter2D -> 6.EasyOCR

input_files = os.listdir(dir_path + input_file_path)
sort_input_files = natsort.natsorted(input_files)
for i in range(0, len(sort_input_files)):
    input_file_name = sort_input_files[i]
    input_len = len(input_file_name)
    file_format = input_file_name[input_len - 4:input_len]

    if file_format == ".pdf" or file_format == ".PDF":
        print(input_file_name + " : pdf to image")
        pdf_to_image(input_file_path, input_file_name)
    else:
        print(input_file_name + " : input image")
        input_img = cv2.imread(input_file_path + input_file_name, cv2.IMREAD_COLOR)
        cv2.imwrite(split_file_path + input_file_name[0:input_len - 4] + "-img.jpg", input_img)

blur(dir_path, split_file_path, blur_file_path)
crop(dir_path, blur_file_path, cropped_file_path)
upsize_image(dir_path, cropped_file_path, upsized_file_path)
filter2D(dir_path, upsized_file_path, filter2D_file_path)
easyOCR(dir_path, split_file_path, easyocr_before_txt_path, document_list)