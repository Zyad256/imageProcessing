import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
































# img_bgr = cv2.imread("kana.jpg")
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# img_pil = Image.fromarray(img_rgb)
# cv2.imwrite("output.jpg", img_bgr)
# img_pil.save("output_pil.jpg")


# img = cv2.imread("kana.jpg",cv2.IMREAD_GRAYSCALE)


# blue_Image = img[:, :, 0]
# blue_zero = np.zeros_like(blue_Image)
# blue_Image = cv2.merge([blue_Image, green_zero, red_zero])


# img = cv2.imread("kana.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# added = cv2.add(img,50)
# subtracted = cv2.subtract(img,50)
# multiplied = cv2.multiply(img,2)
# divided = cv2.divide(img,2)
# complemented = 255 - img
# img_added = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)
# img_subtracted = cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)
# img_multiplied = cv2.cvtColor(multiplied, cv2.COLOR_BGR2RGB)
# img_divided = cv2.cvtColor(divided, cv2.COLOR_BGR2RGB)
# img_complemented = cv2.cvtColor(complemented, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(12, 8))
# plt.suptitle("Arthimatic Operations", fontsize=16)
# plt.subplot(231), plt.imshow(img_rgb), plt.title("Original Image") ,plt.axis("off")
# plt.subplot(232), plt.imshow(img_added), plt.title("Added Image") ,plt.axis("off")
# plt.subplot(233), plt.imshow(img_subtracted), plt.title("Subtracted Image") ,plt.axis("off")
# plt.subplot(234), plt.imshow(img_multiplied), plt.title("Multiplied Image") ,plt.axis("off")
# plt.subplot(235), plt.imshow(img_divided), plt.title("Divided Image") ,plt.axis("off")
# plt.subplot(236), plt.imshow(img_complemented), plt.title("Complemented Image") ,plt.axis("off")
# plt.imsave("original_image.jpg", img_rgb)
# plt.tight_layout()



# img = cv2.imread("shapes.jpeg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret , binary = cv2.threshold(gray, 227, 255, cv2.THRESH_BINARY)
# inverted = ~binary # just when it's in 8bit format
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 6))
# plt.subplot(221), plt.imshow(img), plt.title("Original Image"), plt.axis("off")
# plt.subplot(222), plt.imshow(gray, cmap='gray'), plt.title("Grayscale Image"), plt.axis("off")
# plt.subplot(223), plt.imshow(inverted, cmap='gray'), plt.title("Inverted Image"), plt.axis("off")
# plt.subplot(224), plt.imshow(binary, cmap='gray'), plt.title("Binary Image"), plt.axis("off")
# plt.tight_layout()
# plt.show()
# plt.show()


# should be apply on grayscale image
# img_float = np.float32(img) / 255  
# log_img = np.log(1 + img_float)
# log_img = np.uint8(255 * log_img) 

# gamma = 0.5 
# gamma_corrected = np.power(img_float, gamma)
# gamma_corrected = np.uint8(255 * gamma_corrected)


# hist = cv2.calcHist((cv2.imread("kana.jpg"),cv2.IMREAD_GRAYSCALE),[0],None,[256],[0,256])
# plt.plot(hist)
# plt.xlim([0,256])
# plt.title('Histogram')

#plt.hist((cv2.imread("kana.jpg", cv2.IMREAD_GRAYSCALE)).ravel(), bins=256, range=(0, 256), color='gray')

# img_equalized = cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])




# img = cv2.imread("kana.jpg", cv2.IMREAD_GRAYSCALE)
# w,h = img.shape
# T = 256 / (w * h)
# frequency = [0] * 256


# for i in range(w):
#     for j in range(h):
#         intensity = img[i, j]
#         frequency[intensity] += 1


# cumulative_sum = [0] * 256
# cumulative_sum[0] = frequency[0]
# for k in range(1, 256):
#     cumulative_sum[k] = cumulative_sum[k - 1] + frequency[k]

# for i in range(256):
#     cumulative_sum[i] = min(255, int(cumulative_sum[i] * T))



# equalized_img = np.zeros_like(img)

# for i in range(w):
#     for j in range(h):
#         old_val = img[i, j]
#         new_val = cumulative_sum[old_val]
#         equalized_img[i, j] = new_val