from PIL import Image
import numpy as np

def main():
    image = Image.open("402px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg")
    red, green, blue = div_RBG(image)
    RSVD = np.linalg.svd(red)
    BSVD = np.linalg.svd(blue)
    GSVD = np.linalg.svd(green)
    errors = dict()
    k = [1, 5, 10, 15, 20, 30, 60, 100]
    for i in k:
        new_red = k_best_approx(i, RSVD)
        new_blue = k_best_approx(i, BSVD)
        new_green = k_best_approx(i, GSVD)
        create_new_image(image, new_red, new_blue, new_green, i)
        errors[i] = error_value(GSVD[1], i)
    for k in errors:
        print(k, " ", errors[k])


def create_new_image(org_image, red, blue, green, k):
    pix = np.array(org_image)
    new_image = np.zeros((pix.shape[0], pix.shape[1], 3))
    new_image[:, :, 0] = red
    new_image[:, :, 1] = green
    new_image[:, :, 2] = blue
    new_image = Image.fromarray(new_image.astype('uint8'))
    new_image.save(str(k)+".png")


def div_RBG(image):
    pix = np.array(image)
    return pix[:, :, 0], pix[:, :, 1], pix[:, :, 2]


def k_best_approx(k, SVD):
    sigma = np.zeros((SVD[0].shape[0], SVD[2].shape[0]))
    for i in range(k):
        sigma[i][i] = SVD[1][i]
    return SVD[0]@sigma@SVD[2]


def error_value(singular, k):
    singular_values_sq = [i ** 2 for i in singular]
    return sum(singular_values_sq[k+1:])/sum(singular_values_sq)


if __name__ == "__main__":
    main()
