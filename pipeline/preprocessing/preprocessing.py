import numpy as np
import scipy.io as sio
import cv2

# to be consistent with different versions of the opencv
cv2.CV_LOAD_IMAGE_GRAYSCALE = 0

def consturct_pixel_vectors(rio_path, pixpath):
    """ Construct pixel vectors from the raw pixel values.
    """

    nobjs, nimgs, rsize = 100, 72, 32*32*3
    img_ids = np.arange(0, 360, 5)

    raw_pixs_mat = np.zeros((nimgs, rsize), dtype=np.float32)
    for i in range(1, nobjs+1):
        for ii in range(len(img_ids)):
            img_name = 'obj'+str(i)+'__'+str(img_ids[ii])+'.png'
            img_path = rio_path + img_name
            img = cv2.imread(img_path).flatten()
            raw_pixs_mat[ii, :] = img

        objmatname = pixpath+'obj'+str(i)+'.mat'

        sio.savemat(objmatname, mdict={'obj': raw_pixs_mat})
        raw_pixs_mat = np.zeros((nimgs, rsize), dtype=np.float32)


def construct_binary_vectors(rio_path, pixbinmats):
    """ Construct binary vectors from the pixel values.
        Note that these vector values are either 1 or 0.
    """

    nobjs, nimgs, rsize = 100, 72, 32*32*3
    img_ids = np.arange(0, 360, 5)

    bin_pixs_mat = np.zeros((nimgs, rsize), dtype=np.float32)
    for i in range(1, nobjs+1):
        for ii in range(len(img_ids)):
            img_name = 'obj'+str(i)+'__'+str(img_ids[ii])+'.png'
            img_path = rio_path + img_name

            gimg = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            bimg = cv2.threshold(gimg, 50, 255, cv2.THRESH_BINARY)[1]
            # set 255 to 1
            nonz = bimg.nonzero()
            bimg[nonz] = 1.0
            bin_pixs_mat[ii, :] = bimg.flatten()

        objmatname = pixbinmats+'obj'+str(i)+'.mat'
        sio.savemat(objmatname, mdict={'obj': bin_pixs_mat})
        bin_pixs_mat = np.zeros((nimgs, rsize), dtype=np.float32)


def display_images(path, name_list, num_of_imgs):
    """ Display images in the image name list.
        That should be reconsidered for later process.
    """

    resize_dims = (32, 32)
    for i in range(len(name_list)):
        for j in range(num_of_imgs):
            showimg = cv2.imread(path + name_list[i][j])
            imgtitle = "Class id" + str(i) + name_list[i][j]
            cv2.imshow(imgtitle, cv2.resize(showimg, resize_dims))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():

    rio_path = '../COIL100resize32/'
    pixpath, pixbinmats = '../DATA/pixel/', '../DATA/pixel_binary/'

    #consturct_pixel_vectors(rio_path, pixpath)
    #construct_binary_vectors(rio_path, pixbinmats)


if __name__ == '__main__':
    main()
