import numpy as np
import scipy.io as sio
import cv2
from skimage.feature import hog
from skimage import data, color, exposure


def extract_hogs(data_path, feature_mats):
    """ Extract histogram oriented gradients.
    """
    nobjs, nimgs = 100, 72
    img_ids = np.arange(0, 360, 5)

    hogsize, orients, pix_pcell = 800, 8, (3, 3)
    hog_pixs_mat = np.zeros((nimgs, hogsize), dtype=np.float64)

    for i in range(1, nobjs+1):
        for ii in range(len(img_ids)):
            img_name = 'obj'+str(i)+'__'+str(img_ids[ii])+'.png'
            img_path = data_path + img_name
            img = cv2.imread(img_path)
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fdes, higm = hog(gimg, orientations=orients, pixels_per_cell=pix_pcell, cells_per_block=(1, 1), visualise=True)
            hog_pixs_mat[ii - 1, :] = fdes
            print fdes.shape, np.count_nonzero(fdes)
        sio.savemat(hog_mats + 'obj' + str(i) + '.mat', mdict={'obj' : hog_pixs_mat})
        hog_pixs_mat = np.zeros((nimgs, hogsize), dtype=np.float64)


def extract_nhistograms(data_path, feature_mats):
    """ Extract 3D Color Histograms (norm)
        Bin size is choosed to be 10.
    """
    nobjs, nimgs,binsize, histsize = 100, 72, 10, 1000
    img_ids = np.arange(0, 360, 5)
    hist_pixs_mat = np.zeros((nimgs, histsize), dtype=np.float32)

    hist_norm_pixs_mat = np.zeros((nimgs, histsize), dtype=np.float32)

    for i in range(1, nobjs+1):
        for ii in range(len(img_ids)):
            img_name = 'obj'+str(i)+'__'+str(img_ids[ii])+'.png'
            img_path = data_path + img_name
            img = cv2.imread(img_path)
            nhist = cv2.calcHist([img], [0, 1, 2], None, [binsize, binsize, binsize], [0, 256, 0, 256, 0, 256])
            nhist = cv2.normalize(norm_histogram, norm_histogram).flatten()
            hist_norm_pixs_mat[ii - 1, :] = norm_histogram

        sio.savemat(hist_norm_mats + 'obj' + str(i) + '.mat', mdict={'obj': hist_norm_pixs_mat})
        hist_norm_pixs_mat = np.zeros((nimgs, histsize), dtype=np.float32)


def extract_SURF_keypoints(data_path, feature_mats):
    """" Extract surf keypoints and descriptors from the COIL100 images.
    """
    nobjs, nimgs = 100, 72
    img_ids = np.arange(0, 360, 5)

    img_keyps = []

    for i in range(1, nobjs+1):
        for ii in range(len(img_ids)):
            img_name = 'obj'+str(i)+'__'+str(img_ids[ii])+'.png'
            img_path = data_path + img_name
            img = cv2.imread(img_path)
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            surf = cv2.xfeatures2d.SURF_create()
            keypoints, descriptors = surf.detectAndCompute(gimg, None)

            img_keyps.append(len(keypoints))
        surf_np = np.asarray(img_keyps)
        print "SURF: ObjID: %d Min ID: %d, Min Features: %d  Max ID: %d, Max Features: %d, Average Features: %f "%(i, np.argmin(surf_np), np.min(surf_np), np.argmax(surf_np), np.max(surf_np),np.mean(img_keyps))
        img_keyps =  []


def extract_SIFT_keypoints(data_path, feature_mats):
    """" Extract sift keypoints and descriptors with different contrast thresholds
         from the COIL100 images.
    """
    nobjs, nimgs = 100, 72
    img_ids = np.arange(0, 360, 5)

    cons_threshold = [0.04, 0.03, 0.02, 0.01, 0.004, 0.001, 0.0001, 0.0002, 0.0005]
    img_keyps, obj_keyps = [], []

    for j in range(len(cons_threshold)):
        for i in range(1, nobjs+1):
            for ii in range(len(img_ids)):
                img_name = 'obj'+str(i)+'__'+str(img_ids[ii])+'.png'
                img_path = data_path + img_name
                img = cv2.imread(img_path)
                gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=cons_threshold[j])
                keypoints, descriptors = sift.detectAndCompute(gimg, None)
                img_keyps.append(len(keypoints))

            obj_keyps.append(np.sum(img_keyps))
            sift_np = np.asarray(img_keyps)
            img_keyps = []
        print "Total keypoints: %d : Avg. keypoints of all objects: %f \n" % (np.sum(obj_keyps), np.mean(obj_keyps))
        obj_keyps = []

def main():

    data_path = '../COIL100resize32/'
    surf_feature_mats, sift_feature_mats =  '../DATA/pixel_sift/', '../DATA/pixel_surf/'
    nhist_feature_mats, hog_feature_mats = '../DATA/pixel_histogram_normalized/', '../DATA/pixel_hog/'

    # extract_SIFT_keypoints(data_path, sift_feature_mats)
    # extract_SURF_keypoints(data_path, surf_feature_mats)
    # extract_nhistograms(data_path, nhist_feature_mats)
    # extract_hogs(data_path, nhist_feature_mats)


if __name__ == '__main__':
    main()
