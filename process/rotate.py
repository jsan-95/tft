import numpy as np
import cv2 as cv

def createTransformationMatrix(img, x, y, z):
    im_h, im_w, im_ch_ct = img.shape
    view_x, view_y, view_z = float(x), float(y), float(z)

    # Create a rotation matrix
    view_x1 = (view_x - 90) * (np.pi / 180)
    view_y1 = (view_y - 90) * (np.pi / 180)
    view_z1 = (view_z - 90) * (np.pi / 180)
    R_array = np.array([view_x1, view_y1, view_z1])

    R, jacobian = cv.Rodrigues(R_array)
    R[0][2] = 0
    R[1][2] = 0
    R[2][2] = 1

    # Create and combine with translation matrix
    Trans_Mat = np.array([[1, 0, -im_w / 2],
                          [0, 1, -im_h / 2],
                          [0, 0, 1]])

    R_T_Mat = np.dot(R, Trans_Mat)
    R_T_Mat[2][2] += im_h

    # Create and combine with camera matriview_x
    Intrinsic_Mat = np.array([[im_h, 0, im_w / 2],
                              [0, im_h, im_h / 2],
                              [0, 0, 1]])

    rotation_matrix = np.dot(Intrinsic_Mat, R_T_Mat)
    return rotation_matrix


def rotate(img, x, y, z):
    im_h, im_w, im_ch_ct = img.shape
    M_Transformation_Matrix = createTransformationMatrix(img, x, y, z)
    dst = cv.warpPerspective(img, M_Transformation_Matrix, (im_w, im_h))

    return dst
