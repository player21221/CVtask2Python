import numpy as np
import cv2


def define_point_cloud(image, imsk, mskclr):
    pc = []
    # for i in [0, image.cols] and j in [0, image.rows]:
    #     if imsk[i, j] == mskclr:
    #         pc.append(image[i, j])
    for i,j in image:
        if imsk[i,j] == mskclr:
            pc.extend(image[i,j])
    return pc


def construct_color_filter(pts):
    p0 = np.mean(pts, axis=0)
    U, s, Vt = np.linalg.svd(pts - p0)
    v = Vt[0, :]  # cylinder axis
    t = (pts - p0).dot(v)
    t1 = np.percentile(t, 5)  # cylinder bases
    t2 = np.percentile(t, 95)
    dp = np.linalg.norm(np.outer(t, v) + p0 - pts, axis=1)
    R = np.percentile(dp, 95)  # cylinder radius
    f = (v, p0, (t1, t2, R))
    return f


def apply_color_filter(f, im):
    v, p0, (t1, t2, R) = f
    ny, nx, nc = im.shape
    assert nc == 3
    p = np.reshape(im, (-1, 3))
    t = (p - p0).dot(v)
    dt = np.abs(t - (t1 + t2) / 2) - (t2 - t1) / 2
    dt = np.maximum(dt, 0)
    dp = np.linalg.norm(np.outer(t, v) + p0 - p, axis=1)
    dp = np.maximum(dp - R, 0)
    d = dp + dt
    return np.reshape(d, (ny, nx))


# pts = [[200, 100, 50], [100, 50, 25]]
# pts=[[200,100],[100,50],[50,25]]
# pts=((200, 100, 50),(100, 50, 25))

image = cv2.imread("msg271576874-60545.jpg")
imsk = cv2.imread("msg271576874-60545_Mask.png")
mskclr = np.array([0, 255, 0])

# pcld = define_point_cloud(image, imsk, mskclr)
# fu = construct_color_filter(pcld)
# res = apply_color_filter(fu, image)

imagedst=0
imagedst=image

cv2.erode(image, np.ones([3,3]), imagedst, np.array([-1,-1]), 1)


cv2.imshow("result", image)
cv2.waitKey()
# a = construct_color_filter(pts)
# print(a)
