
import numpy as np
import opencv as cv

def construct_color_filter(pts):
    p0 = np.mean(pts, axis=0)
    U,s,Vt = np.linalg.svd(pts - p0)
    v= Vt[0,:] # cylinder axis
    t = (pts - p0).dot(v)
    t1 = np.percentile (t, 5) # cylinder bases
    t2 = np.percentile (t, 95)
    dp = np.linalg.norm(np.outer(t, v) + p0 - pts, axis=1)
    R = np.percentile (dp, 95) # cylinder radius
    f = (v, p0, (t1, t2, R))
    return f



def apply_color_filter(f, im):
    v, p0, (t1, t2, R) = f
    ny,nx,nc = im.shape
    assert nc == 3
    p = np.reshape(im,(-1,3))
    t = (p - p0).dot(v)
    dt = np.abs(t - (t1 + t2) / 2) - (t2 - t1) / 2
    dt = np.maximum(dt, 0)
    dp = np.linalg.norm(np.outer(t, v) + p0 - p, axis=1)
    dp = np.maximum (dp - R, 0)
    d = dp + dt
    return np.reshape (d, (ny, nx))


pts=[[200, 100, 50],[100, 50, 25]]
#pts=[[200,100],[100,50],[50,25]]
#pts=((200, 100, 50),(100, 50, 25))

a = construct_color_filter(pts)
print(a)
