import sys, math, cv2, numpy as np

MIN_SIZES = [[10,16,24],[32,48],[64,96],[128,192,256]]
STEPS = [8,16,32,64]

def make_priors(w, h):
    f2 = (int(int((h+1)/2)/2), int(int((w+1)/2)/2))
    f3 = (f2[0]//2, f2[1]//2); f4=(f3[0]//2, f3[1]//2); f5=(f4[0]//2, f4[1]//2); f6=(f5[0]//2, f5[1]//2)
    pri=[]
    for k,f in enumerate([f3,f4,f5,f6]):
        for i in range(f[0]):
            for j in range(f[1]):
                for ms in MIN_SIZES[k]:
                    pri.append(((j+0.5)*STEPS[k]/w, (i+0.5)*STEPS[k]/h, ms/w, ms/h))
    return np.array(pri, np.float32)

def detect(net, img, thr=0.7, nms=0.3):
    h, w = img.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(img, 1.0, (w,h), (0,0,0), swapRB=False, crop=False))
    loc, conf, iou = net.forward(["loc","conf","iou"])
    loc = loc.reshape(-1,14); conf = conf.reshape(-1,2); iou = np.clip(iou.reshape(-1),0,1)
    pri = make_priors(w,h)
    assert len(pri)==len(loc), (len(pri), len(loc))
    score = np.sqrt(np.clip(conf[:,1]*iou, 0, None))
    keep = score >= thr
    p, l, s = pri[keep], loc[keep], score[keep]
    cxy = (p[:,0:2] + l[:,0:2]*0.1*p[:,2:4]) * (w,h)
    wh = p[:,2:4]*np.exp(l[:,2:4]*0.2) * (w,h)
    xy = cxy - wh/2
    boxes = np.hstack([xy, wh])
    if len(boxes)==0: return np.zeros((0,5))
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), s.tolist(), thr, nms, top_k=5000)
    idx = np.array(idx).reshape(-1)
    return np.hstack([boxes[idx], s[idx,None]])

if __name__=="__main__":
    net = cv2.dnn.readNetFromONNX(sys.argv[1])
    cap = cv2.VideoCapture(sys.argv[2])
    counts=[]; first=None
    for i in range(60):
        ok, fr = cap.read()
        if not ok: break
        d = detect(net, fr)
        counts.append(len(d))
        if first is None and len(d): first=(i, fr.shape, d)
    print("frames:", len(counts), "faces per frame:", counts)
    if first:
        i, shp, d = first
        print("frame", i, shp, "boxes x,y,w,h,score:\n", np.round(d,1))
