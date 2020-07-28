import bz2
import os
import os.path
import cv2
import time
import numpy as np
import threading
from urllib.request import urlopen
from model import create_model
from align import AlignDlib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

imgs, bb = [], []
imgFrame = None
stop_threads = False
class LeerImg(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global imgs
        global bb   
        while True:
            time.sleep(0.5)
            imgs, bb = align_image(imgFrame)
            global stop_threads 
            if stop_threads: 
                break

class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 

def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]

def align_image(jc_orig):
    imgs = []
    bb = alignment.getAllFaceBoundingBoxes(jc_orig)
    for b in bb:
      imgs.append(alignment.align(96, jc_orig, b, 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE))
    return imgs, bb 

dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

metadata = load_metadata('PD')
alignment = AlignDlib('models/landmarks.dat')
embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img, bb = align_image(img)
    for im in img:
        im = (im / 255.).astype(np.float32)
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(im, axis=0))[0]

targets = np.array([m.name for m in metadata])
encoder = LabelEncoder()
encoder.fit(targets)
svc = LinearSVC()

y_train = encoder.transform(targets)
X_train = embedded

svc = LinearSVC()
svc.fit(X_train, y_train)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
t = LeerImg()
t.start()
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        imgFrame = frame
        for color, img, b in zip(colors, imgs, bb):
            img = (img / 255.).astype(np.float32)
            pre = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            label = encoder.inverse_transform(svc.predict([pre]))[0]
            tl = (b.left(), b.top())
            br = (b.right(), b.bottom())
            confidence = np.max(svc.decision_function([pre]))
            text = '{}: {:.0f}%'.format(label, (confidence+0.5)*100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_threads = True
        t.join() 
        break

capture.release()
cv2.destroyAllWindows()