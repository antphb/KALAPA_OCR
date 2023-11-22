import argparse
from config import Config
from model import *
from utils import *



def predict_answer(img):
    img = np.array([img])
    prediction = crnn.predict(img)
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                        greedy=True)[0][0])
    pred = ""
    for p in out[0]:  
        if int(p) != -1:
            pred += Config.label[int(p)]
    return pred

if __name__ =="__main__":
    args = argparse.ArgumentParser()
    image_path = args.add_argument('--image_path', type=str, default='image/4.jpg',required=True, help='path to image')
    crnn = CRNN(Config.input_shape, len(Config.label)+1, Config.max_label_len)
    crnn.load_weights('check_weight.hdf5')
    print("ORC image prediction",predict_answer(img_processing(args.image_path)))
