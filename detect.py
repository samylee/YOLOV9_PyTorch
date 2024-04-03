import torch
import cv2
import numpy as np

from models.YOLOV9 import YOLOV9
from utils.utils import letterbox, non_max_suppression, scale_coords, load_pretrain_weights


def preprocess(img_path, net_size):
    img0 = cv2.imread(img_path)  # BGR

    # Letterbox
    img = letterbox(img0, net_size)[0]

    # BGR to RGB
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    # norm to torch
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img, img0


def model_init(model_path, C=80):
    # load moel
    model = YOLOV9(C=C, deploy=True)
    load_pretrain_weights(model, model_path)
    model.eval()
    return model


if __name__ == '__main__':
    # load moel
    checkpoint_path = 'weights/yolov9-c-converted-samylee.pt'
    C = 80
    model = model_init(checkpoint_path, C)

    # params init
    net_size = 640
    conf_thresh = 0.25
    iou_thresh = 0.45

    # coco
    with open('assets/coco.names', 'r') as f:
        classes = [x.strip().split()[0] for x in f.readlines()]

    # preprocess
    img_path = 'demo/000004.jpg'
    img, im0 = preprocess(img_path, net_size)

    # forward
    pred = model(img)

    # postprocess
    pred = non_max_suppression(pred, conf_thresh, iou_thresh)

    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{classes[int(cls)]} {conf:.2f}'
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(im0, c1, c2, (0, 255, 0), 2)
                cv2.putText(im0, label, c1, 0, 0.6, (0, 255, 255), 2)
    cv2.imwrite('assets/result4.jpg', im0)
    # cv2.imshow('test', im0)
    # cv2.waitKey(0)