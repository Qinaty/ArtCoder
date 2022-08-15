from pyzxing import BarCodeReader
import os
import cv2
# from pylibdmtx import pylibdmtx

qr_imgs = os.listdir('./data')

for qr_img in qr_imgs:
    qr_path = os.path.join('./data', qr_img)
    qr_img = cv2.imread(qr_path)

    # ZXing
    # reader = BarCodeReader()
    # result = reader.decode(qr_path)
    # print(result)

    # WeChat - cannot decode datamatrix
    # # reader = cv2.wechat_qrcode_WeChatQRCode("./wechat_qrcode_protocal/detect.prototxt",
    # #                                             "./wechat_qrcode_protocal/detect.caffemodel",
    # #                                             "./wechat_qrcode_protocal/sr.prototxt",
    # #                                             "./wechat_qrcode_protocal/sr.caffemodel")
    # reader = cv2.wechat_qrcode_WeChatQRCode()
    # result= reader.detectAndDecode(qr_img)
    # print(result)

    # OpenCV
    # reader = cv2.QRCodeDetector()
    # result = reader.detectAndDecode(qr_img)
    # print(result)

    # Pylibdmtx
    # result = pylibdmtx.decode(qr_img)
    # print(result)