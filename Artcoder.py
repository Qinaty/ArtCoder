import os

from vgg import Vgg16
import utils
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from SS_layer import SSlayer
import time


def artcoder(STYLE_IMG_PATH, CONTENT_IMG_PATH, CODE_PATH, OUTPUT_DIR,
             LEARNING_RATE=0.001, CONTENT_WEIGHT=1e7, STYLE_WEIGHT=1e15, CODE_WEIGHT=1e20, MODULE_SIZE=16, MODULE_NUM=37,
             EPOCHS=50000, Dis_b=80, Dis_w=180, Correct_b=50, Correct_w=200, USE_ACTIVATION_MECHANISM=True):
    # STYLE_IMG_PATH = './style/redwave4.jpg'
    # CONTENT_IMG_PATH = './content/boy.jpg'
    # CODE_PATH = './code/boy.jpg'
    # OUTPUT_DIR = './output/'
    #  Modify
    img_name = CONTENT_IMG_PATH.split('/')[-1].split('.')[0]
    # print(img_name)
    # output_dir = os.path.join(OUTPUT_DIR, img_name)
    output_dir = os.path.join(OUTPUT_DIR, time.strftime(img_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()))
    print(output_dir)
    if os.path.exists(output_dir):
        utils.del_file(output_dir)
    else:
        os.mkdir(output_dir)
    IMAGE_SIZE = MODULE_SIZE * MODULE_NUM   # 592*592

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    vgg = Vgg16(requires_grad=False).cuda()  # vgg16 model
    ss_layer = SSlayer(requires_grad=False).cuda()

    # style_img = utils.load_image(filename=STYLE_IMG_PATH, size=IMAGE_SIZE)
    content_img = utils.load_image(filename=CONTENT_IMG_PATH, size=IMAGE_SIZE)
    code_img = utils.load_image(filename=CODE_PATH, size=IMAGE_SIZE)
    init_img = utils.add_pattern(content_img, code_img)


    # style_img = transform(style_img)
    content_img = transform(content_img)
    init_img = transform(init_img)  # content_img + QR pattern

    # repeat(1,1,1,1) (C H W)->(B C H W)
    init_img = init_img.repeat(1, 3, 1, 1).cuda()
    # style_img = style_img.repeat(1, 1, 1, 1).cuda()  # make fake batch
    content_img = content_img.repeat(1, 3, 1, 1).cuda()
    # print(content_img.shape)

    # features_style = vgg(style_img)  # feature maps extracted from VGG
    features_content = vgg(content_img)

    # gram_style = [utils.gram_matrix(i) for i in features_style]  # gram matrix of style feature
    mse_loss = nn.MSELoss()

    #   从content_img + QR pattern开始优化
    y = init_img.detach()  # y is the target output. Optimized start from the content image.
    y = y.requires_grad_()  # let y to require grad

    optimizer = optim.Adam([y], lr=LEARNING_RATE)  # let optimizer to optimize the tensor y

    error_matrix, ideal_result = utils.get_action_matrix(
        img_target=utils.tensor_to_PIL(y),
        img_code=code_img,
        Dis_b=Dis_b, Dis_w=Dis_w
    )
    # get_target 把01的ideal result变成RGB图像
    code_target = ss_layer(utils.get_target(ideal_result, b_robust=Correct_b, w_robust=Correct_w))

    print(" Start training =============================================")
    for epoch in range(EPOCHS):

        def closure(code_target=code_target):

            optimizer.zero_grad()
            y.data.clamp_(0, 1) # 归一化
            features_y = vgg(y)  # feature maps of y extracted from VGG
            # gram_style_y = [utils.gram_matrix(i) for i in
            #                 features_y]  # gram matrixs of feature_y in relu1_2,2_2,3_3,4_3

            content_loss = 0
            for i in [0, 1, 2, 3]:
                content_loss += mse_loss(features_y[i], features_content[i])
            content_loss = CONTENT_WEIGHT * content_loss
            # fc = features_content.relu3_3  # content tar·get in relu3_3
            # fy = features_y.relu3_3  # y in relu3_3
            # content_loss = CONTENT_WEIGHT * mse_loss(fc, fy)  # content loss

            # style_loss = 0  # add style_losses in relu1_2,2_2,3_3,4_3
            # for i in [0, 1, 2, 3]:
                # style_loss += mse_loss(gram_style_y[i], gram_style[i])
            # style_loss = STYLE_WEIGHT * style_loss

            code_y = ss_layer(y)

            if USE_ACTIVATION_MECHANISM == 1:
                error_matrix, ideal_result = utils.get_action_matrix(
                    img_target=utils.tensor_to_PIL(y),
                    img_code=code_img,
                    Dis_b=Dis_b, Dis_w=Dis_w)
                activate_num = np.sum(error_matrix)
                activate_weight = torch.tensor(error_matrix.astype('float32'))
                code_y = code_y.cpu() * activate_weight
                code_target = code_target.cpu() * activate_weight
            else:
                code_y = code_y.cpu()
                code_target = code_target.cpu()
                activate_num = MODULE_NUM * MODULE_NUM

            code_loss = CODE_WEIGHT * mse_loss(code_target.cuda(), code_y.cuda())

            # tv_loss = TV_WEIGHT * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
            #                        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            # total_loss = style_loss + code_loss + content_loss
            total_loss = code_loss + content_loss
            total_loss.backward(retain_graph=True)  # 保留backward后的中间参数

            # if epoch % 20 == 0:
            #     print(
            #         "Epoch {}: Style Loss : {:4f}. Content Loss: {:4f}. Code Loss: {:4f}. Activated module number: {:4.2f}. Discriminate_b：{:4.2f}. Discriminate_w：{:4.2f}.".format(
            #             epoch, style_loss, content_loss, code_loss, activate_num, Dis_b, Dis_w)
            #     )
            if epoch % 20 == 0:
                print(
                    "Epoch {}: Content Loss: {:4f}. Code Loss: {:4f}. Activated module number: {:4.2f}. Discriminate_b：{:4.2f}. Discriminate_w：{:4.2f}.".format(
                        epoch, content_loss, code_loss, activate_num, Dis_b, Dis_w)
                )
            if epoch % 1000 == 0:
                img_name = 'epoch=' + str(epoch) + '__Wcode=' + str("%.1e" % CODE_WEIGHT) + \
                           '__Wcontent' + str("%.1e" % CONTENT_WEIGHT) + '.jpg'
                utils.save_image_epoch(y, output_dir, img_name, code_img, addpattern=True)
                print('Save output: ' + img_name)
                return total_loss
            # if epoch % 1000 == 0:
            #     save_dir = os.path.join(output_dir, "model")
            #     os.mkdir(save_dir)
            #     save_name = str(epoch) + ".pth"
            #     save_path = os.path.join(save_dir, save_name)
            #     torch.save(encoder_decoder.module.state_dict(), save_path)

        optimizer.step(closure)
