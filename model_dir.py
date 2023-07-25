import os

model_dir = '/home/shaobo/project/models'
model_dir_no_rnn = os.path.join(model_dir, 'R_Unet_no_rnn.pth')
model_dir_aps = os.path.join(model_dir, 'R_Unet_gray.pth')
model_sod_dir = os.path.join(model_dir, 'sod.pth')
model_swift_dir = os.path.join(model_dir, 'swift.pth')
model_rnn_dir = os.path.join(model_dir, 'R_Unet_ca_2.pth')
model_swin_rnn_dir = os.path.join(model_dir, 'R_swin.pth')
model_swin_dir = os.path.join(model_dir, 'swin.pth')
model_p2t_dir = os.path.join(model_dir, 'p2t.pth')
real_data_dir = '/mnt2/shaobo/evimo/sunny_record/'
labeled_image_dir = '/mnt2/shaobo/evimo/sunny_record/label_images'