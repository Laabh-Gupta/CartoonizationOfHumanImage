"""General-purpose test script for image-to-image translation."""

import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flipping
    opt.display_id = -1   # no visdom

    dataset = create_dataset(opt)  # create dataset
    model = create_model(opt)      # create model
    # print(model.netG)

    ### ✅ Custom loading of net_G_A.pth for CycleGAN test
    # Replace model.setup(opt)
    model.eval()
    model.load_networks = lambda suffix: None  # disable default loading
    netG_A_path = os.path.join(opt.checkpoints_dir, opt.name, f'{opt.epoch}_net_G_A.pth')
    print(f"[INFO] Manually loading: {netG_A_path}")
    assert os.path.exists(netG_A_path), f"Model file not found: {netG_A_path}"
    print(f"[INFO] Successfully loaded: {netG_A_path}")
    model.netG.load_state_dict(torch.load(netG_A_path, map_location=model.device))
    model.netG.eval()

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create results webpage
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('Creating web directory:', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # inference loop
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('Processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize, use_wandb=opt.use_wandb)

    webpage.save()
    print("Testing complete. Results saved.")
