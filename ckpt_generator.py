from genericpath import exists
import os
import shutil
import glob
import torch
from utils.utils import isMode
from utils.parameters import E1_CLASSES, E2_CLASSES, E3_CLASSES
from models.model import E1Model, E1E2Model, E1E2E3Model



def generateHybrid(model_id, mode='r_e1', domain=0, version='v0', in_dim=1761, out_dim=2, root='checkpoints'):

    # if isMode(mode, 'e1_e2_e3'):
    #     dataName = 'E1E2E3Data'
    #     model = E1E2E3Model()
    if isMode(mode, 'e1_e2'):
        dataName = 'E1E2Data'
        classes = [f'{e1_cls}_{e2_cls}' for e1_cls in E1_CLASSES
                   for e2_cls in E2_CLASSES]
        model = E1E2Model(E1_CLASSES, E2_CLASSES, model_id,
                          input_dim=in_dim, out_dim=out_dim)
    elif isMode(mode, 'e1'):
        dataName = 'E1Data'
        classes = E1_CLASSES
        model = E1Model(E1_CLASSES, model_id,
                        input_dim=in_dim, out_dim=out_dim)

    ckpt_dir = os.path.join(
        root, f'domain_{domain}', mode, dataName, str(model_id), version)

    print(f"ckpt dir : {ckpt_dir}")
    if isMode(mode, 'e1_e2'):

        if not os.path.exists(os.path.join(ckpt_dir, 'temp_parts')):
            os.makedirs(os.path.join(ckpt_dir, 'temp_parts'))

        for cls_ in classes:
            # Initialize model
            temp_model = E1E2Model(E1_CLASSES, E2_CLASSES, model_id,
                                   input_dim=in_dim, out_dim=out_dim)

            # Find checkpoint
            files_rx = os.path.join(ckpt_dir, f'e1e2_best_{cls_}_*.pth')
            files = glob.glob(files_rx)
            if len(files) == 0:
                raise ValueError(
                    f"No matching checkpoint found with regex {files_rx}")
            ckpt = files[0]

            # Load checkpoint
            temp_model.load_state_dict(torch.load(
                ckpt, map_location=torch.device('cpu')))

            # Save only class branch
            print(f"Saving {cls_} branch")
            e1_cls, e2_cls = cls_.split('_')
            torch.save(temp_model.model[e1_cls][e2_cls].state_dict(), os.path.join(
                ckpt_dir, 'temp_parts', f'{cls_}.pth'))

        for cls_ in classes:
            e1_cls, e2_cls = cls_.split('_')
            ckpt_name = os.path.join(ckpt_dir, 'temp_parts', f'{cls_}.pth')
            model.model[e1_cls][e2_cls].load_state_dict(torch.load(ckpt_name, map_location=torch.device('cpu')))

        torch.save(model.state_dict(), os.path.join(
            ckpt_dir, 'hybrid.pth'))

        shutil.rmtree(os.path.join(ckpt_dir, 'temp_parts'))

    elif isMode(mode, 'e1'):

        if not os.path.exists(os.path.join(ckpt_dir, 'temp_parts')):
            os.makedirs(os.path.join(ckpt_dir, 'temp_parts'))

        for cls_ in classes:
            # Initialize Model
            temp_model = E1Model(E1_CLASSES, model_id,
                                 input_dim=in_dim, out_dim=out_dim)

            # Find checkpoint
            files_rx = os.path.join(ckpt_dir, f'e1_best_{cls_}_*.pth')
            files = glob.glob(files_rx)
            if len(files) == 0:
                raise ValueError(
                    f"No matching checkpoint found with regex {files_rx}")
            ckpt = files[0]

            # Load checkpoint
            temp_model.load_state_dict(torch.load(
                ckpt, map_location=torch.device('cpu')))

            # Save only class branch
            print(f"Saving {cls_} branch")
            torch.save(temp_model.model[cls_].state_dict(), os.path.join(
                ckpt_dir, 'temp_parts', f'{cls_}.pth'))

        for cls_ in classes:
            ckpt_name = os.path.join(ckpt_dir, 'temp_parts', f'{cls_}.pth')
            model.model[cls_].load_state_dict(torch.load(ckpt_name, map_location=torch.device('cpu')))

        torch.save(model.state_dict(), os.path.join(
            ckpt_dir, 'hybrid.pth'))

        shutil.rmtree(os.path.join(ckpt_dir, 'temp_parts'))


# for model_id in range(1, 9):
#     for version in range(0, 3):
#         generateHybrid(
#             model_id=model_id,
#             mode='r_e1',
#             domain=0,
#             version=f'v{version}',
#             in_dim=1761,
#             out_dim=2,
#             root='/home1/BTP/ds_btp_1/checkpoints/',
#         )
for model_id in range(1, 9):
    for version in range(0, 3):
        generateHybrid(
            model_id=model_id,
            mode='r_e1_e2',
            domain=0,
            version=f'v{version}',
            in_dim=1761,
            out_dim=2,
            root='/home1/BTP/ds_btp_1/checkpoints/',
        )
