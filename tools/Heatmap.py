import argparse
import os.path
import warnings
from functools import partial

import cv2
import torch
import mmcv
import numpy as np
from mmengine import Config, DictAction, MessageHub
from mmengine.utils import ProgressBar

try:
    from pytorch_grad_cam import AblationCAM, EigenCAM
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      'pytorch_grad_cam package.')

from mmrotate.models import build_detector
from mmcv.runner import load_checkpoint
from mmyolo.utils.boxam_utils import (DetAblationLayer, reshape_transform)
from mmyolo.utils.misc import get_file_list
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus

GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
}
GRAD_BASED_METHOD_MAP = {'gradcam': GradCAM, 'gradcam++': GradCAMPlusPlus}
ALL_SUPPORT_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASED_METHOD_MAP.keys())

message_hub = MessageHub.get_current_instance()
message_hub.runtime_info['epoch'] = 0

def dota_letterbox(im, new_shape=(1024, 1024), color=(114, 114, 114), auto=True, stride=32):
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (left, top)

class MARNetDOTAWrapper:
    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        self.device = torch.device(device)
        self.score_thr = score_thr
        self.dota_input_shape = (1024, 1024)  

        self.detector =build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        load_checkpoint(self.detector, checkpoint, map_location=self.device)
        self.detector.to(self.device).eval()

        self.need_loss = False
        self.image = None
        self.pred_instances = None
        self.scale_ratio = 1.0
        self.pad_offset = (0, 0)

    def set_input_data(self, image, pred_instances=None):
        self.image = image.copy()
        self.pred_instances = pred_instances

        img_resized, self.scale_ratio, self.pad_offset = dota_letterbox(
            self.image, new_shape=self.dota_input_shape
        )
        return self

    def __call__(self):
        img_resized, _, _ = dota_letterbox(self.image, new_shape=self.dota_input_shape)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255.0

        img_metas = [{
            'img_shape': img_resized.shape[:2],
            'ori_shape': self.image.shape[:2],
            'pad_shape': img_resized.shape[:2],
            'scale_factor': self.scale_ratio,
            'pad_offset': self.pad_offset, 
            'flip': False,
            'img_norm_cfg': {
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True
            }
        }]

        if self.need_loss:
            loss = self.detector(img=[img_tensor], img_metas=[img_metas], return_loss=True)
            total_loss = sum(loss.values())
            return total_loss
        else:
            with torch.no_grad():
                results = self.detector(img=[img_tensor], img_metas=[img_metas], return_loss=False)
            return [results[0]] if len(results) > 0 else [None]

    def need_loss(self, flag):
        self.need_loss = flag
        for param in self.detector.parameters():
            param.requires_grad_(flag)
        return self

class MARNetDOTACAMVisualizer:
    def __init__(self, method_class, model_wrapper, target_layers, reshape_transform=None, is_need_grad=False, extra_params=None):
        self.method_class = method_class
        self.model_wrapper = model_wrapper
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad
        self.extra_params = extra_params or {}
        self.cam = self._init_cam()

    def _init_cam(self):
        cam_kwargs = {
            'model': self.model_wrapper.detector,
            'target_layers': self.target_layers,
            'use_cuda': self.model_wrapper.device.type == 'cuda'
        }
        if self.reshape_transform:
            cam_kwargs['reshape_transform'] = self.reshape_transform
        if self.extra_params:
            cam_kwargs.update(self.extra_params)
        return self.method_class(**cam_kwargs)

    def switch_activations_and_grads(self, model_wrapper):
        pass

    def __call__(self, image, targets=None):
        img_resized, _, _ = dota_letterbox(image, new_shape=self.model_wrapper.dota_input_shape)
        img_float = np.float32(img_resized) / 255.0
        input_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(self.model_wrapper.device)

        if self.is_need_grad:
            input_tensor.requires_grad_(True)

        if targets is None:
            grayscale_cam = self.cam(input_tensor=input_tensor)
        else:
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        grayscale_cam = cv2.resize(grayscale_cam[0], (image.shape[1], image.shape[0]))
        return [grayscale_cam]

    def show_am(self, image, pred_instances, grayscale_boxam, with_norm_in_bboxes=True):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        cam_image = self._overlay_dota_cam(img_float, grayscale_boxam[0])
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        scale_ratio = self.model_wrapper.scale_ratio
        pad_left, pad_top = self.model_wrapper.pad_offset

        for bbox, score in zip(pred_instances['bboxes'], pred_instances['scores']):
            x, y, w, h, angle = bbox

            x = (x - pad_left) / scale_ratio
            y = (y - pad_top) / scale_ratio
            w = w / scale_ratio
            h = h / scale_ratio

            rect = cv2.RotatedRect((x, y), (w, h), angle)
            box_points = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(cam_image, [box_points], 0, (0, 255, 0), 2)
            cv2.putText(
                cam_image,
                f'score:{score:.2f}',
                (int(x)-50, int(y)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 
                (0, 255, 0),
                2
            )
        return cam_image

    def _overlay_dota_cam(self, img_float, grayscale_cam):
        grayscale_cam = np.uint8(255 * grayscale_cam)
        grayscale_cam = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_VIRIDIS)
        grayscale_cam = np.float32(grayscale_cam) / 255.0

        cam_image = (1 - 0.3 * grayscale_cam) * img_float + 0.3 * grayscale_cam
        cam_image = np.uint8(255 * np.clip(cam_image, 0, 1))
        return cam_image

class MARNetDOTABoxScoreTarget:
    def __init__(self, pred_instances, device='cuda:0', ignore_loss_params=None):
        self.pred_instances = pred_instances
        self.device = device

    def __call__(self, model_output):
        if self.pred_instances is None or len(self.pred_instances) == 0:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        valid_scores = self.pred_instances['scores'][self.pred_instances['scores'] > 0.25]
        return valid_scores.sum() if len(valid_scores) > 0 else torch.tensor(0.0, requires_grad=True).to(self.device)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MARNet Heatmap (DOTA Dataset)')
    parser.add_argument(
        'img', help='DOTA image path, include image file, dir.')
    parser.add_argument('config', help='MARNet DOTA config file (mmrotate format)')
    parser.add_argument('checkpoint', help='MARNet DOTA checkpoint file')
    parser.add_argument(
        '--method',
        default='gradcam',
        choices=ALL_SUPPORT_METHODS,
        help='Type of method to use, supports '
        f'{", ".join(ALL_SUPPORT_METHODS)}.')
    parser.add_argument(
        '--target-layers',
        default=['backbone.stages.6'],
        nargs='+',
        type=str,
        help='The target layers to get heatmap (e.g. backbone.stages.6)')
    parser.add_argument(
        '--out-dir', default='./marnet_dota_heatmap', help='Path to save DOTA heatmap')
    parser.add_argument(
        '--show', action='store_true', help='Show the DOTA heatmap results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference (cuda:0 / cpu)')
    parser.add_argument(
        '--score-thr', type=float, default=0.25, help='DOTA bbox score threshold (lower for small targets)')
    parser.add_argument(
        '--topk',
        type=int,
        default=-1,
        help='Select topk predict results to show. -1 means all (for DOTA multi-target)')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=1024,
        help='DOTA input shape (default 1024x1024)')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--norm-in-bbox', action='store_true', help='Norm in bbox of DOTA am image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config.')
    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM (reduce for DOTA large images)')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM for DOTA.')

    args = parser.parse_args()
    return args

def init_detector_and_visualizer(args, cfg):
    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [max_shape, max_shape]
    assert len(max_shape) == 2, "DOTA max_shape must be (h, w)"

    model_wrapper = MARNetDOTAWrapper(
        cfg, args.checkpoint, args.score_thr, device=args.device
    )
    model_wrapper.dota_input_shape = (max_shape[0], max_shape[1])

    if args.preview_model:
        print(model_wrapper.detector)
        print('\n Please remove `--preview-model` to get the DOTA BoxAM.')
        return None, None

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model_wrapper.detector.{target_layer}'))
        except Exception as e:
            print(f"Model structure:\n{model_wrapper.detector}")
            raise RuntimeError(f"Target layer {target_layer} does not exist", e)

    ablationcam_extra_params = {
        'batch_size': args.batch_size,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': args.ratio_channels_to_ablate
    }

    if args.method in GRAD_BASED_METHOD_MAP:
        method_class = GRAD_BASED_METHOD_MAP[args.method]
        is_need_grad = True
    else:
        method_class = GRAD_FREE_METHOD_MAP[args.method]
        is_need_grad = False

    visualizer = MARNetDOTACAMVisualizer(
        method_class,
        model_wrapper,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=ablationcam_extra_params)
    return model_wrapper, visualizer

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if not os.path.exists(args.out_dir) and not args.show:
        os.makedirs(args.out_dir, exist_ok=True)

    model_wrapper, visualizer = init_detector_and_visualizer(args, cfg)
    if model_wrapper is None:
        return

    image_list, source_type = get_file_list(args.img)
    if len(image_list) == 0:
        raise RuntimeError("No DOTA images found in the specified path.")

    progress_bar = ProgressBar(len(image_list))

    for image_path in image_list:
        image = cv2.imread(image_path)
        if image is None:
            warnings.warn(f"Failed to read DOTA image: {image_path}, skip this.")
            progress_bar.update()
            continue

        model_wrapper.set_input_data(image)
        result = model_wrapper()[0]

        if result is None or not hasattr(result, 'pred_instances'):
            warnings.warn(f"Empty detection results for DOTA image: {image_path}, skip this.")
            progress_bar.update()
            continue

        pred_instances = result.pred_instances
        
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]

        if len(pred_instances) == 0:
            warnings.warn(f"No valid bboxes for DOTA image: {image_path}, skip this.")
            progress_bar.update()
            continue

        if args.topk > 0:
            pred_instances = pred_instances[:args.topk]

        pred_instances_np = {
            'bboxes': pred_instances.bboxes.numpy(),
            'scores': pred_instances.scores.numpy()
        }
        targets = [
            MARNetDOTABoxScoreTarget(
                pred_instances_np,
                device=args.device
            )
        ]

        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(True)
            model_wrapper.set_input_data(image, pred_instances)
            visualizer.switch_activations_and_grads(model_wrapper)

        try:
            grayscale_boxam = visualizer(image, targets=targets)
        except Exception as e:
            warnings.warn(f"Failed to generate heatmap for DOTA image: {image_path}, error: {e}")
            progress_bar.update()
            continue

        image_with_heatmap = visualizer.show_am(
            image,
            pred_instances_np,
            grayscale_boxam,
            with_norm_in_bboxes=args.norm_in_bbox)

        if source_type['is_dir']:
            filename = os.path.relpath(image_path, args.img).replace('/', '_')
        else:
            filename = os.path.basename(image_path)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        if out_file:
            mmcv.imwrite(image_with_heatmap, out_file)
        else:
            cv2.namedWindow(filename, 0)
            cv2.imshow(filename, image_with_heatmap)
            cv2.waitKey(0)

        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(False)
            visualizer.switch_activations_and_grads(model_wrapper)

        progress_bar.update()

    if not args.show:
        print(f'All done! DOTA heatmap results have been saved at {os.path.abspath(args.out_dir)}')

if __name__ == '__main__':
    main()