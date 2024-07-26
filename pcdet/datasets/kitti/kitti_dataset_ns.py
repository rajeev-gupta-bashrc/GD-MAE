import copy
import pickle
from pathlib import Path
import cv2
import numpy as np

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti, calibration_nuscene
from ..dataset import DatasetTemplate


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.kitti_infos = self.balanced_infos_resampling(self.kitti_infos)

    def include_kitti_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / 'data/nuscenes' / info_path
            if not self.client.exists(info_path):
                continue
            infos = self.client.load_pickle(info_path, True)
            nuscenes_infos.extend(infos['infos'])

        self.kitti_infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        with self.client.get_local_path(split_dir) as path:
            self.sample_id_list = [x.strip() for x in open(path).readlines()]

    def get_lidar(self, idx):
        info = self.kitti_infos[idx]
        lidar_path = self.root_path / info['lidar_path']
        points = self.client.load_to_numpy(str(lidar_path), dtype=np.float32).reshape([-1, 5])[:, :4]
        # store points
        points.tofile(str(self.root_path / f'ns_to_kt/{idx}.bin'))
        return points

    def get_image(self, idx):
        info = self.kitti_infos[idx]
        img_file = self.root_path / info['cams']['CAM_FRONT']['data_path']
        # read image and store 
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        h0, w0 = 384, 1280
        # img = img[(h-h0)//2:900-(h-h0)//2, (w-w0)//2:1600-(w-w0)//2, :]
        cv2.imwrite(str(self.root_path / f'ns_to_kt/{idx}.png'), img)
        
        assert self.client.exists(img_file)
        return self.client.load_img(str(img_file)).astype(np.float32), img_file  # BGR, uint8

    def get_image_shape(self, idx):
        info = self.kitti_infos[idx]
        img_file = self.root_path / info['cams']['CAM_FRONT']['data_path']
        assert self.client.exists(img_file)
        return np.array(self.client.load_img(str(img_file)).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert self.client.exists(label_file)
        return object3d_kitti.get_objects_from_label(label_file, self.client)

    def get_calib(self, index, inverse=True):
        # Ref: https://chatgpt.com/c/004a304d-7667-4b72-bf56-1c1f24caf7a8 https://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        info = self.kitti_infos[index]
        cam_front = info['cams']['CAM_FRONT']
        s2e_t = cam_front['sensor2ego_translation']
        s2e_r = cam_front['sensor2ego_rotation']
        e2g_t = cam_front['ego2global_translation']
        e2g_r = cam_front['ego2global_rotation']
        s2l_t = cam_front['sensor2lidar_translation']
        s2l_r = cam_front['sensor2lidar_rotation']
        cam_intr = cam_front['cam_intrinsic']
        # Tr_velo_to_cam
        sensor2lidar_transform = np.eye(4)
        sensor2lidar_transform[:3, :3] = s2l_r
        sensor2lidar_transform[:3, 3] = s2l_t
        lidar2sensor_transform = np.linalg.inv(sensor2lidar_transform)
        #P2
        if inverse:
            l2s_t = lidar2sensor_transform[:3, 3].reshape(3, 1)
            # P2 = np.hstack((cam_intr, l2s_t))
            P2 = np.hstack((cam_intr, np.zeros((3, 1))))
            P2[0, 3] = P2[0, 3] * (-P2[0, 0])
            P2[1, 3] = P2[1, 3] * (-P2[1, 1])
        else:
            P2 = np.hstack((cam_intr, s2l_t.reshape(3, 1)))
            
        #R0
        R0 = np.eye(3)
        #calib
        calib_dict = {
            'P0': P2,
            'P1': P2,
            'P2': P2,
            'P3': P2,
            'R0': R0,
            'Tr_velo2cam': lidar2sensor_transform[:3, :],
            'Tr_imu2velo': lidar2sensor_transform[:3, :]
        }
        # save calib to file
        f = open(str(self.root_path / f'ns_to_kt/{index}.txt'), 'w')
        for key, val in calib_dict.items():
            f.write(key + ': ')
            f.write(' '.join(map(str, val.flatten())))
            f.write('\n')
        f.close()
        # print(calib_dict)
        return calibration_nuscene.Calibration(calib_dict)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        assert self.client.exists(plane_file)

        with self.client.get_local_path(plane_file) as path:
            lines = open(path).readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        all_db_infos = {}

        infos = self.client.load_pickle(info_path)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]

                self.client.put(gt_points.tobytes(), filepath)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        self.client.dump_pickle(all_db_infos, db_info_save_path)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        points = self.get_lidar(index)
        image, image_path = self.get_image(index)
        img_shape = self.get_image_shape(index)
        
        #user defined
        calib = self.get_calib(index, inverse=True)
        
        input_dict = {
            'frame_id': Path(info['lidar_path']).stem,
            'calib' : calib,
            'points': points,
            'metadata': {'token': info['token']},
            'image' : image, 
            'image_path' : image_path
        }

        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]
            input_dict['points'] = points
        
        # if "calib_matricies" in get_item_list:
        input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)
        
        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        # if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
        #     gt_boxes = data_dict['gt_boxes']
        #     gt_boxes[np.isnan(gt_boxes)] = 0
        #     data_dict['gt_boxes'] = gt_boxes

        # if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
        #     data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    dataset.client.dump_pickle(kitti_infos_train, train_filename)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    dataset.client.dump_pickle(kitti_infos_val, val_filename)
    print('Kitti info val file is saved to %s' % val_filename)

    dataset.client.dump_pickle(kitti_infos_train + kitti_infos_val, trainval_filename)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    dataset.client.dump_pickle(kitti_infos_test, test_filename)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
