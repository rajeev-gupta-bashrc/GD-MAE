import math
import numpy as np
import matplotlib.pyplot as plt 


class Calibration(object):
    def __init__(self, calib_dict):
        if not isinstance(calib_dict, dict):
            raise ValueError('Expected a dictionary')
        else:
            calib = calib_dict

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def img_to_lidar(self, pts_img, pts_depth):
        pts_rect = self.img_to_rect(pts_img[:, 0], pts_img[:, 1], pts_depth)
        pts_lidar = self.rect_to_lidar(pts_rect)
        return pts_lidar
    
    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner




def encircle_pixel_in_image(image_arr, posn, rgbd, radii, print_complexity=False):
    comp = 0
    h_, k_ = posn
    inc = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    ## Algo 1:
    for _ in inc:
        inc_h, inc_k = _
        h = h_
        while(abs(h-h_)<=radii):
            k = k_
            while (abs(k-k_)<=radii) and (math.sqrt((h-h_)**2 + (k-k_)**2) <= radii) and (h<image_arr.shape[0]) and h>=0 and (k<image_arr.shape[1]) and k>=0:
                image_arr[h, k] = rgbd
                comp += 1
                k+=inc_k
            h+=inc_h
    ## Algo 2:
    # i, j = 0, 0
    # while i <= radii:
    #     j=0
    #     while j <= radii:
    #         for _ in inc: 
    #             inc_h, inc_k = _
    #             h = h_+i*inc_h
    #             k = k_+j*inc_k
    #             comp+=1
    #             if (math.sqrt(i**2 + j**2) <= radii) and (h<image_arr.shape[0]) and h>=0 and (k<image_arr.shape[1]) and k>=0:
    #                 image_arr[h, k] = rgb
    #             else: continue
    #         j+=1
    #     i+=1
    
    if print_complexity: print('complexity of algorithm ', comp)
    return image_arr, comp
    
    
def point_cloud_to_image(get_dict, point_radii = 4, to_show=True, bg_shade=[0, 0, 0], fill_depth_as_pixel_value=False):
    points = get_dict['points']
    calib = get_dict['calib'][0] #calibration object
    im_shape = get_dict['image_rescale_shape'][0]
    return _point_cloud_to_image(calib, points, im_shape, point_radii, to_show, bg_shade, fill_depth_as_pixel_value)
    
def _point_cloud_to_image(calib, points, im_shape, point_radii, to_show=True, bg_shade=[0, 0, 0], fill_depth_as_pixel_value=False):
    # expected format == [x, y, z, intensity]
    if points.shape[1] == 5: points = points[:, 1:]
    if points.shape[1] == 4: 
        reflactance = points[:, 3]
        pts_img, pts_depth = calib.lidar_to_img(points[:, :3])
        # get min max of pts_depth
        min_depth = min(pts_depth)
        max_depth = max(pts_depth)
        print(pts_img.shape, min(pts_img[:, 0]), max(pts_img[:, 0]), min(pts_img[:, 1]), max(pts_img[:, 1]))
        # create image arr
        image_arr = np.zeros(tuple(np.append(im_shape, 4)))                 #shape = (w, h, 4) for r,g,b,depth
        image_arr[:, :, :3] = bg_shade
        image_arr.astype(float)                                             #includes depth data
        # image_pt[:, :, :] = 255
        # get indices
        ijs = pts_img.astype(int)
        for index, ij in enumerate(ijs):
            j, i = ij                                   #(h, w) for h:row, w:column
            pixel_value = reflactance[index]
            # pixel_value = 255
            if fill_depth_as_pixel_value:
                pixel_value = 1-(pts_depth[index] - min_depth) / (max_depth - min_depth)
            image_arr, num_colored_pixels = encircle_pixel_in_image(image_arr, [i, j], [pixel_value, 0, 0, pts_depth[index]], point_radii)
            
        if to_show:
            plt.imshow(image_arr[:, :, :3])
            plt.axis('off')  # Optional: turn off axis
            plt.show()
        return image_arr
    else:
        print('points aren\'t in expected (N, 4 or 5) format, got shape ', points.shape)

        
def image_to_point_cloud(get_dict, extrapolation_radii=0):
    image_arr = point_cloud_to_image(get_dict, point_radii = extrapolation_radii, to_show=False)
    calib = get_dict['calib'][0]
    return _image_to_point_cloud(calib, image_arr)


def _image_to_point_cloud(calib, image_arr):
    non_zero_indexes = np.argwhere(image_arr != 0)[:, :2]
    # print(non_zero_indexes.shape, ' -shape of non_zero_indexes')
    non_zero_indexes = np.unique(non_zero_indexes, axis=0)
    # print(non_zero_indexes.shape, ' -shape of highlighted points in image')
    # print(non_zero_indexes)
    pts_img_depth = np.zeros((non_zero_indexes.shape[0], 3), dtype=int)
    pts_img_depth[:, :2] = non_zero_indexes
    pts_img_depth[:, 2] = image_arr[non_zero_indexes[:, 0], non_zero_indexes[:, 1], 3]
    pts_img_depth[:, [0, 1]] = pts_img_depth[:, [1, 0]]                                 #(h, w) for h:row, w:column
    pts_img = pts_img_depth[:, :2]
    pts_depth = pts_img_depth[:, 2]
    print(pts_img.shape, ' -shape of pts_image')
    pts_lidar = calib.img_to_lidar(pts_img, pts_depth)
    return pts_lidar