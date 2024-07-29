import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.patches import Polygon
import cv2

# 设置图像的高度 H 和宽度 W 为 128 像素
H, W = 128, 128

def get_cube(center=(0, 0, 2), rotation_angles=[0., 0., 0.], with_normals=False, scale=1.):
    corners = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
    corners = corners - np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3)
    corners = corners * scale
    rot_mat = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix()
    corners = np.matmul(corners, rot_mat.T)
    corners = corners + np.array(center, dtype=np.float32).reshape(1, 3)

    faces = np.array([
        [corners[0], corners[1], corners[3], corners[2]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[0], corners[2], corners[6], corners[4]],
        [corners[-1], corners[-2], corners[-4], corners[-3]],
        [corners[-1], corners[-2], corners[-6], corners[-5]],
        [corners[-1], corners[-3], corners[-7], corners[-5]],
    ])

    if with_normals:
        normals = np.array([(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        normals = np.matmul(normals, rot_mat.T)
        return faces, normals
    else:
        return faces

def get_camera_intrinsics(fx=70, fy=70, cx=W/2., cy=H/2.):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    assert(K.shape == (3, 3) and K.dtype == np.float32)
    return K

def get_perspective_projection(x_c, K):
    assert(x_c.shape == (3,) and K.shape == (3, 3))
    x_s = K @ x_c
    x_s /= x_s[2]
    return x_s[:2]

def project_cube(cube, K):
    s = cube.shape
    assert(s[-1] == 3)
    cube = cube.reshape(-1, 3)
    projected_cube = np.stack([get_perspective_projection(p, K) for p in cube])
    projected_cube = projected_cube.reshape(*s[:-1], 2)
    return projected_cube

def plot_projected_cube(projected_cube, figsize=(5, 5), figtitle=None, colors=None, face_mask=None):
    assert(projected_cube.shape == (6, 4, 2))
    fig, ax = plt.subplots(figsize=figsize)
    if figtitle is not None:
        fig.suptitle(figtitle)
    if colors is None:
        colors = ['C0' for i in range(len(projected_cube))]
    if face_mask is None:
        face_mask = [True for i in range(len(projected_cube))]
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.set_xlabel('Width'), ax.set_ylabel("Height")
    for (cube_face, c, mask) in zip(projected_cube, colors, face_mask):
        if mask:
            ax.add_patch(Polygon(cube_face, color=c))
    plt.savefig('projected_cube.png')

def get_face_color(normal, point_light_direction=(0, 0, 1)):
    assert(normal.shape == (3,))
    point_light_direction = np.array(point_light_direction, dtype=np.float32)
    light_intensity = np.dot(normal, point_light_direction) / (np.linalg.norm(normal) * np.linalg.norm(point_light_direction))
    light_intensity = max(light_intensity, 0)  # clip to 0
    color_intensity = 0.1 + (light_intensity * 0.5 + 0.5) * 0.8
    color = np.stack([color_intensity for i in range(3)])
    return color

def get_face_colors(normals, light_direction=(0, 0, 1)):
    colors = np.stack([get_face_color(normal, light_direction) for normal in normals])
    return colors

def get_face_mask(cube, normals, camera_location=(0, 0, 0)):
    assert(cube.shape == (6, 4, 3) and normals.shape[-1] == 3)
    camera_location = np.array(camera_location).reshape(1, 3)

    face_center = np.mean(cube, axis=1)
    viewing_direction = camera_location - face_center
    dot_product = np.sum(normals * viewing_direction, axis=-1)
    mask = dot_product > 0.0
    return mask

def get_animation(K_list, cube_list, figsize=(5, 5), title=None):
    assert(len(K_list) == len(cube_list))
    cubes = [i[0] for i in cube_list]
    normals = [i[1] for i in cube_list]
    colors = [get_face_colors(normals_i) for normals_i in normals]
    masks = [get_face_mask(cube_i, normals_i) for (cube_i, normals_i) in zip(cubes, normals)]
    projected_cubes = [project_cube(cube, Ki) for (cube, Ki) in zip(cubes, K_list)]
    uv = projected_cubes[0]
    patches = [Polygon(uv_i, closed=True, color='white') for uv_i in uv]

    def animate(n):
        uv = projected_cubes[n]
        color = colors[n]
        mask = masks[n]
        for patch, uv_i, color_i, mask_i in zip(patches, uv, color, mask):
            if mask_i:
                patch.set_xy(uv_i)
                patch.set_color(color_i)
            else:
                uv_i[:] = -80
                patch.set_color(color_i)
                patch.set_xy(uv_i)
        return patches

    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        fig.suptitle(title)
    plt.close()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    for patch in patches:
        ax.add_patch(patch)
    anim = animation.FuncAnimation(fig, animate, frames=len(K_list), interval=100, blit=True)
    return anim

# 投影和绘制一个旋转的立方体
K_list = [get_camera_intrinsics() for i in range(30)]
cube_list = [get_cube(rotation_angles=[0, angle, 0], with_normals=True) for angle in np.linspace(0, 360, 30)]
anim = get_animation(K_list, cube_list, title="Rotation of Cube")
anim.save('rotation_of_cube.mp4', writer='ffmpeg')

# 改变焦距的动画
K_list = [get_camera_intrinsics(fx=f) for f in np.linspace(10, 150, 30)]
cube_list = [get_cube(rotation_angles=(0, 30, 50), with_normals=True) for i in range(30)]
anim = get_animation(K_list, cube_list, title="Change of focal length along the x-axis.")
anim.save('change_of_focal_length.mp4', writer='ffmpeg')

# 加载两张图片并预先计算的点对
img1 = cv2.cvtColor(cv2.imread('./image-1.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./image-2.jpg'), cv2.COLOR_BGR2RGB)

npz_file = np.load('./panorama_points.npz')
points_source = npz_file['points_source']
points_target = npz_file['points_target']

def draw_matches(img1, points_source, img2, points_target):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1])
    output_img[:r1, c:c + c1, :] = np.dstack([img2])

    for p1, p2 in zip(points_source, points_target):
        (x1, y1) = p1[:2]
        (x2, y2) = p2[:2]

        cv2.circle(output_img, (int(x1), int(y1)), 10, (0, 255, 255), 10)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 10, (0, 255, 255), 10)
        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 5)

    plt.imsave('matches.png', output_img)

def get_Ai(xi_vector, xi_prime_vector):
    assert(xi_vector.shape == (3,) and xi_prime_vector.shape == (3,))
    xi = xi_vector
    xi_prime = xi_prime_vector
    Ai = np.array([
        [-xi[0], -xi[1], -1, 0, 0, 0, xi_prime[0]*xi[0], xi_prime[0]*xi[1], xi_prime[0]],
        [0, 0, 0, -xi[0], -xi[1], -1, xi_prime[1]*xi[0], xi_prime[1]*xi[1], xi_prime[1]]
    ], dtype=np.float32)
    assert(Ai.shape == (2, 9))
    return Ai

def get_A(points_source, points_target):
    N = points_source.shape[0]
    A = np.vstack([get_Ai(points_source[i], points_target[i]) for i in range(N)])
    assert(A.shape == (2*N, 9))
    return A

def get_homography(points_source, points_target):
    A = get_A(points_source, points_target)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    assert(H.shape == (3, 3))
    return H

def stich_images(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    return output_img

def get_keypoints(img1, img2):
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    p_source = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good ]).reshape(-1,2)
    p_target = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good ]).reshape(-1,2)
    N = p_source.shape[0]
    p_source = np.concatenate([p_source, np.ones((N, 1))], axis=-1)
    p_target = np.concatenate([p_target, np.ones((N, 1))], axis=-1)
    return p_source, p_target

# 保存匹配的图像
draw_matches(img1, points_source[:5], img2, points_target[:5])

# 生成全景图
p_source, p_target = get_keypoints(img1, img2)
H = get_homography(p_target, p_source)
stiched_image = stich_images(img1, img2, H)
plt.imsave('stitched_panorama.png', stiched_image)
