
'''
Autor: Muhammed Melih Durmaz
Date: 2021.09.21

This piece of code is trying it's best to estimate the position of the camera,
given a 2D,3D correspondence data and at least 2 images.
It deos provide a 6DOF pose estimate but it s not yet completely succesful.

'''



import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D


'''
This class is not written by me. It is used only for ease of visualisation
Credits to Zongchang (Jim) Chen.
GitHub Link: https://github.com/czcbangkai
'''
class Arrow3D(FancyArrowPatch):
    '''
    3D arrow class can be shown in matplotlib 3D model.
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def visualize3D( camera_pose, camera_orientation, ax, fig):
    '''
    Visualize 3D model with Matplotlib 3D.
    Input:
        image_path: Input image path - string
    Output:
        None -
    '''

    # Decompose the camera coordinate
    arrow_length = camera_pose[2] * 1.5
    xs = [camera_pose[0], camera_pose[0] + camera_orientation[0] * arrow_length]
    ys = [camera_pose[1], camera_pose[1] + camera_orientation[1] * arrow_length]
    zs = [camera_pose[2], camera_pose[2] + camera_orientation[2] * arrow_length]

    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)

    # Plot camera location
    ax.scatter([camera_pose[0]], [camera_pose[1]], [camera_pose[2]])
    arrow = Arrow3D(xs, ys, zs, mutation_scale=5, lw=2, arrowstyle="-|>", color="k")
    ax.add_artist(arrow)

    plt.savefig("output/pose_estimate.png")

    # Show the plots
    plt.show()


def main():
    img = cv2.imread("input/img3.png")
    size = img.shape

    data_2d = np.load('input/vr2d.npy').astype(np.float)

    print(data_2d.shape)

    data_3d = np.load('input/vr3d.npy')

    print(data_3d.shape)

    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(data_3d[:, :, 0], data_3d[:, :, 1], data_3d[:, :, 2])
    ax2.scatter3D(data_3d[:, :, 0], data_3d[:, :, 1], data_3d[:, :, 2])


    # The camera has no distortion
    dist_coeffs = np.zeros((4, 1))


    camera_matrix = np.array([(100.0,   0.0, 960.0), # focal length,            0,           Cx
                              (  0.0, 100.0, 540.0), #            0, focal length,           Cy
                              (  0.0,   0.0,   1.0)  #            0,            0, Aspect Ratio
                              ],
                              dtype="double")


    data_2d = np.reshape(data_2d, (data_2d.shape[0], data_2d.shape[2]))

    print("Solving PnP")
    # Calculate rotation and translation of the scene
    success, rotation_vector, translation_vector = cv2.solvePnP(data_3d, data_2d, camera_matrix, dist_coeffs, flags=0)

    # Convert rotation vectors to matrices
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    translation_matrix = cv2.Rodrigues(translation_vector)[0]

    # Invert scene matrices to get camera rotation and translation matrices
    rotation_matrix_inv     = cv2.invert(rotation_matrix)[1]
    translation_matrix_inv  = cv2.invert(translation_matrix)[1]


    print("Solved PnP")

    # Camera translation vector
    C = np.matmul(-rotation_matrix_inv.transpose(), translation_vector)

    # Camera rotation vector
    O = np.matmul(rotation_matrix_inv.T, np.array([0, 0, 1]).T)

    visualize3D(C, O, ax2, fig2)

    #
    level_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                                    camera_matrix, dist_coeffs)

    # Project camera pose to the image
    point1 = (int(data_2d[0][0]), int(data_2d[0][1]))
    point2 = (int(level_end_point2D[0][0][0]), int(level_end_point2D[0][0][1]))
    cv2.line(img, point1, point2, (255, 255, 255), 2)

    # Display image
    cv2.imshow("Test", img)
    cv2.waitKey()


if __name__ == '__main__':
    main()



