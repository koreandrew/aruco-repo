# 220204 - rvec, tvec calculation modified -> low computing cost
# 220216 - plot position of markers / roll, yaw, pitch of markers // unit : [mm]

import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot_frame(ids):

    # draw camera
    ax.scatter(0, 0, 0, color="k")
    ax.quiver(0, 0, 0, 1, 0, 0, length=20, color="r")
    ax.quiver(0, 0, 0, 0, 1, 0, length=20, color="g")
    ax.quiver(0, 0, 0, 0, 0, 1, length=20, color="b")
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0, 0, 0, 0, 0], color="k", linestyle=":")

    # ax.set_xlim(-400, 100);
    # ax.set_ylim(100, 400);
    # ax.set_zlim(-400, -100)
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")

    for i in range(len(ids)):
        [x, y, z] = marker_pos_temp[ids[i]][-1]
        ax.text(x, y, z, '%s' % ("ID : " + str(ids[i])), size=10, color='r')
        ux, vx, wx = V_x[ids[i]]
        uy, vy, wy = V_y[ids[i]]
        uz, vz, wz = V_z[ids[i]]

        # draw markers
        ax.quiver(x, y, z, ux, vx, wx, length=20, color="r")
        ax.quiver(x, y, z, uy, vy, wy, length=20, color="g")
        ax.quiver(x, y, z, uz, vz, wz, length=20, color="b")

    # fig.canvas.draw()
    # fig.canvas.flush_events()
    plt.pause(0.05)
    ax.clear()


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to detect")
# args = vars(ap.parse_args())

aruco_dict = cv.aruco.DICT_4X4_50
type_aruco = "DICT_4X4_50"
# ids_in_use = 5 # 사용하는 id 개수
cam_pos_x = 0
cam_pos_y = 0
cam_pos_z = 0
cam_pos_dict = {}
cam_pos_temp = {}
marker_pos_dict = {}
marker_pos_temp = {}
rpy_dict = {}
V_x = {}
V_y = {}
V_z = {}

marker_length = 16  # unit :[m]
axis_length = 10
matrix_coefficients = np.load("C:/Users/RimLAB/Desktop/Computer_Vision/aruco_marker/cam_calib_data/mtx_220208.npy")
distortion_coefficients = np.load("C:/Users/RimLAB/Desktop/Computer_Vision/aruco_marker/cam_calib_data/dist_220208.npy")

# print("[INFO] detecting '{}' tags...".format(args["type"]))
print("[INFO] detecting '{}' markers...".format(type_aruco))
# arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoDict = cv.aruco.getPredefinedDictionary(aruco_dict)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
# check frame width, height
# w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# print(w, h)

# frame setting
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

time.sleep(1.0)

# checking time
start_time = time.time()

# setting for marker plot
plt.ion()  # plot 갱신 // ion : interactive on
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
fig.add_axes(ax)
ax.view_init(elev=-45, azim=-45)
# ax.set_xlim(-0.5, 0.5);ax.set_ylim(-0.5, 0.5);ax.set_zlim(-0.5, 0.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


# loop over the imgs from the video stream
while True:
    ret, img = cap.read()
    # img = imutils.resize(img, width=1000)
    img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # blur = cv.GaussianBlur(img_bw, (5, 5), 0)
    blur = cv.bilateralFilter(img_bw, 5, 10, 10)

    # ret, img_bw = cv.threshold(blur, 125, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # ret, img_bw = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # cv.THRESH_BINARY_INV : 이미지 반전(흑<->백)
    # img_bw = cv.ximgproc.niBlackThreshold(blur, 255, cv.THRESH_BINARY, 31, 0)
    img_bw = cv.adaptiveThreshold(img_bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 0)

    # detect ArUco markers in the input img
    arucoParams = cv.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv.aruco.detectMarkers(img_bw, arucoDict, parameters=arucoParams)
    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
    if ids is not None:
        ids = ids.flatten()

        rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners, marker_length,
                                                                      matrix_coefficients,
                                                                      distortion_coefficients)
        for i in range(0, len(ids)):  # Iterate in markers
            # cv2.aruco.estimatePoseSingleMarkers(corners, marker length[m], camera matrix, distortion coeff)

            (rvec[i] - tvec[i]).any()  # get rid of that nasty numpy value array error

            R = np.zeros(shape=(3, 3))  # R : rotation matrix => R^T = R^(-1)
            cv.Rodrigues(rvec[i][0], R, jacobian=0)
            # print(rvec[i])
            # print(R)
            T = tvec[i][0].T  # T : translation vector(center of marker)

            cam_pos = np.dot(R.T, -T).squeeze()
            cam_pos_temp[ids[i]] = [cam_pos]
            # print("MarkerID {} : {} {} {}".format(ids[i], cam_pos[0], cam_pos[1], cam_pos[2]))

            marker_pos = tvec[i][0]
            marker_pos_temp[ids[i]] = [marker_pos]
            print("id_"+str(ids[i])+" = "+"["+str(marker_pos[0])+","+str(marker_pos[1])+","+str(marker_pos[2])+"]"+";")

            if ids[i] in marker_pos_dict:
                marker_pos_dict[ids[i]].append([marker_pos[0], marker_pos[1], marker_pos[2]])
            else:
                marker_pos_dict[ids[i]] = [[marker_pos[0], marker_pos[1], marker_pos[2]]]

            # r = np.arctan2(-R[2][1], R[2][2])
            # p = np.arcsin(R[2][0])
            # y = np.arctan2(-R[1][0], R[0][0])
            # rpy = - np.array([r, p, y])
            rpy = np.deg2rad(cv.RQDecomp3x3(R.T)[0]).squeeze()
            rpy_dict[ids[i]] = [rpy]

            V_x[ids[i]] = np.dot(R.T, np.array([1, 0, 0]))
            V_y[ids[i]] = np.dot(R.T, np.array([0, 1, 0]))
            V_z[ids[i]] = np.dot(R.T, np.array([0, 0, 1]))

            cv.aruco.drawAxis(img, matrix_coefficients, distortion_coefficients, rvec[i], tvec[i], axis_length)

        cv.aruco.drawDetectedMarkers(img, corners) # Draw A square around the markers
        marker_pos_dict = dict(sorted(marker_pos_dict.items()))

        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv.line(img, topLeft, topRight, (0, 255, 0), 2)
            cv.line(img, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the img
            cv.putText(img, "ID=" + str(markerID), (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 2)
            # print("ID : {} center x,y : ({},{})".format(markerID,cX,cY))

    if ids is not None:
        plot_frame(ids)

    cv.imshow("Gray Image", img_bw)
    cv.imshow("Image", img)

    key = cv.waitKey(1) & 0xFF

    # press 'esc' to break
    if key == 27:
        break

end_time = time.time()
print("time : {}".format(end_time - start_time))  # unit [sec]

plt.close()
cap.release()
cv.destroyAllWindows()
