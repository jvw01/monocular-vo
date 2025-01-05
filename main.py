import os
from time import sleep
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_frame import processFrame
import matplotlib.transforms as transforms

bootstrap = True # boolean variable to determine if we are bootstrapping or not
low_keypoint_plot_threshold = 200
import matplotlib.pyplot as plt
from test_init.init import initialization_cv2

# Setup
dataset = 0 # 0: KITTI, 1: Malaga, 2: parking, 3: test
trailing_trajectory_plot = False
parking_path = "" #"data/parking/images/"
malaga_path = "" #"data/malaga/Images/"
kitti_path = "" #"data/kitti/image_0/"
test_path = "" # for testing purposes

params = {} # dict for passing parameters to processFrame

if dataset == 0:
    assert 'kitti_path' in locals()
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'data/kitti/poses/05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    last_frame = 2761
    K = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])

    # set parameters for VO pipeline
    params["K"] = K
    params["L_m"] = 0
    params["min_depth"] = 1
    params["max_depth"] = 100
    angle_threshold_for_triangulation = 1 # in degrees
    params["angle_threshold_for_triangulation"] = angle_threshold_for_triangulation * np.pi / 180 # convert to radians
    params["distance_threshold"] = 1 # threshold for sorting out duplicate new keypoints

elif dataset == 1:
    assert 'malaga_path' in locals()
    images = sorted(os.listdir(os.path.join(malaga_path, 
                'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/')))
    left_images = images[0::2]
    last_frame = 1580
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])

    # set parameters for VO pipeline
    params["K"] = K
    params["L_m"] = 3
    params["min_depth"] = 1
    params["max_depth"] = 120
    angle_threshold_for_triangulation = 1 # in degrees
    params["angle_threshold_for_triangulation"] = angle_threshold_for_triangulation * np.pi / 180 # convert to radians
    params["distance_threshold"] = 2 # threshold for sorting out duplicate new keypoints

elif dataset == 2:
    assert 'parking_path' in locals()
    last_frame = 530
    K = np.array([
        [331.37, 0, 320],
        [0, 369.568, 240],
        [0, 0, 1]
    ])
    
    ground_truth = np.loadtxt(os.path.join(parking_path, 'data/parking/poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]

    # set parameters for VO pipeline
    params["K"] = K
    params["L_m"] = 1
    params["min_depth"] = 0
    params["max_depth"] = 150
    angle_threshold_for_triangulation = 3 # in degrees
    params["angle_threshold_for_triangulation"] = angle_threshold_for_triangulation * np.pi / 180 # convert to radians
    params["distance_threshold"] = 2 # threshold for sorting out duplicate new keypoints

elif dataset == 3:
    assert 'kitti_path' in locals()
    last_frame = 100
    K = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])

    # set parameters for VO pipeline (tuned)
    params["K"] = K
    params["L_m"] = 2
    params["min_depth"] = 1
    params["max_depth"] = 80
    angle_threshold_for_triangulation = 4 # in degrees
    params["angle_threshold_for_triangulation"] = angle_threshold_for_triangulation * np.pi / 180 # convert to radians
    params["distance_threshold"] = 3 # threshold for sorting out duplicate new keypoints

else:
    raise AssertionError("Invalid dataset selection for bootstrapping")

if bootstrap:
    # Load frames to perform bootstrapping on
    bootstrap_frames = [0, 4] #4

    if dataset == 0:
        img0 = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', 
                        f"{bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', 
                        f"{bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    elif dataset == 1:
        img0 = cv2.imread(os.path.join(malaga_path, 
                        'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                        left_images[bootstrap_frames[0]]), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(malaga_path, 
                        'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                        left_images[bootstrap_frames[1]]), cv2.IMREAD_GRAYSCALE)
    elif dataset == 2:
        img0 = cv2.imread(os.path.join(parking_path, 
                        f"data/parking/images/img_{bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(parking_path, 
                        f"data/parking/images/img_{bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    else:
        raise AssertionError("Invalid dataset selection")
    
    key_points, p_W_landmarks_normalized = initialization_cv2(img0,img1,dataset, K, verbosity=0)
    
    # check datatype of key_points and p_W_landmarks
    if key_points.dtype != np.float32:
        key_points = key_points.astype(np.float32)
    if p_W_landmarks_normalized.dtype != np.float32:
        p_W_landmarks_normalized = p_W_landmarks_normalized.astype(np.float32)

    # Save into the dictionary with normalized landmarks
    S_prev = {
        "keypoints": key_points.T, # dim: Kx2
        "landmarks": p_W_landmarks_normalized.T, # dim: Kx3
        "candidate_keypoints": None,
        "first_observations": None,
        "pose_at_first_observation": None
    }

else:
    # Circumvent bootstrapping by loading precomputed bootstrapping data
    if dataset == 3:
        key_points = np.loadtxt(os.path.join(test_path, 'test_data/keypoints.txt'), dtype=np.float32) # note: cv2.calcOpticalFlowPyrLK expects float32
        p_W_landmarks = np.loadtxt(os.path.join(test_path, 'test_data/p_W_landmarks.txt'), dtype=np.float32)
        img0 = cv2.imread(os.path.join(test_path, f"test_data/image_0/000000.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(test_path, f"test_data/image_0/000001.png"), cv2.IMREAD_GRAYSCALE)
        
        # Swap columns of keypoints
        key_points[:, [1, 0]] = key_points[:, [0, 1]]

        S_prev = {
                    "keypoints": key_points, # dim: Kx2
                    "landmarks": p_W_landmarks, # dim: Kx3
                    "candidate_keypoints": None, # no candidate keypoints in the beginning
                    "first_observations": None, # no candidate keypoints in the beginning
                    "pose_at_first_observation": None # no candidate keypoints in the beginning
                }
    else:
        raise NotImplementedError(f'Pre-computed bootstrapping values not available for this dataset.')

# CONTINUOUS OPERATION
if bootstrap:
    range_frames = range(bootstrap_frames[1] + 1, last_frame)
else:
    range_frames = range(1, 100) # Hardcode this because we don't have the rest of the dataset available right now
img_prev = img0

trajectory = np.zeros((3, len(range_frames)))
trajectory_points_with_low_keypoints = []
indices_with_low_keypoints = []
n_tracked_keypoints_list = []
n_promotable_keypoints_before_angle_filtering_list = []
n_promoted_keypoints_list = []
n_lost_candidates_at_angle_filtering = []
n_lost_candidates_at_cartesian_mask = []
n_new_candidate_keypoints_list = []

# Visualisation of VO pipeline
fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.set_axis_off()
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))

ax1.set_title('Keypoints')  # Set the title of the figure
img_plot = ax1.imshow(img0, cmap='gray')
untrackable_keypoints_plot, = ax1.plot([], [], 'ro', markersize=3, label='Untrackable KP')
trackable_outlier_keypoints_plot, = ax1.plot([], [], 'o', color='orange', markersize=3, label='Trackable outlier KP')
trackable_keypoints_plot, = ax1.plot([], [], 'go', markersize=3, label='Trackable KP')

untrackable_candidate_keypoints_plot, = ax1.plot([], [], 'rx', markersize=3, label='Untrackable CKP')
trackable_unpromotable_candidate_keypoints_plot, = ax1.plot([], [], 'bo', markersize=3, label='Trackable unpromotable CKP')
untriangulatable_promotable_candidate_keypoints_plot, = ax1.plot([], [], 'cx', markersize=3, label='Untriangulatable promotable CKP')
promotable_candidate_keypoints_outside_thresholds_plot, = ax1.plot([], [], 'mx', markersize=3, label='Promotable CKP outside_thresholds')
promoted_candidate_keypoints_plot, = ax1.plot([], [], 'y*', markersize=3, label=' Promoted CKP')
new_candidate_keypoints_plot, = ax1.plot([], [], 'b+', markersize=3, label=' New CKP')
candidate_keypoints_duplicate_with_keypoints_plot, = ax1.plot([], [], 'r^', markersize=2, label='CKP duplicate w KP')
candidate_keypoints_duplicate_with_prev_candidate_keypoints_plot, = ax1.plot([], [], 'rv', markersize=2, label=' CKP duplicate w prev CKP')
ax1.legend()

ax2.set_title('Estimated Trajectory')
landmarks_plot, = ax2.plot([], [], 'ro', markersize=3, label='Landmarks')
trajectory_plot, = ax2.plot([], [], 'b-o', markersize=3, label='Trajectory')
trajectory_points_with_low_keypoints_plot, = ax2.plot([], [], "o", color="orange", markersize=5, label=f'low keypoints (<{low_keypoint_plot_threshold})', linestyle='None')
ax2.legend()

ax3.set_title('Number of tracked keypoints (last 20 frames)', y=-0.22)
keypoints_plot, = ax3.plot([], [])

if not trailing_trajectory_plot:
    if dataset == 0:
        ax2.set_aspect('equal',adjustable='box')
        ax2.set_xlim(-100, 500)
        ax2.set_ylim(-100, 200)
    elif dataset == 1:
        ax2.set_aspect('equal',adjustable='box')
        ax2.set_xlim(-200, 10)
        ax2.set_ylim(0, 150)
    elif dataset == 2:
        ax2.set_aspect('equal',adjustable='box')
        ax2.set_xlim(0, 1400)
        ax2.set_ylim(-700, 100)

for index, i in enumerate(range_frames):
    print(f"\n\nProcessing frame {i}\n{'=' * 21}\n")
    if dataset == 0:
        img = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
    elif dataset == 1:
        img = cv2.imread(os.path.join(malaga_path, 
                        'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                        left_images[i]), cv2.IMREAD_GRAYSCALE)
    elif dataset == 2:
        img = cv2.imread(os.path.join(parking_path, 
                        f"data/parking/images/img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
    elif dataset == 3:
        img = cv2.imread(os.path.join(test_path, 'test_data/image_0/', f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)

    else:
        raise AssertionError("Invalid dataset selection")
    
    S, T_WC, debug_dict = processFrame(img, img_prev, S_prev, params)
    img_prev = img
    S_prev = S
    trajectory[:, index] = T_WC[:, 3]
    n_tracked_keypoints_list += [debug_dict["n_tracked_keypoints"]]
    n_promotable_keypoints_before_angle_filtering_list += [debug_dict["n_promotable_keypoints_before_angle_filtering"]]
    n_promoted_keypoints_list += [debug_dict["n_promoted_keypoints"]]
    n_lost_candidates_at_angle_filtering += [debug_dict["n_lost_candidates_at_angle_filtering"]]
    n_lost_candidates_at_cartesian_mask += [debug_dict["n_lost_candidates_at_cartesian_mask"]]
    n_new_candidate_keypoints_list += [len(debug_dict["new_candidate_keypoints"])]

    # Update the plots
    img_plot.set_data(img)
    untrackable_keypoints_plot.set_data(debug_dict["untrackable_keypoints"][:, 0], debug_dict["untrackable_keypoints"][:, 1])
    trackable_outlier_keypoints_plot.set_data(debug_dict["trackable_outlier_keypoints"][:, 0], debug_dict["trackable_outlier_keypoints"][:, 1])
    trackable_keypoints_plot.set_data(debug_dict["trackable_keypoints"][:, 0], debug_dict["trackable_keypoints"][:, 1])
    if S["candidate_keypoints"] is not None:
        untrackable_candidate_keypoints_plot.set_data(debug_dict["untrackable_candidate_keypoints"][:, 0], debug_dict["untrackable_candidate_keypoints"][:, 1])
        trackable_unpromotable_candidate_keypoints_plot.set_data(debug_dict["trackable_unpromotable_candidate_keypoints"][:, 0], debug_dict["trackable_unpromotable_candidate_keypoints"][:, 1])
        untriangulatable_promotable_candidate_keypoints_plot.set_data(debug_dict["untriangulatable_promotable_candidate_keypoints"][:, 0], debug_dict["untriangulatable_promotable_candidate_keypoints"][:, 1])
        promotable_candidate_keypoints_outside_thresholds_plot.set_data(debug_dict["promotable_candidate_keypoints_outside_thresholds"][:, 0], debug_dict["promotable_candidate_keypoints_outside_thresholds"][:, 1])
        promoted_candidate_keypoints_plot.set_data(debug_dict["promoted_candidate_keypoints"][:, 0], debug_dict["promoted_candidate_keypoints"][:, 1])
        new_candidate_keypoints_plot.set_data(debug_dict["new_candidate_keypoints"][:, 0], debug_dict["new_candidate_keypoints"][:, 1])
        candidate_keypoints_duplicate_with_keypoints_plot.set_data(debug_dict["candidate_keypoints_duplicate_with_keypoints"][:, 0], debug_dict["candidate_keypoints_duplicate_with_keypoints"][:, 1])
        candidate_keypoints_duplicate_with_prev_candidate_keypoints_plot.set_data(debug_dict["candidate_keypoints_duplicate_with_prev_candidate_keypoints"][:, 0], debug_dict["candidate_keypoints_duplicate_with_prev_candidate_keypoints"][:, 1])
    else:
        untrackable_candidate_keypoints_plot.set_data([], [])
        trackable_unpromotable_candidate_keypoints_plot.set_data([], [])
        untriangulatable_promotable_candidate_keypoints_plot.set_data([], [])
        promotable_candidate_keypoints_outside_thresholds_plot.set_data([], [])
        promoted_candidate_keypoints_plot.set_data([], [])
        new_candidate_keypoints_plot.set_data([], [])
        candidate_keypoints_duplicate_with_keypoints_plot.set_data([], [])
        candidate_keypoints_duplicate_with_prev_candidate_keypoints_plot.set_data([], [])

    untrackable_keypoints_plot.set_label(f'Untrackable KP: {len(debug_dict["untrackable_keypoints"])}')
    trackable_outlier_keypoints_plot.set_label(f'Trackable outlier KP: {len(debug_dict["trackable_outlier_keypoints"])}')
    trackable_keypoints_plot.set_label(f'Trackable KP: {len(debug_dict["trackable_keypoints"])}')
    untrackable_candidate_keypoints_plot.set_label(f'Untrackable CKP: {len(debug_dict["untrackable_candidate_keypoints"])} (total #CKP:{len(S["candidate_keypoints"]) if S["candidate_keypoints"] is not None else 0})')

    trackable_unpromotable_candidate_keypoints_plot.set_label(f'Trackable unpromotable CKP: {len(debug_dict["trackable_unpromotable_candidate_keypoints"])}')
    untriangulatable_promotable_candidate_keypoints_plot.set_label(f'Untriangulatable promotable CKP: {len(debug_dict["untriangulatable_promotable_candidate_keypoints"])}')
    promotable_candidate_keypoints_outside_thresholds_plot.set_label(f'Promotable CKP outside_thresholds: {len(debug_dict["promotable_candidate_keypoints_outside_thresholds"])}')
    promoted_candidate_keypoints_plot.set_label(f'Promoted CKP: {len(debug_dict["promoted_candidate_keypoints"])}')
    new_candidate_keypoints_plot.set_label(f'New CKP: {len(debug_dict["new_candidate_keypoints"])}')
    candidate_keypoints_duplicate_with_keypoints_plot.set_label(f'CKP duplicate w KP: {len(debug_dict["candidate_keypoints_duplicate_with_keypoints"])}')
    candidate_keypoints_duplicate_with_prev_candidate_keypoints_plot.set_label(f'CKP duplicate w prev CKP: {len(debug_dict["candidate_keypoints_duplicate_with_prev_candidate_keypoints"])}')
        
    if dataset == 0:
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=2, fancybox=True, shadow=True, fontsize=6)
    else:
        ax1.legend(loc='center left', bbox_to_anchor=(-0.7, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=6)

    if len(n_tracked_keypoints_list) > 20:
        keypoints_plot.set_data(range(20), n_tracked_keypoints_list[-20:])
        ax3.set_xlim(0, 20)
        ax3.set_ylim(0, max(n_tracked_keypoints_list[-20:]))
    else:
        keypoints_plot.set_data(range(len(n_tracked_keypoints_list)), n_tracked_keypoints_list)
        ax3.set_xlim(0, 20)
        ax3.set_ylim(0, max(n_tracked_keypoints_list))
                     
    if S["keypoints"].shape[0] < low_keypoint_plot_threshold:
        trajectory_points_with_low_keypoints += [T_WC[:, 3]]
        indices_with_low_keypoints += [i]

    landmarks_plot.set_data(S["landmarks"][:, 0], S["landmarks"][:, 2])

    if trailing_trajectory_plot:
        if index < 100:
            trajectory_plot.set_data(trajectory[0, :index+1], trajectory[2, :index+1])
        else:
            trajectory_plot.set_data(trajectory[0, index+1-100:index+1], trajectory[2, index+1-100:index+1])
        if index < 100:
            combined_x = np.concatenate((trajectory[0, :index+1], S["landmarks"][:, 0]))
            combined_z = np.concatenate((trajectory[2, :index+1], S["landmarks"][:, 2]))
        else:
            combined_x = np.concatenate((trajectory[0, index+1-100:index+1], S["landmarks"][:, 0]))
            combined_z = np.concatenate((trajectory[2, index+1-100:index+1], S["landmarks"][:, 2]))

        ax2.set_xlim(np.min(combined_x) - 1, np.max(combined_x) + 1)
        ax2.set_ylim(np.min(combined_z) - 1, np.max(combined_z) + 1)

    else:
        trajectory_plot.set_data(trajectory[0, :index+1], trajectory[2, :index+1])
        trajectory_points_with_low_keypoints_plot.set_data([item[0] for item in trajectory_points_with_low_keypoints], [item[2] for item in trajectory_points_with_low_keypoints])
    # plt.pause(0.0001)

fig, axs = plt.subplots(2, 1, figsize=(7, 7))
axs[0].title.set_text('Trajectory')
axs[0].plot(trajectory[0, :], trajectory[2, :], 'b-o', markersize=3)
axs[0].plot([item[0] for item in trajectory_points_with_low_keypoints], [item[2] for item in trajectory_points_with_low_keypoints], "o", color="orange", markersize=5, label=f'low keypoints (<{low_keypoint_plot_threshold})', linestyle='None')
axs[0].axis("equal")
axs[1].title.set_text('# Tracked keypoints at each frame')
axs[1].plot(n_tracked_keypoints_list, label="n tracked keypoints")
axs[1].plot(n_promotable_keypoints_before_angle_filtering_list, label="n_promotable_keypoints_before_angle_filtering")
axs[1].plot(n_lost_candidates_at_angle_filtering, label="n_lost_candidates_at_angle_filtering")
axs[1].plot(n_lost_candidates_at_cartesian_mask, label="n_lost_candidates_at_cartesian_mask")
axs[1].plot(n_promoted_keypoints_list, label="n promoted keypoints")
axs[1].plot(n_new_candidate_keypoints_list, label="n new candidate keypoints")
trans = transforms.blended_transform_factory(axs[1].transData, axs[1].transAxes)
axs[1].vlines(indices_with_low_keypoints, ymin=0, ymax=1, linewidth=1, color="orange", alpha=0.3, label=f"low keypoints (<{low_keypoint_plot_threshold})", zorder=-1, transform=trans)
axs[1].legend()

plt.tight_layout()
plt.show()

print('------------------\nPipeline finished')
