import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_frame import processFrame
import matplotlib.transforms as transforms
# from bootstrap_vo import bootstrap_vo
# from continuous_vo import continuous_vo

# boolean variable to determine if we are bootstrapping or not
bootstrap = False
low_keypoint_plot_threshold = 200

# Setup
dataset = 0 # 0: KITTI, 1: Malaga, 2: parking
parking_path = "" #"data/parking/images/"
malaga_path = "" #"data/malaga/Images/"
kitti_path = "" #"data/kitti/image_0/"
data_VO_path = "" # for testing purposes

if dataset == 0:
    assert 'kitti_path' in locals()
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'data/kitti/poses/05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    last_frame = 4540
    K = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])
elif dataset == 1:
    assert 'malaga_path' in locals()
    images = sorted(os.listdir(os.path.join(malaga_path, 
                'data/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/')))
    left_images = images[0::2]
    last_frame = len(left_images) 
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
elif dataset == 2:
    assert 'parking_path' in locals()
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'data/parking/K.txt'))
    ground_truth = np.loadtxt(os.path.join(parking_path, 'data/parking/poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    print(ground_truth)
    plt.plot(ground_truth[1],ground_truth[0])
    plt.show()
else:
    raise AssertionError("Invalid dataset selection")

if bootstrap:
    # Load frames to perform bootstrapping on
    bootstrap_frames = [0, 2]

    if dataset == 0:
        img0 = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', 
                        f"{bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', 
                        f"{bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    elif dataset == 1:
        img0 = cv2.imread(os.path.join(malaga_path, 
                        'data/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                        left_images[bootstrap_frames[0]]), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(malaga_path, 
                        'data/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                        left_images[bootstrap_frames[1]]), cv2.IMREAD_GRAYSCALE)
    elif dataset == 2:
        img0 = cv2.imread(os.path.join(parking_path, 
                        f"data/parking/images/img_{bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(parking_path, 
                        f"data/parking/images/img_{bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    else:
        raise AssertionError("Invalid dataset selection")
    
    # TODO: DO BOOTSTRAP ACTION HERE
    S_prev = {
                "keypoints": key_points, # dim: 2xK
                "landmarks": p_W_landmarks.T, # dim: 3xK
                "candidate_keypoints": None, # no candidate keypoints in the beginning
                "first_observations": None, # no candidate keypoints in the beginning
                "pose_at_first_observation": None # no candidate keypoints in the beginning
            }

else:
    # Circumvent bootstrapping by loading precomputed bootstrapping data
    if dataset == 0:
        K = np.loadtxt(os.path.join("", "data_VO/K.txt")) # camera matrix
        key_points = np.loadtxt(os.path.join(data_VO_path, 'data_VO/keypoints.txt'), dtype=np.float32) # note: cv2.calcOpticalFlowPyrLK expects float32
        p_W_landmarks = np.loadtxt(os.path.join(data_VO_path, 'data_VO/p_W_landmarks.txt'), dtype=np.float32)
        img0 = cv2.imread(os.path.join(data_VO_path, f"data_VO/000000.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(data_VO_path, f"data_VO/000001.png"), cv2.IMREAD_GRAYSCALE)
        # # Swap columns of keypoints
        key_points[:, [1, 0]] = key_points[:, [0, 1]]

        S_prev = {
                    "keypoints": key_points, # dim: Kx2
                    "landmarks": p_W_landmarks, # dim: Kx3
                    "candidate_keypoints": None, # no candidate keypoints in the beginning
                    "first_observations": None, # no candidate keypoints in the beginning
                    "pose_at_first_observation": None # no candidate keypoints in the beginning
                }
    else:
        raise NotImplementedError(f'Pre-computed bootstrapping values not available for this dataset with index {i}.')


# Continuous operation
if bootstrap:
    range_frames = range(bootstrap_frames[1] + 1, last_frame + 1)
else:
    # range_frames = range(1, last_frame + 1)
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.set_title('Keypoints')  # Set the title of the figure
img_plot = ax1.imshow(img0, cmap='gray')
# keypoints_plot, = ax1.plot([], [], 'yo', markersize=5, label='Keypoints')
# candidate_keypoints_plot, = ax1.plot([], [], 'bo', markersize=5, label='Candidate Keypoints')
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
    else:
        raise AssertionError("Invalid dataset selection")
    
    S, T_WC, debug_dict = processFrame(img, img_prev, S_prev, K)
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
        
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -1.1), ncol=2, fancybox=True, shadow=True)

    if S["keypoints"].shape[0] < low_keypoint_plot_threshold:
        trajectory_points_with_low_keypoints += [T_WC[:, 3]]
        indices_with_low_keypoints += [i]

    landmarks_plot.set_data(S["landmarks"][:, 0], S["landmarks"][:, 2])
    trajectory_plot.set_data(trajectory[0, :index+1], trajectory[2, :index+1])
    trajectory_points_with_low_keypoints_plot.set_data([item[0] for item in trajectory_points_with_low_keypoints], [item[2] for item in trajectory_points_with_low_keypoints])
    combined_x = np.concatenate((trajectory[0, :index+1], S["landmarks"][:, 0]))
    combined_z = np.concatenate((trajectory[2, :index+1], S["landmarks"][:, 2]))
    ax2.set_xlim(np.min(combined_x) - 1, np.max(combined_x) + 1)
    ax2.set_ylim(np.min(combined_z) - 1, np.max(combined_z) + 1)
    # fig.subplots_adjust(right=0.5)
    plt.pause(0.01)


# plt.close()
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
