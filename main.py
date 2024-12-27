import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_frame import processFrame
# from bootstrap_vo import bootstrap_vo
# from continuous_vo import continuous_vo

# boolean variable to determine if we are bootstrapping or not
bootstrap = False

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
        key_points = key_points.T
        # Swap columns of keypoints
        key_points[[1, 0], :] = key_points[[0, 1], :]
        key_points = key_points.T
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

trajectory = np.zeros((3, len(range_frames)+1))
n_tracked_keypoints_list = []
n_promoted_keypoints_list = []
n_lost_candidates_at_angle_filtering = []
n_lost_candidates_at_cartesian_mask = []

# Visualisation of VO pipeline
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.set_title('Keypoints')  # Set the title of the figure
img_plot = ax1.imshow(img0, cmap='gray')
keypoints_plot, = ax1.plot([], [], 'yo', markersize=5, label='Keypoints')
candidate_keypoints_plot, = ax1.plot([], [], 'bo', markersize=5, label='Candidate Keypoints')
ax1.legend()

ax2.set_title('Estimated Trajectory')
landmarks_plot, = ax2.plot([], [], 'ro', markersize=5, label='Landmarks')
trajectory_plot, = ax2.plot([], [], 'b-o', markersize=5, label='Trajectory')
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
    n_promoted_keypoints_list += [debug_dict["n_promoted_keypoints"]]
    n_lost_candidates_at_angle_filtering += [debug_dict["n_lost_candidates_at_angle_filtering"]]
    n_lost_candidates_at_cartesian_mask += [debug_dict["n_lost_candidates_at_cartesian_mask"]]


    # Update the plots
    img_plot.set_data(img)
    keypoints_plot.set_data(S["keypoints"][:, 0], S["keypoints"][:, 1])
    if S["candidate_keypoints"] is not None:
        candidate_keypoints_plot.set_data(S["candidate_keypoints"][:, 0], S["candidate_keypoints"][:, 1])
    else:
        candidate_keypoints_plot.set_data([], [])
    keypoints_plot.set_label(f'Keypoints: {len(S["keypoints"])}')
    candidate_keypoints_plot.set_label(f'Candidate Keypoints: {len(S["candidate_keypoints"]) if S["candidate_keypoints"] is not None else 0}')
    ax1.legend()

    landmarks_plot.set_data(S["landmarks"][:, 0], S["landmarks"][:, 2])
    trajectory_plot.set_data(trajectory[0, :index+1], trajectory[2, :index+1])
    combined_x = np.concatenate((trajectory[0, :index+1], S["landmarks"][:, 0]))
    combined_z = np.concatenate((trajectory[2, :index+1], S["landmarks"][:, 2]))
    ax2.set_xlim(np.min(combined_x) - 1, np.max(combined_x) + 1)
    ax2.set_ylim(np.min(combined_z) - 1, np.max(combined_z) + 1)

    plt.pause(0.01)

fig, axs = plt.subplots(2, 1, figsize=(7, 7))
axs[0].title.set_text('Trajectory')
axs[0].plot(trajectory[0, :], trajectory[2, :])
axs[1].title.set_text('# Tracked keypoints at each frame')
axs[1].plot(n_tracked_keypoints_list, label="n tracked keypoints")
axs[1].plot(n_promoted_keypoints_list, label="n promoted keypoints")
axs[1].plot(n_lost_candidates_at_angle_filtering, label="n_lost_candidates_at_angle_filtering")
axs[1].plot(n_lost_candidates_at_cartesian_mask, label="n_lost_candidates_at_cartesian_mask")
axs[1].legend()

plt.tight_layout()
plt.show()

print('------------------\nPipeline finished')
