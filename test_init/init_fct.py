import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import signal
from scipy.spatial.distance import cdist

def harris(img, patch_size, kappa):
    sobel_para = np.array([-1, 0, 1])
    sobel_orth = np.array([1, 2, 1])

    Ix = signal.convolve2d(img, sobel_para[None, :], mode="valid")
    Ix = signal.convolve2d(Ix, sobel_orth[:, None], mode="valid").astype(float)

    Iy = signal.convolve2d(img, sobel_para[:, None], mode="valid")
    Iy = signal.convolve2d(Iy, sobel_orth[None, :], mode="valid").astype(float)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix*Iy

    patch = np.ones([patch_size, patch_size])
    pr = patch_size // 2
    sIxx = signal.convolve2d(Ixx, patch, mode="valid")
    sIyy = signal.convolve2d(Iyy, patch, mode="valid")
    sIxy = signal.convolve2d(Ixy, patch, mode="valid")

    scores = (sIxx * sIyy - sIxy ** 2) - kappa * ((sIxx + sIyy) ** 2)

    scores[scores < 0] = 0

    scores = np.pad(scores, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    return scores


def selectKeypoints(scores, num, r):
    keypoints = np.zeros([2, num])
    temp_scores = np.pad(scores, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(num):
        kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)
        keypoints[:, i] = np.array(kp) - r
        temp_scores[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)] = 0

    return keypoints

def describeKeypoints(img, keypoints, r):
    N = keypoints.shape[1]
    desciptors = np.zeros([(2*r+1)**2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(N):
        kp = keypoints[:, i].astype(np.int16) + r
        desciptors[:, i] = padded[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)].flatten()

    return desciptors

def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    dists = cdist(query_descriptors.T, database_descriptors.T, 'euclidean')
    matches = np.argmin(dists, axis=1)
    dists = dists[np.arange(matches.shape[0]), matches]
    min_non_zero_dist = dists.min()

    matches[dists >= match_lambda * min_non_zero_dist] = -1

    # remove double matches
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]

    return unique_matches

def plotMatches(matches, query_keypoints, database_keypoints):
    query_indices = np.nonzero(matches >= 0)[0]
    match_indices = matches[query_indices]

    x_from = query_keypoints[0, query_indices]
    x_to = database_keypoints[0, match_indices]
    y_from = query_keypoints[1, query_indices]
    y_to = database_keypoints[1, match_indices]

    for i in range(x_from.shape[0]):
        plt.plot([y_from[i], y_to[i]], [x_from[i], x_to[i]], 'g-', linewidth=3)



def fundamentalEightPointNormalized(p1, p2):
    pass

    # Normalize each set of points so that the origin
    # is at centroid and mean distance from origin is sqrt(2).
    p1_tilde, T1 = normalise2DPts(p1)
    p2_tilde, T2 = normalise2DPts(p2)

    # Linear solution
    F = fundamentalEightPoint(p1_tilde, p2_tilde)

    # Undo the normalization
    F = T2.T @ F @ T1

    return F

def fundamentalEightPoint(p1, p2):
    pass
    # Sanity checks
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"
    
    num_points = p1.shape[1]
    assert(num_points>=8), \
            'Insufficient number of points to compute fundamental matrix (need >=8)'

    # Compute the measurement matrix A of the linear homogeneous system whose
    # solution is the vector representing the fundamental matrix.
    A = np.zeros((num_points,9))
    for i in range(num_points):
        A[i,:] = np.kron( p1[:,i], p2[:,i] ).T
    
    # "Solve" the linear homogeneous system of equations A*f = 0.
    # The correspondences x1,x2 are exact <=> rank(A)=8 -> there exist an exact solution
    # If measurements are noisy, then rank(A)=9 => there is no exact solution, 
    # seek a least-squares solution.
    _, _, vh= np.linalg.svd(A,full_matrices = False)
    F = np.reshape(vh[-1,:], (3,3)).T

    # Enforce det(F)=0 by projecting F onto the set of 3x3 singular matrices
    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F = u @ np.diag(s) @ vh

    return F

def normalise2DPts(pts):
    pass
    N = pts.shape[1]

    # Convert homogeneous coordinates to Euclidean coordinates (pixels)
    pts_ = pts/pts[2,:]

    # Centroid (Euclidean coordinates)
    mu = np.mean(pts_[:2,:], axis = 1)

    # Average distance or root mean squared distance of centered points
    # It does not matter too much which criterion to use. Both improve the
    # numerical conditioning of the Fundamental matrix estimation problem.
    pts_centered = (pts_[:2,:].T - mu).T

    # Option 1: RMS distance
    sigma = np.sqrt( np.mean( np.sum(pts_centered**2, axis = 0) ) )

    # Option 2: average distance
    # sigma = mean( sqrt(sum(pts_centered.^2)) );

    s = np.sqrt(2) / sigma
    T = np.array([
        [s, 0, -s * mu[0]],
        [0, s, -s * mu[1]],
        [0, 0, 1]])

    pts_tilde = T @ pts_

    return pts_tilde, T


def decomposeEssentialMatrix(E):
    pass
    u, _, vh = np.linalg.svd(E)

    # Translation
    u3 = u[:, 2]

    # Rotations
    W = np.array([ [0, -1,  0],
                   [1,  0,  0],
                   [0,  0,  1]])

    R = np.zeros((3,3,2))
    R[:, :, 0] = u @ W @ vh
    R[:, :, 1] = u @ W.T @ vh

    for i in range(2):
        if np.linalg.det(R[:, :, i]) < 0:
            R[:, :, i] *= -1

    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    return R, u3

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K):
    pass

    # Projection matrix of camera 1
    M1 = K @ np.eye(3,4)

    total_points_in_front_best = 0
    for iRot in range(2):
        R_C2_C1_test = Rots[:,:,iRot]
        
        for iSignT in range(2):
            T_C2_C1_test = u3 * (-1)**iSignT
            
            M2 = K @ np.c_[R_C2_C1_test, T_C2_C1_test]
            P_C1 = linearTriangulation(points0_h, points1_h, M1, M2)
            
            # project in both cameras
            P_C2 = np.c_[R_C2_C1_test, T_C2_C1_test] @ P_C1
            
            num_points_in_front1 = np.sum(P_C1[2,:] > 0)
            num_points_in_front2 = np.sum(P_C2[2,:] > 0)
            total_points_in_front = num_points_in_front1 + num_points_in_front2
                  
            if (total_points_in_front > total_points_in_front_best):
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test;
                T = T_C2_C1_test;
                total_points_in_front_best = total_points_in_front;

    return R, T

def linearTriangulation(p1, p2, M1, M2):
    pass
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"
    assert(M1.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"
    assert(M2.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"

    num_points = p1.shape[1]
    P = np.zeros((4, num_points))

    # Linear Algorithm
    for i in range(num_points):
        # Build matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1[:, i]) @ M1
        A2 = cross2Matrix(p2[:, i]) @ M2
        A = np.r_[A1, A2]

        # Solve the homogeneous system of equations
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:, i] = vh.T[:,-1]

    # Dehomogenize (P is expressed in homoegeneous coordinates)
    P /= P[3,:]

    return P

def cross2Matrix(x):
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M

def ransacLocalization(matched_query_keypoints, corresponding_landmarks, K):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.
    """
    pass
    use_p3p = False
    tweaked_for_more = True
    adaptive = True  # whether or not to use ransac adaptively

    if use_p3p:
        num_iterations = 1000 if tweaked_for_more else 200
        pixel_tolerance = 10
        k = 3
    else:
        num_iterations = 2000
        pixel_tolerance = 10
        k = 6

    if adaptive:
        num_iterations = float("inf")

    # Initialize RANSAC
    best_inlier_mask = np.zeros(matched_query_keypoints.shape[1])

    # (row, col) to (u, v)
    matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)
    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    i = 0
    while num_iterations > i:

        # Model from k samples (DLT or P3P)
        indices = np.random.permutation(corresponding_landmarks.shape[0])[:k]
        landmark_sample = corresponding_landmarks[indices, :]
        keypoint_sample = matched_query_keypoints[:, indices]

        if use_p3p:
            success, rotation_vectors, translation_vectors = cv2.solveP3P(
                landmark_sample,
                keypoint_sample.T,
                K,
                None,
                flags=cv2.SOLVEPNP_P3P,
            )
            t_C_W_guess = []
            R_C_W_guess = []
            for rotation_vector in rotation_vectors:
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                for translation_vector in translation_vectors:
                    R_C_W_guess.append(rotation_matrix)
                    t_C_W_guess.append(translation_vector)

        else:
            M_C_W_guess = estimatePoseDLT(
                keypoint_sample.T, landmark_sample, K
            )
            R_C_W_guess = M_C_W_guess[:, :2]
            t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        if not use_p3p:
            C_landmarks = (
                np.matmul(
                    R_C_W_guess, corresponding_landmarks[:, :, None]
                ).squeeze(-1)
                + t_C_W_guess[None, :]
            )
            projected_points = projectPoints(C_landmarks, K)
            difference = matched_query_keypoints - projected_points.T
            errors = (difference**2).sum(0)
            is_inlier = errors < pixel_tolerance**2

        else:
            # If we use p3p, also consider inliers for the 4 solutions.
            is_inlier = np.zeros(corresponding_landmarks.shape[0])
            for alt_idx in range(len(R_C_W_guess)):

                # Project points
                C_landmarks = np.matmul(
                    R_C_W_guess[alt_idx], corresponding_landmarks[:, :, None]
                ).squeeze(-1) + t_C_W_guess[alt_idx][None, :].squeeze(-1)
                projected_points = projectPoints(C_landmarks, K)

                difference = matched_query_keypoints - projected_points.T
                errors = (difference**2).sum(0)
                alternative_is_inlier = errors < pixel_tolerance**2
                if alternative_is_inlier.sum() > is_inlier.sum():
                    is_inlier = alternative_is_inlier

        min_inlier_count = 30 if tweaked_for_more else 6

        if (
            is_inlier.sum() > max_num_inliers
            and is_inlier.sum() >= min_inlier_count
        ):
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        if adaptive:
            # estimate of the outlier ratio
            outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]

            # formula to compute number of iterations from estimated outlier ratio
            confidence = 0.95
            upper_bound_on_outlier_ratio = 0.90
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            num_iterations = np.log(1 - confidence) / np.log(
                1 - (1 - outlier_ratio) ** k
            )

            # cap the number of iterations at 15000
            num_iterations = min(15000, num_iterations)

        num_iteration_history.append(num_iterations)
        max_num_inliers_history.append(max_num_inliers)

        i += 1
    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W = estimatePoseDLT(
            matched_query_keypoints[:, best_inlier_mask].T,
            corresponding_landmarks[best_inlier_mask, :],
            K,
        )
        R_C_W = M_C_W[:, :2]
        t_C_W = M_C_W[:, -1]

        if adaptive:
            print(
                "    Adaptive RANSAC: Needed {} iteration to converge.".format(
                    i - 1
                )
            )
            print(
                "    Adaptive RANSAC: Estimated Ouliers: {} %".format(
                    100 * outlier_ratio
                )
            )

    return (
        R_C_W,
        t_C_W,
        best_inlier_mask,
        max_num_inliers_history,
        num_iteration_history,
    )


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d , renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
    
def drawCamera(ax, position, direction, length_scale = 1, head_size = 10, 
        equal_axis = True, set_ax_limits = True):

    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='r')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 0]],
                [position[1], position[1] + length_scale * direction[1, 0]],
                [position[2], position[2] + length_scale * direction[2, 0]],
                **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='g')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 1]],
                [position[1], position[1] + length_scale * direction[1, 1]],
                [position[2], position[2] + length_scale * direction[2, 1]],
                **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='b')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 2]],
                [position[1], position[1] + length_scale * direction[1, 2]],
                [position[2], position[2] + length_scale * direction[2, 2]],
                **arrow_prop_dict)
    ax.add_artist(a)

    if not set_ax_limits:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlim([min(xlim[0], position[0]), max(xlim[1], position[0])])
    ax.set_ylim([min(ylim[0], position[1]), max(ylim[1], position[1])])
    ax.set_zlim([min(zlim[0], position[2]), max(zlim[1], position[2])])
    
    # This sets the aspect ratio to 'equal'
    if equal_axis:
        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                       np.ptp(ax.get_ylim()),
                       np.ptp(ax.get_zlim())))

def estimatePoseDLT(p, P, K):
    p_norm = (np.linalg.inv(K) @ np.c_[p, np.ones((p.shape[0], 1))].T).T

    num_corners = p_norm.shape[0]
    Q = np.zeros((2*num_corners, 12))

    for i in range(num_corners):
        u = p_norm[i, 0]
        v = p_norm[i, 1]

        Q[2*i, 0:3] = P[i,:]
        Q[2*i, 3] = 1
        Q[2*i, 8:11] = -u * P[i,:]
        Q[2*i, 11] = -u
        
        Q[2*i+1, 4:7] = P[i,:]
        Q[2*i+1, 7] = 1
        Q[2*i+1, 8:11] = -v * P[i,:]
        Q[2*i+1, 11] = -v

    u, s, v = np.linalg.svd(Q, full_matrices=True)
    M_tilde = np.reshape(v.T[:,-1], (3,4));

    if (np.linalg.det(M_tilde[:, :3]) < 0):
        M_tilde *= -1

    R = M_tilde[:, :3]

    u, s, v = np.linalg.svd(R);
    R_tilde = u @ v;

    alpha = np.linalg.norm(R_tilde, 'fro')/np.linalg.norm(R, 'fro');
    M_tilde = np.c_[R_tilde, alpha * M_tilde[:,3]];
    
    return M_tilde


def projectPoints(points_3d, K, D=np.zeros([4, 1])):
    # get image coordinates
    projected_points = np.matmul(K, points_3d[:, :, None]).squeeze(-1)
    projected_points /= projected_points[:, 2, None]

    # apply distortion
    projected_points = distortPoints(projected_points[:, :2], D, K)

    return projected_points


def distortPoints(x, D, K):
    """Applies lens distortion D(2) to 2D points x(Nx2) on the image plane. """

    k1, k2 = D[0], D[1]

    u0 = K[0, 2]
    v0 = K[1, 2]

    xp = x[:, 0] - u0
    yp = x[:, 1] - v0

    r2 = xp**2 + yp**2
    xpp = u0 + xp * (1 + k1*r2 + k2*r2**2)
    ypp = v0 + yp * (1 + k1*r2 + k2*r2**2)

    x_d = np.stack([xpp, ypp], axis=-1)

    return x_d
