import numpy as np
from transforms3d.euler import euler2mat


def eval_rdiff_with_sym_axis(pred_rotation, gt_rotation, sym_axis):
    """
    Compute rotation error (unit: degree) based on symmetry axis.
    """
    if sym_axis == "x":
        x1, x2 = pred_rotation[:, 0], gt_rotation[:, 0]
        diff = np.sum(x1 * x2)
    elif sym_axis == "y":
        y1, y2 = pred_rotation[:, 1], gt_rotation[:, 1]
        diff = np.sum(y1 * y2)
    elif sym_axis == "z":
        z1, z2 = pred_rotation[:, 2], gt_rotation[:, 2]
        diff = np.sum(z1 * z2)
    else:  # sym_axis == "", i.e. no symmetry axis
        mat_diff = np.matmul(pred_rotation, gt_rotation.T)
        diff = mat_diff.trace()
        diff = (diff - 1) / 2.0

    diff = np.clip(diff, a_min=-1.0, a_max=1.0)
    return np.arccos(diff) / np.pi * 180  # degree


def eval_rdiff(pred_rotation, gt_rotation, geometric_symmetry):
    """
    Compute rotation error (unit: degree) based on geometric symmetry.
    """
    syms = geometric_symmetry.split("|")
    sym_axis = ""
    sym_N = np.array([1, 1, 1])  # x, y, z
    for sym in syms:
        if sym.find("inf") != -1:
            sym_axis += sym[0]
        elif sym != "no":
            idx = ord(sym[0]) - ord('x')
            value = int(sym[1:])
            sym_N[idx] = value
    if len(sym_axis) >= 2:
        return 0.0
    
    assert sym_N.min() >= 1

    gt_rotations = []
    for xi in range(sym_N[0]):
        for yi in range(sym_N[1]):
            for zi in range(sym_N[2]):
                R = euler2mat(
                    2 * np.pi / sym_N[0] * xi,
                    2 * np.pi / sym_N[1] * yi,
                    2 * np.pi / sym_N[2] * zi,
                )
                gt_rotations.append(gt_rotation @ R)

    r_diffs = []
    for gt_rotation in gt_rotations:
        r_diff = eval_rdiff_with_sym_axis(pred_rotation, gt_rotation, sym_axis)
        r_diffs.append(r_diff)
    
    r_diffs = np.array(r_diffs)
    return r_diffs.min()


def eval_tdiff(pred_translation, gt_translation):
    """
    Compute translation error (unit: cm).
    """
    t_diff = pred_translation - gt_translation
    return np.linalg.norm(t_diff, ord=2) * 100


def eval(pred_pose, gt_pose, geometric_symmetry):
    r_diff = eval_rdiff(pred_pose[:3, :3], gt_pose[:3, :3], geometric_symmetry)
    t_diff = eval_tdiff(pred_pose[:3, 3], gt_pose[:3, 3])
    return r_diff, t_diff


def main():
    # define geometric symmetry
    geometric_symmetry = "zinf|x2"

    # randomize a pose prediction
    pred_pose = np.zeros((4, 4))
    pred_pose[:3, :3] = euler2mat(
                            np.random.uniform(-np.pi, np.pi),
                            np.random.uniform(-np.pi, np.pi),
                            np.random.uniform(-np.pi, np.pi),
                        )
    pred_pose[:3, 3] = np.random.uniform(-0.01, 0.01, 3)
    pred_pose[3, 3] = 1

    # randomize a ground truth pose
    gt_pose = np.zeros((4, 4))
    gt_pose[:3, :3] = euler2mat(
                            np.random.uniform(-np.pi, np.pi),
                            np.random.uniform(-np.pi, np.pi),
                            np.random.uniform(-np.pi, np.pi),
                        )
    gt_pose[:3, 3] = np.random.uniform(-0.01, 0.01, 3)
    gt_pose[3, 3] = 1

    # evaluate rotation error and translation error
    r_diff, t_diff = eval(pred_pose @ gt_pose, gt_pose, geometric_symmetry)
    print("rotation error = {} (deg)".format(r_diff))
    print("translation error = {} (cm)".format(t_diff))


if __name__ == "__main__":
    main()