import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
from prettytable import PrettyTable
from tqdm import tqdm

from utils.geometry import rotation_angle, angle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-cp', '--check_previous', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def get_pairs(file):
    with open(file, 'r') as f:
        pairs = f.readlines()
    return [tuple(x.strip().split(' ')) for x in pairs]

def get_result_dict(info, pose_est, R_gt, t_gt):
    out = {}

    R_est, t_est = pose_est.R, pose_est.t

    out['R_err'] = rotation_angle(R_est.T @ R_gt)
    out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = R_est.tolist()
    out['t_gt'] = R_gt.tolist()

    out['P_err'] = max(out['R_err'], out['t_err'])

    info['inliers'] = []
    out['info'] = info

    return out


def eval_experiment(x):
    iters, experiment, kp1, kp2, R_gt, t_gt, K1, K2, t = x

    if iters is None:
        ransac_dict = {'max_iterations': 5000, 'progressive_sampling': False, 'min_iterations': 50}
    else:
        ransac_dict = {'max_iterations': iters, 'progressive_sampling': False, 'min_iterations': iters}

    opt = {'max_error': t, 'ransac': ransac_dict, 'bundle': {'max_iterations': 100}}

    camera1 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]}
    camera2 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]}

    start = perf_counter()
    if experiment == '5pE':
        pose_est, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, opt)
    else:
        opt['v_umlaut_enm'] = 'NISTER' if '+ ENM' in experiment else 'NONE'
        pose_est, info = poselib.estimate_relative_pose_v_umlaut(kp1, kp2, camera1, camera2, opt)
    info['runtime'] = 1000 * (perf_counter() - start)

    result_dict = get_result_dict(info, pose_est, R_gt, t_gt)
    result_dict['experiment'] = experiment

    return result_dict


def print_results(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'median pose err', 'mean pose err',
                       'Pose AUC@5', 'Pose AUC@10', 'Pose AUC@20',
                       'median time', 'mean time', 'median inliers', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])

        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        exp_name = exp


        tab.add_row([exp_name, np.median(p_errs), np.mean(p_errs),
                     100*np.mean(p_res[:5]), 100*np.mean(p_res[:10]), 100*np.mean(p_res),
                     np.median(times), np.mean(times),
                     np.median(inliers), np.mean(inliers)])
    print(tab)

    print('latex')

    print(tab.get_formatted_string('latex'))


def get_im_generator(args, dataset_path):
    R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
    T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
    K_file = h5py.File(os.path.join(dataset_path, 'K.h5'))
    C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
    pairs = get_pairs(os.path.join(dataset_path, f'{args.feature_file}.txt'))
    if args.first is not None:
        pairs = pairs[:args.first]

    def gen_data():
        for img_name_1, img_name_2 in pairs:
            # R, T and K of Aachen use '\' instead of '/' in image names
            gt_name_1, gt_name_2 = img_name_1.replace('/', '\\'), img_name_2.replace('/', '\\')
            R1 = np.array(R_file[gt_name_1])
            t1 = np.array(T_file[gt_name_1])
            R2 = np.array(R_file[gt_name_2])
            t2 = np.array(T_file[gt_name_2])
            K1 = np.array(K_file[gt_name_1])
            K2 = np.array(K_file[gt_name_2])

            R_gt = np.dot(R2, R1.T)
            t_gt = t2 - np.dot(R_gt, t1)

            matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])

            kp1 = matches[:, :2]
            kp2 = matches[:, 2:4]

            if len(kp1) < 5:
                continue

            yield np.copy(kp1), np.copy(kp2), R_gt, t_gt, K1, K2, args.threshold

    return gen_data, len(pairs)

def gen_wrapper(gen, experiments, iterations_list):
    for x in gen:
        for experiment in experiments:
            for iterations in iterations_list:
                yield iterations, experiment, *x


def eval(args):
    experiments = ['5pE', '5pF-V-Umlaut', '5pF-V-Umlaut + ENM (5p)']

    dataset_path = args.dataset_path
    basename = os.path.basename(os.path.normpath(dataset_path))

    if args.threshold != 1.0:
        basename = f'{basename}-{args.threshold}t'


    matches_basename = os.path.basename(args.feature_file)

    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    else:
        iterations_list = [None]

    json_string = f'twoview-{basename}-{matches_basename}.json'

    json_path = os.path.join('results', json_string)

    if args.check_previous:
        print("Checking previous!")
        if os.path.exists(json_path):
            if not args.append:
                raise ValueError("Ran check previous without append when the results file already exists! Aborting!")

            with open(json_path, 'r') as f:
                prev_results = json.load(f)

            prev_experiments = set([x['experiment'] for x in prev_results])

            experiments = list(set(experiments).difference(prev_experiments))

            print("Some experiments already found. Only performing experiments: ", experiments)
        else:
            print("Prev file not found")

    if args.load:
        print("Loading: ", json_string)
        with open(json_path, 'r') as f:
            results = json.load(f)

    else:
        gen_data, num_pairs = get_im_generator(args, dataset_path)

        total_length = len(experiments) * len(iterations_list) * num_pairs

        print(f"Total runs: {total_length} for {num_pairs} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_wrapper(gen_data(), experiments, iterations_list), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_wrapper(gen_data(), experiments, iterations_list), total=total_length))]

        os.makedirs('results', exist_ok=True)

        if args.append:
            if os.path.exists(json_path):
                print(f"Appending from: {json_path}")
                with open(json_path, 'r') as f:
                    prev_results = json.load(f)
            else:
                print("Prev file not found!")
                prev_results = []

            if args.overwrite:
                prev_results = [x for x in prev_results if x['experiment'] not in experiments]

            results.extend(prev_results)

        with open(os.path.join('results', json_string), 'w') as f:
            json.dump(results, f)

        print("Done")

    print_results(experiments, results)


if __name__ == '__main__':
    args = parse_args()
    eval(args)