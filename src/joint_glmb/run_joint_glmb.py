import numpy as np
import sys
import os
import os.path as osp
import glob
import cv2
import motmetrics as mm
import multiprocessing


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add tracking_utils to PYTHONPATH
lib_path = osp.join(this_dir, '../')
sys.path.insert(0, lib_path)

from tracking_utils.evaluation import Evaluator
from tracking_utils.log import logger
from tracking_utils.timer import Timer

from joint_glmb import GLMB


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def mot15(root):
    seqs_train = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
                  'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
    seqs_test = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli',
                 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']
    train_dir = root + '/2DMOT2015/train'
    test_dir = root + '/2DMOT2015/test'
    return train_dir, test_dir, seqs_train, seqs_test


def mot16(root):
    seqs_train = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
    seqs_test = ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']
    train_dir = root + '/MOT16/train'
    test_dir = root + '/MOT16/test'
    return train_dir, test_dir, seqs_train, seqs_test


def mot17(root):
    seqs_train = ['MOT17-02-DPM', 'MOT17-02-FRCNN', 'MOT17-02-SDP', 'MOT17-04-DPM', 'MOT17-04-FRCNN', 'MOT17-04-SDP',
                  'MOT17-05-DPM', 'MOT17-05-FRCNN', 'MOT17-05-SDP', 'MOT17-09-DPM', 'MOT17-09-FRCNN', 'MOT17-09-SDP',
                  'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP', 'MOT17-11-DPM', 'MOT17-11-FRCNN', 'MOT17-11-SDP',
                  'MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP']
    seqs_test = ['MOT17-01-DPM', 'MOT17-01-FRCNN', 'MOT17-01-SDP', 'MOT17-03-DPM', 'MOT17-03-FRCNN', 'MOT17-03-SDP',
                 'MOT17-06-DPM', 'MOT17-06-FRCNN', 'MOT17-06-SDP', 'MOT17-07-DPM', 'MOT17-07-FRCNN', 'MOT17-07-SDP',
                 'MOT17-08-DPM', 'MOT17-08-FRCNN', 'MOT17-08-SDP', 'MOT17-12-DPM', 'MOT17-12-FRCNN', 'MOT17-12-SDP',
                 'MOT17-14-DPM', 'MOT17-14-FRCNN', 'MOT17-14-SDP']
    train_dir = root + '/MOT17/train'
    test_dir = root + '/MOT17/test'
    return train_dir, test_dir, seqs_train, seqs_test


def mot20(root):
    seqs_train = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
    seqs_test = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
    train_dir = root + '/MOT20/train'
    test_dir = root + '/MOT20/test'
    return train_dir, test_dir, seqs_train, seqs_test


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                # if track_id < 0:
                #     continue
                x1, y1, w, h = tlwh
                # x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=str(track_id), x1=str(x1), y1=str(y1),
                                          w=str(w), h=str(h))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def visualize_gt(img0, tracks):
    for i, id_tlwh in enumerate(tracks):
        tid = id_tlwh[0]
        l, t = id_tlwh[1], id_tlwh[2]
        r, b = id_tlwh[1] + id_tlwh[3], id_tlwh[2] + id_tlwh[4]
        cxy = (int(id_tlwh[1] + id_tlwh[3] / 2), int(id_tlwh[2] + id_tlwh[4] / 2))
        # draw bbox
        img0 = cv2.circle(img0, cxy, radius=8, color=(255, 255, 255), thickness=-1)
        img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
        img0 = cv2.putText(img0, str(tid), org=cxy, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                           color=(0, 255, 255), thickness=2)
    cv2.imshow('Ground Truth', img0)
    cv2.moveWindow('Ground Truth', 200, 200)
    cv2.waitKey(50)


def eval_seq(img_path, detector_path, result_filename, save_dir=None, frame_rate=30, width=19020, height=1080):
    data_type = 'mot'
    show_image = False
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = GLMB(width, height)
    timer = Timer()
    results = []
    frame_id = 0

    npz_lines = np.load(detector_path)
    # np_gt = np.loadtxt(os.path.join(img_path, "gt", "gt.txt"), delimiter=',')
    files = sorted(glob.glob(osp.join(img_path, 'img1') + '/*.jpg'))

    for i, path in enumerate(files):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        img0 = cv2.imread(path)
        # gt_tracks = np_gt[np.logical_and(np_gt[:, 0] == (i+1), np_gt[:, 6] == 1)][:, 1:6]
        # visualize_gt(np.copy(img0), gt_tracks.astype(int))

        try:
            dets, id_feature = npz_lines[str(i) + '_det'], npz_lines[str(i) + '_feat']
            remain_inds = dets[:, 4] > 0.5
            z, feat = dets[remain_inds, :4].astype('f8'), id_feature[remain_inds].astype('f8')
        except:
            z, feat = np.empty((0, 4)), np.empty((0, 128))  # no detection

        # run tracking
        timer.tic()
        X, L = tracker.run_glmb(z, feat, i)
        online_tlwhs = []
        online_ids = []
        X[2, :] = X[2, :] * X[3, :]  # xyah to xywh
        x_visual = np.copy(X)
        X[0, :], X[1, :] = X[0, :] - X[2, :] / 2, X[1, :] - X[3, :] / 2  # xywh to tlwh
        for i, tlwh in enumerate(X.T):
            tid = L[i]
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 10 and not vertical:  # opt.min_box_area
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                if show_image:
                    cxy, id, wh = x_visual[:2, i], L[i], x_visual[2:4, i]
                    l, t = int(cxy[0] - wh[0] / 2), int(cxy[1] - wh[1] / 2)
                    r, b = int(cxy[0] + wh[0] / 2), int(cxy[1] + wh[1] / 2)
                    cxy = (int(cxy[0]), int(cxy[1]))
                    # draw bbox
                    img0 = cv2.circle(img0, cxy, radius=8, color=(255, 255, 255), thickness=-1)
                    img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                    img0 = cv2.putText(img0, str(id), org=cxy, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                                       color=(0, 255, 255), thickness=2)
        if show_image:
            str_show = 'Frame {}'.format(frame_id) + ', FPS:{}'.format(round((1. / (timer.average_time + 1e-8)), 2))
            img0 = cv2.putText(img0, str_show, org=(img0.shape[1] - 400, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(255, 255, 255), thickness=2)
            scale_percent = 0.6  # percent of original size
            dim = (int(img0.shape[1] * scale_percent), int(img0.shape[0] * scale_percent))
            resized = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)  # resize image
            cv2.imshow('Image', resized)
            cv2.moveWindow('Image', 200, 200)
            cv2.waitKey(1)
        timer.toc()
        # # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # if save_dir is not None:
        #     cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), img0)
        frame_id += 1

    # save results
    write_results(result_filename, results, data_type)
    cv2.destroyAllWindows()
    return frame_id, timer.average_time, timer


def demo(data_root, ismot20=False, result_dir="mot16_glmb_cstrack", detector_dir="detector_cstrack"):
    # data_root = '/media/ubuntu/2715608D71CBF6FC/datasets/mot'
    train_dir, test_dir, seqs_train, seqs_test = mot16(data_root)
    if ismot20:
        train_dir, test_dir, seqs_train, seqs_test = mot20(data_root)
    accs = []
    root = os.path.join("../../results/", result_dir)
    mkdir_if_missing(root)
    processes = []
    for seq in seqs_train:
        logger.info('start seq: {}'.format(seq))
        meta_info = open(os.path.join(train_dir, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        width = int(meta_info[meta_info.find('imWidth') + 8:meta_info.find('\nimHeight')])
        height = int(meta_info[meta_info.find('imHeight') + 9:meta_info.find('\nimExt')])

        detector_path = os.path.join("../../detection/", detector_dir, seq + '.npz')
        img_path = os.path.join(data_root, train_dir, seq)

        logger.info('Starting tracking...')
        # result_root = os.path.join(root, seq)
        # mkdir_if_missing(result_root)
        # files = glob.glob(result_root + '/*.jpg')
        # for f in files:
        #     os.remove(f)
        result_filename = os.path.join(root, seq + '.txt')

        # eval_seq(img_path, detector_path, result_filename, save_dir=None, frame_rate=frame_rate, width=width,
        #          height=height)
        p = multiprocessing.Process(target=eval_seq,
                                    args=(img_path, detector_path, result_filename, None, frame_rate, width, height,))
        processes.append(p)
        p.start()
    print("Waiting all processes to be finished................")
    for process in processes:
        process.join()
    for seq in seqs_train:
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(train_dir, seq, 'mot')
        accs.append(evaluator.eval_file(os.path.join(root, seq + '.txt')))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs_train, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(root, 'summary_{}.xlsx'.format(seqs_train[0].split('-')[0])))
    with open(os.path.join(root, 'summary_{}.txt'.format(seqs_train[0].split('-')[0])), 'w') as f:
        f.write(strsummary)
    processes = []
    for seq in seqs_test:
        logger.info('start seq: {}'.format(seq))
        meta_info = open(os.path.join(test_dir, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        width = int(meta_info[meta_info.find('imWidth') + 8:meta_info.find('\nimHeight')])
        height = int(meta_info[meta_info.find('imHeight') + 9:meta_info.find('\nimExt')])

        detector_path = os.path.join("../../detection", detector_dir, seq + '.npz')
        img_path = os.path.join(data_root, test_dir, seq)

        logger.info('Starting tracking...')
        # result_root = os.path.join(root, seq)
        # mkdir_if_missing(result_root)
        # files = glob.glob(result_root + '/*.jpg')
        # for f in files:
        #     os.remove(f)
        result_filename = os.path.join(root, seq + '.txt')

        # eval_seq(img_path, detector_path, result_filename, save_dir=None, frame_rate=frame_rate, width=width,
        #          height=height)
        p = multiprocessing.Process(target=eval_seq,
                                    args=(img_path, detector_path, result_filename, None, frame_rate, width, height,))
        processes.append(p)
        p.start()
    print("Waiting all processes to be finished................")
    for process in processes:
        process.join()


def mot17_eval(result_dir="mot16_glmb_cstrack", output_dir="mot17_glmb_cstrack"):
    data_root = 'D:/dataset/tracking/mot'
    train_dir, test_dir, seqs_train, seqs_test = mot17(data_root)
    seqs = ["02", "04", "05", "09", "10", "11", "13", "01", "03", "06", "07", "08", "12", "14"]
    sub_seqs = ["DPM", "SDP", "FRCNN"]
    for seq in seqs:
        input_file = "../../results/" + result_dir + "/MOT16-" + seq + ".txt"
        for sub_seq in sub_seqs:
            mkdir_if_missing("../../results/" + output_dir)
            output_file = "../../results/" + output_dir + "/MOT17-" + seq + "-" + sub_seq + ".txt"
            with open(input_file) as f:
                with open(output_file, "w") as f1:
                    for line in f:
                        f1.write(line)
    accs = []
    root = os.path.join("../../results/", output_dir)
    for seq in seqs_train:
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(train_dir, seq, 'mot')
        accs.append(evaluator.eval_file(os.path.join(root, seq + '.txt')))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs_train, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(root, 'summary_{}.xlsx'.format(seqs_train[0].split('-')[0])))
    with open(os.path.join(root, 'summary_{}.txt'.format(seqs_train[0].split('-')[0])), 'w') as f:
        f.write(strsummary)


if __name__ == '__main__':
    data_root = '/media/ubuntu/2715608D71CBF6FC/datasets/mot'
    demo(data_root, False, result_dir="mot16_glmb_fairmot256", detector_dir="detector_fairmot256")
    mot17_eval(result_dir="mot16_glmb_fairmot256", output_dir="mot17_glmb_fairmot256")
    demo(data_root, True, result_dir="mot20_detector_gsdt", detector_dir="detector_gsdt")
