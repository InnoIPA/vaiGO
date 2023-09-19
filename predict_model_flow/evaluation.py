# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------------------
# Code of "Evaluate classification or detection performance"
# python version
# Written by Lu Tian
# --------------------------------------------------------

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def draw_pr(recall_point, precision_point, is_show=True):
    plt.plot(recall_point, precision_point, marker='o', linewidth = 1.5, ms=0.5)
    plt.title('PR_curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid(True)  # add grid
    plt.savefig('./pr.png')
    area = auc(recall_point, precision_point)
    print("Area Under PR Curve: ", area)
    if is_show:
        plt.show()

    plt.clf()

def draw_cm(true_positive, false_positive, false_negative, true_negative=0, is_show=True):
    # build a confusion_matrix
    confusion_matrix = np.array([[true_positive, false_positive],
                                [false_negative, true_negative]])

    # draw a confusion_matrix
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # create label
    classes = ["Postive", "Negative"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # How to draw a confusion_matrix
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.tight_layout()
    plt.savefig('./cm.png')
    if is_show:
        plt.show()
    plt.clf()

def compute_classification_accuracy(results, gts):
    """
    Evaluate classification results
    :param results: predicted results
    :param gts: ground truth
    :return: accuracy
    """
    num_label = len(gts[0].split(' ')) - 1
    image_label_gt = {}
    for gt in gts:
        gt_info = gt.split(' ')
        if len(gt_info) is not (num_label + 1):
            print ('label number does not match: ' + gt_info[0])
            return 0
        image_label_gt[gt_info[0]] = np.array(gt_info[1:])

    accuracy = np.zeros(num_label)
    count = 0
    image_names = set()
    for result in results:
        result_info = result.split(' ')
        if result_info[0] not in image_label_gt.keys():
            print ('could not find ground truth of image: ' + result_info[0])
            return 0
        if result_info[0] in image_names:
            print ('duplicate results of image: ' + result_info[0])
            return 0
        if len(result_info) is not (num_label + 1):
            print ('wrong predict label number of image: ' + result_info[0])
            return 0
        prediction = np.array(result_info[1:])
        accuracy += prediction == image_label_gt[result_info[0]]
        count += 1
        image_names.add(result_info[0])
    accuracy /= max(1, count)
    print ('evaluate ' + str(count) + ' images')
    return accuracy

def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    :param rec: recall
    :param prec: precision
    :param use_07_metric: uses the VOC 07 11 point method to compute VOC AP given precision and recall
    :return: ap
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        
        r_point = np.zeros(shape=11)
        p_point = np.zeros(shape=11)
        
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

            r_point[int(10*t)]=t
            p_point[int(10*t)]=p
        return ap, r_point, p_point
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def compute_detection_ap(results, gts, thresh, overlap_thresh, use_07_metric=False, pr_draw=False, cm_draw=False, cm_result_show=False):
    """
    Evaluate detection results
    :param results: image_name class_label score xmin ymin xmax ymax
    :param gts: image_name class_label xmin ymin xmax ymax
    :param thresh: only bboxes whose confidence score under thresh are used
    :param overlap_thresh: threshold of IOU ratio to determine a match bbox
    :param use_07_metric: uses the VOC 07 11 point method to compute VOC AP given precision and recall
    :return: recall, precision, ap
    """
    # load gt
    class_gts = {}
    class_num_positive = {}
    image_names = set()
    for gt in gts:
        gt_info = gt.split(' ')
        if len(gt_info) != 6 and len(gt_info) != 7:
            print('wrong ground truth info: ' + gt_info[0])
            return 0
        image_name = gt_info[0]
        #print(image_name)
        # class_name = gt_info[1]
        class_name = gt_info[5]
        #print(class_name)
        # bbox = [float(item) for item in gt_info[2:6]]
        bbox = [float(item) for item in gt_info[1:5]]
        #print(bbox)
        if len(gt_info) == 6:
            difficult = False
        else:
            difficult = bool(int(gt_info[-1]))

        if class_name not in class_gts.keys():
            class_gts[class_name] = {}
            class_num_positive[class_name] = 0
        if image_name not in class_gts[class_name].keys():
            class_gts[class_name][image_name] = {'bbox': np.array([bbox]),
                                                 'hit': [False],
                                                 'difficult': [difficult]}
        else:
            class_gts[class_name][image_name]['bbox'] = np.vstack((class_gts[class_name][image_name]['bbox'],
                                                                   np.array(bbox)))
            class_gts[class_name][image_name]['hit'].append(False)
            class_gts[class_name][image_name]['difficult'].append(difficult)
        class_num_positive[class_name] += int(True ^ difficult)
        image_names.add(image_name)
    class_names = class_gts.keys()

    # read dets
    class_dets = {}
    for result in results:
        result_info = result.split(' ')
        if len(result_info) != 7:
            print ('wrong detections info: ' + result_info[0])
            return 0
        image_name = result_info[0]
        class_name = result_info[1]
        bbox = [float(item) for item in result_info[2:]]
        if bbox[0] <= thresh:
            continue
        if class_name not in class_names:
            continue
        if class_name not in class_dets.keys():
            class_dets[class_name] = {'images': [],
                                      'scores': [],
                                      'bboxes': []}
        class_dets[class_name]['images'].append(image_name)
        class_dets[class_name]['scores'].append(bbox[0])
        class_dets[class_name]['bboxes'].append(bbox[1:])

    ap = {}
    precision = {}
    recall = {}
    f1_score = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_class = len(class_names)
    total_precision_point = np.zeros(shape=11)

    for class_name in class_names:
        if class_name not in class_dets.keys():
            ap[class_name] = 0
            recall[class_name] = 0
            precision[class_name] = 0
            continue

        gt_images = class_gts[class_name]
        num_positive = class_num_positive[class_name] # class gt num
        det_images = class_dets[class_name]['images']
        det_scores = np.array(class_dets[class_name]['scores'])
        det_bboxes = np.array(class_dets[class_name]['bboxes'])

        # sort by confidence
        sorted_index = np.argsort(-det_scores)
        det_bboxes = det_bboxes[sorted_index, :]
        det_images = [det_images[x] for x in sorted_index]

        # go down dets and mark TPs and FPs
        num_dets = len(det_images)
        true_positive = np.zeros(num_dets)
        false_positive = np.zeros(num_dets)
        for idx in range(num_dets):
            if det_images[idx] not in gt_images.keys():
                false_positive[idx] = 1
                continue

            gt_bboxes = gt_images[det_images[idx]]['bbox'].astype(float)
            gt_hit = gt_images[det_images[idx]]['hit']
            git_difficult = gt_images[det_images[idx]]['difficult']
            det_bbox = det_bboxes[idx, :].astype(float)
            overlaps_max = -np.inf

            if gt_bboxes.size > 0:
                # compute overlaps
                # intersection
                inter_xmin = np.maximum(gt_bboxes[:, 0], det_bbox[0])
                inter_ymin = np.maximum(gt_bboxes[:, 1], det_bbox[1])
                inter_xmax = np.minimum(gt_bboxes[:, 2], det_bbox[2])
                inter_ymax = np.minimum(gt_bboxes[:, 3], det_bbox[3])
                inter_width = np.maximum(inter_xmax - inter_xmin + 1., 0.)
                inter_height = np.maximum(inter_ymax - inter_ymin + 1., 0.)
                inters = inter_width * inter_height

                # union
                unions = ((det_bbox[2] - det_bbox[0] + 1.) * (det_bbox[3] - det_bbox[1] + 1.) +
                          (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.) - inters)

                overlaps = inters / unions
                overlaps_max = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if overlaps_max > overlap_thresh:
                if not git_difficult[jmax]:
                    if not gt_hit[jmax]:
                        true_positive[idx] = 1.
                        gt_hit[jmax] = 1
                    else:
                        false_positive[idx] = 1.
            else:
                false_positive[idx] = 1.

        # compute precision recall
        false_positive = np.cumsum(false_positive)
        true_positive = np.cumsum(true_positive)
        false_negative = num_positive - int(true_positive[-1])
        recall[class_name] = true_positive / float(num_positive) #  same -> true_positive / (true_positive + false_negative)
        precision[class_name] = true_positive / np.maximum(true_positive + false_positive, np.finfo(np.float64).eps)
        f1_score = (2*(precision[class_name]*recall[class_name]))/(precision[class_name]+recall[class_name])
        if cm_result_show:
            print("class name: {}".format(class_name))
            print("true_positive: {}".format(int(true_positive[-1])))
            print("false_positive: {}".format(int(false_positive[-1])))
            print("false_negative: {}".format(false_negative))
            print("f1_score: {}".format(f1_score[-1]))
            print("--------------------")
       
        if use_07_metric:
            ap[class_name], r_point, p_point  = voc_ap(recall[class_name], precision[class_name], use_07_metric)
            total_precision_point = total_precision_point + p_point

        else:
            ap[class_name] = voc_ap(recall[class_name], precision[class_name], use_07_metric)
            print ('evaluate ' + str(len(image_names)) + ' images')

        total_tp = total_tp + int(true_positive[-1])
        total_fp = total_fp + int(false_positive[-1])
        total_fn = total_fn + false_negative
        total_pre = total_tp/(total_tp+total_fp)
        total_rec = total_tp/(total_tp+total_fn)
        total_f1_score = (2*(total_pre*total_rec))/(total_pre+total_rec)

    if pr_draw:
        draw_pr(r_point, total_precision_point/total_class)
    if cm_draw:
        draw_cm(total_tp, total_fp, total_fn)
    
    print("====================")
    print("total:")
    print('evaluate ' + str(len(image_names)) + ' images')
    print("true_positive: {}".format(total_tp))
    print("false_positive: {}".format(total_fp))
    print("false_negative: {}".format(total_fn))
    print("f1_score: {}".format(total_f1_score))
    print("====================")
    return recall, precision, ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate classification or detection performance')
    parser.add_argument('-mode', default='detection',
                        help='mode, detection or classification, default detection')
    parser.add_argument('-result_file', default='',
                        help="""Result file in space-separated text format.
                        For classification, each row is: image_id label [label ...].
                        For detection, each row is: image_id label score xmin ymin xmax ymax [difficult_bool].""")
    parser.add_argument('-gt_file', default='',
                        help="""Ground truth file in space-separated text format. 
                        For classification, each row is: image_id label [label ...].
                        For detection, each row is: image_id label xmin ymin xmax ymax.""")
    parser.add_argument('-detection_metric', default='map',
                        help="""Evaluation metric for detection, default map.
                        Options are map (mean average precision), precision (given recall), recall (given precision), 
                        pr (precision and recall given threshold of confidence score).""")
    parser.add_argument('-detection_iou', default='0.5',
                        help="""Threshold of IOU ratio to
                         determine a match bbox.""")
    parser.add_argument('-detection_thresh', default='0.005',
                        help="""Threshold of confidence score for calculating evaluation metric, default 0.05.
                        For metric = pr, detection_thresh should be the confidence score to determine a positive bbox.
                        For other detection metrics, detection_thresh should be a very small value.""")
    parser.add_argument('-detection_fix_recall', default='0.8',
                        help="""Used when detection_metric is precision, default 0.8.""")
    parser.add_argument('-detection_fix_precision', default='0.8',
                        help="""Used when detection_metric is recall, default 0.8.""")
    parser.add_argument('-detection_use_07_metric', default='True',
                        help="""Uses the VOC 07 11 point method to compute VOC AP given precision and recall.""")
    parser.add_argument('-draw_pr', default='True',
                        help="""Uses the VOC 07 point method to draw VOC AP precision and recall curve.""")
    parser.add_argument('-draw_cm', default='True',
                        help="""Uses the VOC 07 point method to draw confusion matrix.""")
    parser.add_argument('-cm_result_show', default='True',
                        help="""If you want to see each category of confusion matrix info and F1-Score.""")

    args = parser.parse_args()

    results_file = open(args.result_file, 'r')
    #fix for pytyhon3 JimH
    #results_lines = filter(None, [item.strip() for item in results_file.readlines()])
    results_lines = list(filter(None, [item.strip() for item in results_file.readlines()]))
    gts_file = open(args.gt_file, 'r')
    #fix for python3 Jimh
    #gts_lines = filter(None, [item.strip() for item in gts_file.readlines()])
    gts_lines = list(filter(None, [item.strip() for item in gts_file.readlines()]))
    if len(gts_lines) < 1:
        print ('ground truth file is empty!')
    if len(results_lines) < 1:
        print ('result file is empty!')
    if args.mode == 'classification':
        accuracy = compute_classification_accuracy(results_lines, gts_lines)
        print ('classification accuracy of each class: ' + str(accuracy))
        print ('mean classification accuracy: ' + str(np.mean(accuracy)))
    elif args.mode == 'detection':
        detection_thresh = float(args.detection_thresh)
        detection_iou = float(args.detection_iou)
        use_07_metric = False
        pr_draw = False
        cm_draw = False
        cm_result_show = False
        if args.detection_use_07_metric == 'True':
            use_07_metric = True
        if args.draw_pr == 'True':
            pr_draw = True
        if args.draw_cm == 'True':
            cm_draw = True
        if args.cm_result_show == 'True':
            cm_result_show = True
        recall, precision, ap = compute_detection_ap(results_lines, gts_lines, detection_thresh, detection_iou,
                                                    use_07_metric, pr_draw, cm_draw, cm_result_show)
        if args.detection_metric == 'map':
            for class_name in ap.keys():
                print (class_name + ' AP: ' + str(ap[class_name]))
            print ('mAP: ' + str((float(sum(ap.values()))) / max(1, len(ap))))
        elif args.detection_metric == 'precision':
            fix_recall = float(args.detection_fix_recall)
            for class_name in ap.keys():
                if np.sum(recall[class_name] >= fix_recall) == 0:
                    output_precision = 0
                else:
                    output_precision = np.max(precision[class_name][recall[class_name] >= fix_recall])
                print (class_name + ', set recall is ' + str(fix_recall) + ', precision: ' + str(output_precision))
        elif args.detection_metric == 'recall':
            fix_precision = float(args.detection_fix_precision)
            for class_name in ap.keys():
                if np.sum(precision[class_name] >= fix_precision) == 0:
                    output_recall = 0
                else:
                    output_recall = np.max(recall[class_name][precision[class_name] >= fix_precision])
                print (class_name + ', set precision is ' + str(fix_precision) + ', recall: ' + str(output_recall))
        elif args.detection_metric == 'pr':
            for class_name in ap.keys():
                if len(recall[class_name]) > 0:
                    output_recall = recall[class_name][-1]
                else:
                    output_recall = 0
                if np.sum(recall[class_name] >= output_recall) == 0:
                    output_precision = 0
                else:
                    output_precision = np.max(precision[class_name][recall[class_name] >= output_recall])
                print (class_name + ', set confidence score is ' + str(detection_thresh) + \
                      ', precision: ' + str(output_precision) + ', recall: ' + str(output_recall))
        else:
            print ('wrong evaluation metric!')
    results_file.close()
    gts_file.close()