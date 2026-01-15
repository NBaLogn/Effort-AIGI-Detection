from sklearn import metrics
import numpy as np
import os


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return ""
    if isinstance(metric_dict, (float, int, np.float32, np.float64)):
        return f"{metric_dict:.4f}"

    res = "\n"
    res += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != "avg":
            res = res + f"| {key}: "
            for k, v in value.items():
                res = res + f" {k}={v} "
            res = res + "| \n"
        else:
            res += "============================================================================================= \n"
            res += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == "dataset_dict":
                    for key, value in avg_value.items():
                        res = res + f"| {key}: {value} | \n"
                else:
                    res = res + f"| avg {avg_key}: {avg_value} | \n"
    res += "============================================================================================="
    return res


# def get_test_metrics(y_pred, y_true, img_names):
#     def get_video_metrics(image, pred, label):
#         result_dict = {}
#         new_label = []
#         new_pred = []
#         # print(image[0])
#         # print(pred.shape)
#         # print(label.shape)
#         for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
#             # 分割字符串，获取'a'和'b'的值
#             s = item[0]
#             if '\\' in s:
#                 parts = s.split('\\')
#             else:
#                 parts = s.split('/')
#             a = parts[-2]
#             b = parts[-1]

#             # 如果'a'的值还没有在字典中，添加一个新的键值对
#             if a not in result_dict:
#                 result_dict[a] = []

#             # 将'b'的值添加到'a'的列表中
#             result_dict[a].append(item)
#         image_arr = list(result_dict.values())
#         # 将字典的值转换为一个列表，得到二维数组

#         for video in image_arr:
#             pred_sum = 0
#             label_sum = 0
#             leng = 0
#             for frame in video:
#                 pred_sum += float(frame[1])
#                 label_sum += int(frame[2])
#                 leng += 1
#             new_pred.append(pred_sum / leng)
#             new_label.append(int(label_sum / leng))
#         fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
#         v_auc = metrics.auc(fpr, tpr)
#         fnr = 1 - tpr
#         v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#         return v_auc, v_eer


#     y_pred = y_pred.squeeze()
#     # auc
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
#     auc = metrics.auc(fpr, tpr)
#     # eer
#     fnr = 1 - tpr
#     eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#     # ap
#     ap = metrics.average_precision_score(y_true, y_pred)
#     # acc
#     prediction_class = (y_pred > 0.5).astype(int)
#     correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
#     acc = correct / len(prediction_class)
#     if type(img_names[0]) is not list:
#         # calculate video-level auc for the frame-level methods.
#         try:
#             v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
#         except Exception as e:
#             print(e)
#             v_auc=auc
#     else:
#         # video-level methods
#         v_auc=auc

#     return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
            s = item[0]
            if "\\" in s:
                parts = s.split("\\")
            else:
                parts = s.split("/")

            # Robust video name extraction
            # Structure possibilities:
            # 1. .../real/video1/frame1.jpg -> parts[-2] == 'video1'
            # 2. .../real/frames/video1/frame1.jpg -> parts[-2] == 'video1'
            # 3. .../real/frame1.jpg -> parts[-2] == 'real' (standalone)

            # Actually, using the directory path as the key is safest.
            video_dir = os.path.dirname(s)

            if video_dir not in result_dict:
                result_dict[video_dir] = []

            result_dict[video_dir].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))

        unique_labels = np.unique(new_label)
        if len(unique_labels) < 2:
            v_auc = 0.0
            v_eer = 0.0
        else:
            fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
            v_auc = metrics.auc(fpr, tpr)
            fnr = 1 - tpr
            v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        # Calculate video-level acc
        prediction_class = (np.array(new_pred) > 0.5).astype(int)
        correct = (prediction_class == np.array(new_label)).sum().item()
        v_acc = correct / len(prediction_class)

        return v_auc, v_eer, v_acc

    y_pred = y_pred.squeeze()
    # Check if both classes are present
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        auc = 0.0
        eer = 0.0
        ap = 0.0
    else:
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        try:
            v_auc, v_eer, v_acc = get_video_metrics(img_names, y_pred, y_true)
            return {
                "acc": acc,
                "auc": auc,
                "eer": eer,
                "ap": ap,
                "pred": y_pred,
                "video_auc": v_auc,
                "video_eer": v_eer,
                "video_acc": v_acc,
                "label": y_true,
            }
        except Exception as e:
            print(e)
            v_auc = auc
            return {
                "acc": acc,
                "auc": auc,
                "eer": eer,
                "ap": ap,
                "pred": y_pred,
                "label": y_true,
            }
    else:
        # video-level methods
        v_auc = auc
        v_eer = eer
        v_acc = acc
        return {
            "acc": acc,
            "auc": auc,
            "eer": eer,
            "ap": ap,
            "pred": y_pred,
            "video_auc": v_auc,
            "video_eer": v_eer,
            "video_acc": v_acc,
            "label": y_true,
        }
