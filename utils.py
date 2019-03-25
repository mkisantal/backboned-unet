import torch


def iou(predictions, labels, threshold=None, average=True, device=torch.device("cpu"), classes=21):

    """ Calculating Intersection over Union score for semantic segmentation. """

    gt = (labels * 255.0).long().to(device)  # torchvision normalizes gt images, restoring integer values here

    # getting mask for valid pixels, then converting "void class" to background
    valid = gt != 255
    gt[gt == 255] = 0

    # converting to onehot image whith class channels
    onehot_gt_tensor = torch.LongTensor(gt.shape[0], classes, gt.shape[-2], gt.shape[-1]).zero_().to(device)
    onehot_gt_tensor.scatter_(1, gt, 1)  # write ones along "channel" dimension
    classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0

    if threshold is None:
        # taking the argmax along channels
        pred = torch.argmax(predictions, dim=1).unsqueeze(1)
        pred_tensor = torch.LongTensor(pred.shape[0], classes, pred.shape[-2], pred.shape[-1]).zero_().to(device)
        pred_tensor.scatter_(1, pred, 1)
    else:
        # counting predictions above threshold
        pred_tensor = (predictions > threshold).long()

    onehot_gt_tensor *= valid.long()
    pred_tensor *= valid.long().to(device)

    intersection = (pred_tensor & onehot_gt_tensor).sum([2, 3]).float()
    union = (pred_tensor | onehot_gt_tensor).sum([2, 3]).float()

    iou = intersection / (union + 1e-12)

    if average:
        average_iou = iou[:, 1:].sum(dim=1) / (classes_in_image[:, 1:].sum(dim=1)).float()  # discard background IoU
        iou = average_iou

    return iou.cpu().numpy()


if __name__ == "__main__":
    predictions = torch.empty(7, 21, 224, 224).normal_()
    labels = (torch.empty(7, 1, 224, 224).normal_(10, 10) % 21)/255.0
    out = iou(predictions, labels)
    print('done.')
