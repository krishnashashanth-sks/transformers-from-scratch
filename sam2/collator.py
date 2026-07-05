import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    # Determine max lengths for variable sequences in this batch
    max_points_len = max([item['points'].shape[0] for item in batch])
    max_boxes_len = max([item['boxes'].shape[0] for item in batch])

    batched_data = {
        'image': torch.stack([item['image'] for item in batch]),
        'ground_truth_mask': torch.stack([item['ground_truth_mask'] for item in batch]),
        'original_image_size': torch.stack([item['original_image_size'] for item in batch]),
    }

    # Pad points, point_labels, boxes
    padded_points = []
    padded_point_labels = []
    padded_boxes = []
    points_mask = [] # True for actual data, False for padding
    boxes_mask = []

    for item in batch:
        # Points
        current_points = item['points']
        current_point_labels = item['point_labels']
        current_points_len = current_points.shape[0]

        if current_points_len < max_points_len:
            padding_len = max_points_len - current_points_len
            # Pad coordinates with 0, labels with -1 (or any unused value)
            padded_points.append(F.pad(current_points, (0, 0, 0, padding_len), 'constant', 0.))
            padded_point_labels.append(F.pad(current_point_labels, (0, padding_len), 'constant', -1))
            points_mask.append(torch.cat([torch.ones(current_points_len, dtype=torch.bool),
                                          torch.zeros(padding_len, dtype=torch.bool)]))
        else:
            padded_points.append(current_points)
            padded_point_labels.append(current_point_labels)
            points_mask.append(torch.ones(current_points_len, dtype=torch.bool))

        # Boxes
        current_boxes = item['boxes']
        current_boxes_len = current_boxes.shape[0]

        if current_boxes_len < max_boxes_len:
            padding_len = max_boxes_len - current_boxes_len
            # Pad coordinates with 0
            padded_boxes.append(F.pad(current_boxes, (0, 0, 0, padding_len), 'constant', 0.))
            boxes_mask.append(torch.cat([torch.ones(current_boxes_len, dtype=torch.bool),
                                         torch.zeros(padding_len, dtype=torch.bool)]))
        else:
            padded_boxes.append(current_boxes)
            boxes_mask.append(torch.ones(current_boxes_len, dtype=torch.bool))

    batched_data['points'] = torch.stack(padded_points)
    batched_data['point_labels'] = torch.stack(padded_point_labels)
    batched_data['boxes'] = torch.stack(padded_boxes)
    batched_data['points_mask'] = torch.stack(points_mask)
    batched_data['boxes_mask'] = torch.stack(boxes_mask)

    return batched_data
