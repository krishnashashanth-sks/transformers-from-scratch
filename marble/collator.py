import torch

def collate_fn(batch):
    # Pad text sequences to the maximum length in the batch (though already padded to max_seq_len in Dataset)
    # If max_seq_len was dynamic, we would pad here.
    tokenized_text = torch.stack([item['tokenized_text'] for item in batch])

    # Stack fixed-size tensors
    ref_image = torch.stack([item['ref_image'] for item in batch])
    dummy_nerf_latent = torch.stack([item['dummy_nerf_latent'] for item in batch])

    # Handle camera intrinsics: they are dictionaries, stack their values if they are tensors
    ref_intrinsic_batch = {
        k: torch.stack([item['ref_intrinsic'][k] for item in batch])
        for k in batch[0]['ref_intrinsic'].keys()
    }
    ref_extrinsic = torch.stack([item['ref_extrinsic'] for item in batch])

    nerf_intrinsic_batch = {
        k: torch.stack([item['nerf_intrinsic'][k] for item in batch])
        for k in batch[0]['nerf_intrinsic'].keys()
    }

    # Handle variable-length lists of NeRF images and extrinsics
    # Find the maximum number of views in the current batch
    max_nerf_views = max([item['nerf_images'].shape[0] for item in batch])

    padded_nerf_images = []
    padded_nerf_extrinsics = []
    for item in batch:
        num_views = item['nerf_images'].shape[0]
        # Pad images
        padding_needed_img = max_nerf_views - num_views
        if padding_needed_img > 0:
            pad_shape_img = list(item['nerf_images'].shape)
            pad_shape_img[0] = padding_needed_img
            # Use black/transparent padding for images (all zeros)
            padding_img = torch.zeros(pad_shape_img, dtype=item['nerf_images'].dtype, device=item['nerf_images'].device)
            padded_nerf_images.append(torch.cat([item['nerf_images'], padding_img], dim=0))
        else:
            padded_nerf_images.append(item['nerf_images'])

        # Pad extrinsics
        padding_needed_ext = max_nerf_views - num_views
        if padding_needed_ext > 0:
            pad_shape_ext = list(item['nerf_extrinsics'].shape)
            pad_shape_ext[0] = padding_needed_ext
            # Use identity matrix for padding extrinsics, or simply zeros if they won't be used
            # For simplicity, using zeros which would typically be filtered out by an attention mask.
            padding_ext = torch.zeros(pad_shape_ext, dtype=item['nerf_extrinsics'].dtype, device=item['nerf_extrinsics'].device)
            padded_nerf_extrinsics.append(torch.cat([item['nerf_extrinsics'], padding_ext], dim=0))
        else:
            padded_nerf_extrinsics.append(item['nerf_extrinsics'])

    nerf_images = torch.stack(padded_nerf_images)
    nerf_extrinsics = torch.stack(padded_nerf_extrinsics)

    return {
        'tokenized_text': tokenized_text,
        'ref_image': ref_image,
        'ref_extrinsic': ref_extrinsic,
        'ref_intrinsic': ref_intrinsic_batch,
        'nerf_images': nerf_images,
        'nerf_extrinsics': nerf_extrinsics,
        'nerf_intrinsic': nerf_intrinsic_batch,
        'dummy_nerf_latent': dummy_nerf_latent
    }