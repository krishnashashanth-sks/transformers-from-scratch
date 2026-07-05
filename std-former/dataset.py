import torch
import torchvision.transforms as transforms

class DummyMovingMNIST(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, sequence_length=10, image_size=64, transform=None):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a dummy sequence of grayscale images as tensors in [0, 1]
        # Shape: (sequence_length, channels, height, width)
        # For Moving MNIST, channels = 1
        dummy_sequence_tensors = torch.rand(self.sequence_length, 1, self.image_size, self.image_size)

        transformed_frames = []
        for i in range(self.sequence_length):
            # Convert the tensor frame (1, H, W) to a PIL Image (H, W)
            # Scale to 0-255 implicitly by ToPILImage if needed, but the transform will handle normalization later.
            # We need to squeeze the channel dimension to make it (H, W) for ToPILImage if it's grayscale.
            frame_tensor_chw = dummy_sequence_tensors[i] # Current shape (1, H, W)

            # If transform is defined, apply it. ToPILImage expects a tensor, then Resize expects PIL.
            # So, we convert tensor (1,H,W) to PIL, then apply the transform.
            if self.transform:
                # Convert the float tensor in [0,1] to an 8-bit integer tensor for ToPILImage
                # This intermediate step is crucial for ToPILImage to correctly create a grayscale PIL image.
                frame_uint8 = (frame_tensor_chw * 255).to(torch.uint8)
                frame_pil = transforms.ToPILImage()(frame_uint8.squeeze(0)) # Squeeze for (H,W) PIL image
                transformed_frames.append(self.transform(frame_pil))
            else:
                transformed_frames.append(frame_tensor_chw)

        dummy_sequence_processed = torch.stack(transformed_frames)

        # Moving MNIST typically returns (sequence, sequence) where target is the same as input
        return dummy_sequence_processed, dummy_sequence_processed
