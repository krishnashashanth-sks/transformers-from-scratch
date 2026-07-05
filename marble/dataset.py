from torch.utils.data import Dataset
import torch
import os
from collections import Counter
from PIL import Image
import random
import json
import numpy as np

class SyntheticWorldDataset(Dataset):
    def __init__(self, base_output_dir, max_vocab_size=5000, min_freq=5,max_seq_len=5,img_size=256,latent_dim_neRF=128):
        self.base_output_dir = base_output_dir
        self.scene_paths = []
        for scene_id in sorted(os.listdir(base_output_dir)):
            scene_path = os.path.join(base_output_dir, scene_id)
            if os.path.isdir(scene_path):
                self.scene_paths.append(scene_path)

        if not self.scene_paths:
            raise ValueError(f"No scene directories found in {base_output_dir}")
        
        self.img_size=img_size
        self.self.max_seq_len=max_seq_len
        self.latent_dim_neRF=latent_dim_neRF
        self.vocab_map = self._build_vocab(max_vocab_size, min_freq)
        self.vocab_size = len(self.vocab_map)
        print(f"Built vocabulary with {self.vocab_size} tokens.")

        self.transform = self._get_image_transform()

    def _build_vocab(self, max_vocab_size, min_freq):
        all_words = []
        for scene_path in self.scene_paths:
            text_path = os.path.join(scene_path, 'text_description.txt')
            with open(text_path, 'r') as f:
                text = f.read().lower()
                all_words.extend(text.replace('.', '').replace(',', '').split())

        word_counts = Counter(all_words)
        # Filter by min_freq and then take top max_vocab_size
        filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
        sorted_words = sorted(filtered_words, key=lambda x: word_counts[x], reverse=True)

        vocab = {'<pad>': 0, '<unk>': 1}
        for word in sorted_words:
            if len(vocab) < max_vocab_size:
                vocab[word] = len(vocab)
            else:
                break
        return vocab

    def _tokenize_text(self, text):
        tokens = []
        words = text.lower().replace('.', '').replace(',', '').split()
        for word in words:
            tokens.append(self.vocab_map.get(word, self.vocab_map['<unk>']))
        # Pad or truncate
        if len(tokens) < self.self.max_seq_len:
            tokens.extend([self.vocab_map['<pad>']] * (self.max_seq_len - len(tokens)))
        else:
            tokens = tokens[:self.self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

    def _get_image_transform(self):
        # Simple transform for now: resize, convert to tensor, normalize to [0, 1]
        def transform_fn(image_path):
            img = Image.open(image_path).convert('RGBA')
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0 # Normalize to [0, 1]
            img = torch.tensor(img).permute(2, 0, 1) # HWC to CHW
            return img
        return transform_fn

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]

        # a. Load and tokenize text description
        text_file = os.path.join(scene_path, 'text_description.txt')
        with open(text_file, 'r') as f:
            text_description = f.read().strip()
        tokenized_text = self._tokenize_text(text_description)

        # b. Load a random reference image and its camera parameters
        ref_images_dir = os.path.join(scene_path, 'reference_images')
        ref_image_files = [f for f in os.listdir(ref_images_dir) if f.endswith('.png')]
        if not ref_image_files:
            raise RuntimeError(f"No reference images found in {ref_images_dir}")
        random_ref_image_file = random.choice(ref_image_files)
        ref_image_path = os.path.join(ref_images_dir, random_ref_image_file)
        processed_ref_image = self.transform(ref_image_path)

        with open(os.path.join(ref_images_dir, 'camera_params.json'), 'r') as f:
            ref_camera_params = json.load(f)
        # Find the extrinsic for the chosen random reference image
        # Assuming naming 'reference_image_X.png' corresponds to X-th extrinsic in list
        ref_idx = int(random_ref_image_file.split('_')[-1].split('.')[0])
        ref_extrinsic = torch.tensor(ref_camera_params['extrinsics'][ref_idx], dtype=torch.float32)
        ref_intrinsic = {k: torch.tensor(v, dtype=torch.float32) for k,v in ref_camera_params['intrinsics'].items()}

        # c. Load a set of multi-view images (e.g., 5-10 random views) and preprocess them
        nerf_data_dir = os.path.join(scene_path, 'nerf_data')
        nerf_image_files = [f for f in os.listdir(nerf_data_dir) if f.endswith('.png')]
        if not nerf_image_files:
            raise RuntimeError(f"No NeRF multi-view images found in {nerf_data_dir}")

        num_views_to_load = min(10, len(nerf_image_files)) # Load up to 10 views
        selected_nerf_image_files = random.sample(nerf_image_files, num_views_to_load)

        processed_nerf_images = []
        nerf_extrinsics = []
        with open(os.path.join(nerf_data_dir, 'camera_params.json'), 'r') as f:
            nerf_camera_params = json.load(f)
        nerf_intrinsic = {k: torch.tensor(v, dtype=torch.float32) for k,v in nerf_camera_params['intrinsics'].items()}

        # Extract the index from the filename to match with extrinsics
        all_nerf_extrinsics_np = np.array(nerf_camera_params['extrinsics'])

        for f_name in selected_nerf_image_files:
            img_path = os.path.join(nerf_data_dir, f_name)
            processed_nerf_images.append(self.transform(img_path))
            # Filenames are 'view_XXXX.png'
            view_idx = int(f_name.split('_')[-1].split('.')[0])
            nerf_extrinsics.append(torch.tensor(all_nerf_extrinsics_np[view_idx], dtype=torch.float32))

        processed_nerf_images = torch.stack(processed_nerf_images) # (num_views, C, H, W)
        nerf_extrinsics = torch.stack(nerf_extrinsics) # (num_views, 4, 4)

        # e. Create a dummy NeRF latent code
        dummy_nerf_latent = torch.randn(self.latent_dim_neRF, dtype=torch.float32)

        return {
            'tokenized_text': tokenized_text,
            'ref_image': processed_ref_image,
            'ref_extrinsic': ref_extrinsic,
            'ref_intrinsic': ref_intrinsic,
            'nerf_images': processed_nerf_images,
            'nerf_extrinsics': nerf_extrinsics,
            'nerf_intrinsic': nerf_intrinsic,
            'dummy_nerf_latent': dummy_nerf_latent
        }