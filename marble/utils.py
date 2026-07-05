import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np
from math import log10
import json
import shutil
import pyrender
import trimesh
from PIL import Image
import os
import random

os.environ['PYOPENGL_PLATFORM'] = 'egl' # Set the platform to EGL for headless rendering

def generate_procedural_scene(max_objects=5):
    """
    Generates a synthetic 3D scene by placing random geometric primitives.

    Args:
        max_objects (int): The maximum number of objects to place in the scene.

    Returns:
        trimesh.Scene: A trimesh Scene object containing the generated 3D world.
        list: A list of dictionaries, each describing an object in the scene.
    """
    scene = trimesh.Scene()
    scene_description = []

    num_objects = random.randint(1, max_objects)

    object_types = ['box', 'sphere', 'cylinder']
    colors = {
        'red': [1.0, 0.0, 0.0, 1.0],
        'green': [0.0, 1.0, 0.0, 1.0],
        'blue': [0.0, 0.0, 1.0, 1.0],
        'yellow': [1.0, 1.0, 0.0, 1.0],
        'purple': [0.5, 0.0, 0.5, 1.0],
        'orange': [1.0, 0.5, 0.0, 1.0]
    }
    color_names = list(colors.keys())

    for i in range(num_objects):
        obj_type = random.choice(object_types)
        color_name = random.choice(color_names)
        color_rgba = colors[color_name]

        # Random position (within a reasonable bound)
        position = np.random.uniform(-2, 2, size=3)
        # Random scale
        scale = np.random.uniform(0.2, 1.0)
        # Random rotation (Euler angles, then convert to matrix)
        rotation_angles = np.random.uniform(0, 2 * np.pi, size=3)
        rotation_matrix = trimesh.transformations.euler_matrix(
            rotation_angles[0], rotation_angles[1], rotation_angles[2], 'sxyz'
        )[:3,:3]

        if obj_type == 'box':
            mesh = trimesh.creation.box(extents=[scale, scale, scale])
        elif obj_type == 'sphere':
            mesh = trimesh.creation.icosphere(radius=scale)
        elif obj_type == 'cylinder':
            mesh = trimesh.creation.cylinder(radius=scale * 0.5, height=scale)

        # Apply color to the mesh
        mesh.visual.face_colors = color_rgba

        # Create a transform matrix for position and rotation
        transform = trimesh.transformations.identity_matrix()
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position

        mesh.apply_transform(transform)
        scene.add_geometry(mesh, geom_name=f"{color_name}_{obj_type}_{i}")

        scene_description.append({
            'type': obj_type,
            'color': color_name,
            'position': position.tolist(),
            'scale': scale,
            'rotation': rotation_angles.tolist()
        })

    print(f"Generated scene with {num_objects} objects.")
    return scene, scene_description

def generate_text_description(scene_description):
    """
    Generates a descriptive string for a 3D scene based on its object descriptions.

    Args:
        scene_description (list): A list of dictionaries, each describing an object in the scene.

    Returns:
        str: A natural language description of the scene.
    """
    if not scene_description:
        return "An empty scene."

    descriptions = []
    for i, obj in enumerate(scene_description):
        obj_type = obj['type']
        color = obj['color']
        position = obj['position']
        scale = obj['scale']

        desc = f"a {color} {obj_type} (scale: {scale:.2f}) at position ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
        descriptions.append(desc)

    if len(descriptions) == 1:
        return f"A scene containing {descriptions[0]}.\n"
    else:
        first_part = ", ".join(descriptions[:-1])
        return f"A scene containing {first_part}, and {descriptions[-1]}.\n"

def generate_reference_images(trimesh_scene, output_dir='reference_images', num_cameras=3, resolution=(256, 256)):
    """
    Generates reference images from fixed camera poses using pyrender.

    Args:
        trimesh_scene (trimesh.Scene): The 3D scene to render.
        output_dir (str): Directory to save the reference images.
        num_cameras (int): Number of fixed camera poses to use (1-3).
        resolution (tuple): Resolution of the rendered images (width, height).

    Returns:
        list: A list of paths to the saved reference images.
        list: A list of camera pose matrices (extrinsics).
        dict: A dictionary containing camera intrinsic parameters.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup pyrender scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0])) # Transparent background

    # Add individual trimesh geometries to pyrender scene
    for geometry_name, geometry in trimesh_scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            # Ensure geometry has vertex_colors or face_colors, or assign a default
            if geometry.visual.vertex_colors is not None and len(geometry.visual.vertex_colors) > 0:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=geometry.visual.vertex_colors[0].tolist())
            elif geometry.visual.face_colors is not None and len(geometry.visual.face_colors) > 0:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=geometry.visual.face_colors[0].tolist())
            else:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 1.0, 1.0, 1.0]) # Default white

            mesh = pyrender.Mesh.from_trimesh(geometry, material=material)
            scene.add(mesh)

    # Add a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=np.eye(4))

    # Camera intrinsics (standard perspective camera)
    fx, fy, cx, cy = resolution[0], resolution[1], resolution[0]/2, resolution[1]/2
    camera_intrinsics = {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'width': resolution[0], 'height': resolution[1]
    }
    main_pyrender_camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

    # Add a single camera to the scene, its pose will be updated per render
    camera_node = scene.add(main_pyrender_camera, pose=np.eye(4), name='main_render_camera')

    # Fixed camera poses (extrinsics - world_to_camera matrices for pyrender, will convert to camera_to_world for output)
    # Each row is [x, y, z, r_x, r_y, r_z] for camera position and Euler angles for orientation
    camera_configs = [
        ([0.0, 0.0, 5.0], [0, 0, 0]), # Front view
        ([3.0, 3.0, 3.0], [np.deg2rad(-30), np.deg2rad(30), 0]), # Angled view 1
        ([-3.0, 3.0, 3.0], [np.deg2rad(-30), np.deg2rad(-30), 0]) # Angled view 2
    ]
    camera_poses_world_to_camera = []
    camera_poses_camera_to_world = []

    for i in range(min(num_cameras, len(camera_configs))):
        pos, rot = camera_configs[i]
        # Create camera_to_world matrix
        camera_to_world = trimesh.transformations.compose_matrix(translate=pos, angles=rot)

        # Invert to get world_to_camera for pyrender
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_poses_world_to_camera.append(world_to_camera)
        camera_poses_camera_to_world.append(camera_to_world) # Store camera_to_world for NeRF compatibility

    # Render the scene
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

    saved_image_paths = []
    for i in range(min(num_cameras, len(camera_configs))):
        # Update the pose of the single camera node for each render
        camera_node.matrix = camera_poses_world_to_camera[i]

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES)

        img_path = os.path.join(output_dir, f'reference_image_{i}.png')
        Image.fromarray(color).save(img_path)
        saved_image_paths.append(img_path)

    renderer.delete()

    print(f"Generated {len(saved_image_paths)} reference images in '{output_dir}'.")
    return saved_image_paths, camera_poses_camera_to_world, camera_intrinsics

def generate_nerf_multiview_data(trimesh_scene, output_dir='nerf_multiview_data', num_views=50, resolution=(256, 256), radius_range=(3.0, 7.0), fov_y=np.pi / 3.0):
    """
    Generates a set of multi-view images and camera parameters for NeRF training.

    Args:
        trimesh_scene (trimesh.Scene): The 3D scene to render.
        output_dir (str): Directory to save the multi-view images and camera data.
        num_views (int): Number of random camera views to generate.
        resolution (tuple): Resolution of the rendered images (width, height).
        radius_range (tuple): Min and max distance of cameras from the scene origin.
        fov_y (float): Vertical field of view for the camera in radians.

    Returns:
        list: A list of paths to the saved multi-view images.
        list: A list of camera pose matrices (camera_to_world extrinsics).
        dict: A dictionary containing camera intrinsic parameters.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup pyrender scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]))  # Transparent background

    # Add individual trimesh geometries to pyrender scene
    for geometry_name, geometry in trimesh_scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            if geometry.visual.vertex_colors is not None and len(geometry.visual.vertex_colors) > 0:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=geometry.visual.vertex_colors[0].tolist())
            elif geometry.visual.face_colors is not None and len(geometry.visual.face_colors) > 0:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=geometry.visual.face_colors[0].tolist())
            else:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 1.0, 1.0, 1.0])  # Default white

            mesh = pyrender.Mesh.from_trimesh(geometry, material=material)
            scene.add(mesh)

    # Add a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=np.eye(4))

    # Camera intrinsics (fixed for all views)
    # pyrender.IntrinsicsCamera uses fx, fy, cx, cy
    # For a perspective camera, fx = fy = 0.5 * width / tan(fov_x / 2), and cx = width / 2
    # We'll calculate fx, fy based on a vertical FOV (fov_y)
    aspect_ratio = resolution[0] / resolution[1]
    focal_length_y = resolution[1] / (2.0 * np.tan(fov_y / 2.0))
    fx = focal_length_y * aspect_ratio
    fy = focal_length_y
    cx = resolution[0] / 2.0
    cy = resolution[1] / 2.0

    camera_intrinsics = {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'width': resolution[0], 'height': resolution[1]
    }
    pyrender_camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    camera_node = scene.add(pyrender_camera, pose=np.eye(4), name='multiview_camera')

    # Initialize renderer once
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

    saved_image_paths = []
    camera_poses_camera_to_world = []

    for i in range(num_views):
        # Random camera position (spherical coordinates -> cartesian)
        radius = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        phi = np.random.uniform(np.pi / 6, np.pi / 2)  # Polar angle (from up, avoiding straight top/bottom views)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        camera_position = np.array([x, y, z])

        # Look at origin (0,0,0) - target point
        target_point = np.array([0.0, 0.0, 0.0])
        up_vector = np.array([0.0, 0.0, 1.0]) # Z-up

        # --- Manual construction of camera_to_world matrix (replacing trimesh.transformations.look_at) ---
        # Camera's local Z-axis (forward vector, points from camera to target)
        z_axis_world = target_point - camera_position
        z_axis_world = z_axis_world / np.linalg.norm(z_axis_world)

        # Camera's local X-axis (right vector), orthogonal to world up and camera forward
        x_axis_world = np.cross(up_vector, z_axis_world)
        x_axis_world = x_axis_world / np.linalg.norm(x_axis_world)

        # Camera's local Y-axis (up vector), orthogonal to camera forward and right
        y_axis_world = np.cross(z_axis_world, x_axis_world) # Re-orthogonalize up vector
        y_axis_world = y_axis_world / np.linalg.norm(y_axis_world)

        # Construct rotation matrix (columns are camera's X, Y, Z axes in world coordinates)
        rotation_matrix = np.array([x_axis_world, y_axis_world, z_axis_world]).T

        # Construct camera_to_world matrix
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = rotation_matrix
        camera_to_world[:3, 3] = camera_position
        # --- End manual construction ---

        # Update camera node pose in the scene
        # pyrender uses world_to_camera matrix (inverse of camera_to_world)
        camera_node.matrix = np.linalg.inv(camera_to_world)

        # Render the scene
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES)

        img_path = os.path.join(output_dir, f'view_{i:04d}.png')
        Image.fromarray(color).save(img_path)
        saved_image_paths.append(img_path)
        camera_poses_camera_to_world.append(camera_to_world)

    renderer.delete()

    print(f"Generated {len(saved_image_paths)} multi-view images in '{output_dir}'.")
    return saved_image_paths, camera_poses_camera_to_world, camera_intrinsics

def create_full_synthetic_dataset(num_scenes=5, base_output_dir='synthetic_dataset'):
    """
    Generates a full synthetic dataset of 3D worlds with multimodal inputs and NeRF training data.

    Args:
        num_scenes (int): The number of unique 3D worlds to generate.
        base_output_dir (str): The base directory to save all generated scenes.
    """
    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir) # Clear previous data
    os.makedirs(base_output_dir)

    print(f"Generating {num_scenes} unique 3D worlds...")

    for i in range(num_scenes):
        scene_id = f'scene_{i:04d}'
        scene_output_dir = os.path.join(base_output_dir, scene_id)
        os.makedirs(scene_output_dir)

        print(f"\n--- Generating {scene_id} ---")

        # 2. Generate procedural 3D scene
        trimesh_scene, scene_description_dict = generate_procedural_scene(max_objects=random.randint(1, 5))

        # Save scene description dictionary
        with open(os.path.join(scene_output_dir, 'scene_description.json'), 'w') as f:
            json.dump(scene_description_dict, f, indent=4)

        # 3a. Generate Text Description
        text_desc = generate_text_description(scene_description_dict)
        with open(os.path.join(scene_output_dir, 'text_description.txt'), 'w') as f:
            f.write(text_desc)
        print("Text Description saved.")

        # 3b. Generate Reference Images
        ref_images_dir = os.path.join(scene_output_dir, 'reference_images')
        ref_image_paths, ref_cam_extrinsics, ref_cam_intrinsics = generate_reference_images(
            trimesh_scene, output_dir=ref_images_dir, num_cameras=3, resolution=(256, 256)
        )
        with open(os.path.join(ref_images_dir, 'camera_params.json'), 'w') as f:
            json.dump({
                'intrinsics': {k: float(v) for k, v in ref_cam_intrinsics.items()},
                'extrinsics': [ext.tolist() for ext in ref_cam_extrinsics]
            }, f, indent=4)
        print("Reference Images and Camera Parameters saved.")

        # 3c. Generate Multi-view Image Set for NeRFs
        nerf_data_dir = os.path.join(scene_output_dir, 'nerf_data')
        nerf_image_paths, nerf_cam_extrinsics, nerf_cam_intrinsics = generate_nerf_multiview_data(
            trimesh_scene, output_dir=nerf_data_dir, num_views=50, resolution=(256, 256)
        )
        with open(os.path.join(nerf_data_dir, 'camera_params.json'), 'w') as f:
            json.dump({
                'intrinsics': {k: float(v) for k, v in nerf_cam_intrinsics.items()},
                'extrinsics': [ext.tolist() for ext in nerf_cam_extrinsics]
            }, f, indent=4)
        print("NeRF Multi-view Images and Camera Parameters saved.")

    print(f"\nSuccessfully generated {num_scenes} synthetic scenes in '{base_output_dir}'.")

def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (np.ndarray): Ground truth image (H, W, C or H, W).
        img2 (np.ndarray): Rendered image (H, W, C or H, W).
        data_range (float): The dynamic range of the pixel values (e.g., 1.0 for [0, 1] images).

    Returns:
        float: The PSNR value.
    """
    # Ensure images are float type and within data_range
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1 = np.clip(img1, 0, data_range)
    img2 = np.clip(img2, 0, data_range)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        # MSE is zero means no noise, PSNR is infinite
        return float('inf')

    # PSNR formula
    psnr = 10 * log10((data_range ** 2) / mse)
    return psnr

def calculate_ssim(img1, img2, data_range=1.0, win_size=11, K1=0.01, K2=0.03):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (np.ndarray): Ground truth image (H, W, C or H, W).
        img2 (np.ndarray): Rendered image (H, W, C or H, W).
        data_range (float): The dynamic range of the pixel values (e.g., 1.0 for [0, 1] images).
        win_size (int): The size of the Gaussian window for filtering.
        K1 (float): Small constant to avoid division by zero (C1 = (K1 * data_range)**2).
        K2 (float): Small constant to avoid division by zero (C2 = (K2 * data_range)**2).

    Returns:
        float: The SSIM value.
    """
    # Ensure images are float type and within data_range
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1 = np.clip(img1, 0, data_range)
    img2 = np.clip(img2, 0, data_range)

    # Constants for SSIM
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Handle multi-channel images by iterating over channels or treating as a single channel
    if img1.ndim == 3:
        # For color images, compute SSIM for each channel and average
        ssim_channels = []
        for i in range(img1.shape[-1]):
            ssim_channels.append(calculate_ssim_single_channel(img1[..., i], img2[..., i], data_range, win_size, C1, C2))
        return np.mean(ssim_channels)
    else:
        return calculate_ssim_single_channel(img1, img2, data_range, win_size, C1, C2)

def calculate_ssim_single_channel(img1, img2, data_range, win_size, C1, C2):
    # Calculate local means, variances, and covariance using Gaussian filter
    mu1 = gaussian_filter(img1, sigma=1.5, mode='reflect', cval=0, truncate=3.5)
    mu2 = gaussian_filter(img2, sigma=1.5, mode='reflect', cval=0, truncate=3.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, sigma=1.5, mode='reflect', cval=0, truncate=3.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sigma=1.5, mode='reflect', cval=0, truncate=3.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5, mode='reflect', cval=0, truncate=3.5) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # Avoid division by zero by adding a small epsilon or handling NaNs
    ssim_map = numerator / (denominator + 1e-12)

    return np.mean(ssim_map)


def positional_encoding(coords, L):
    """
    Applies positional encoding to input coordinates.

    Args:
        coords (torch.Tensor): Input coordinates (e.g., 3D points or 3D direction vectors) of shape (N, D).
        L (int): The number of frequencies to use for encoding.

    Returns:
        torch.Tensor: The encoded coordinates of shape (N, D * 2 * L + D).
    """
    encoded_features = [coords]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            encoded_features.append(fn(2.0**i * coords))
    return torch.cat(encoded_features, dim=-1)
def get_rays(H, W, K, c2w):
    """
    Calculates ray origins and directions for all pixels in an image.

    Args:
        H (int): Image height.
        W (int): Image width.
        K (dict): Camera intrinsic parameters {'fx', 'fy', 'cx', 'cy'}.
        c2w (torch.Tensor): Camera-to-world extrinsic matrix (4x4).

    Returns:
        tuple: ray_origins (torch.Tensor) of shape (H*W, 3) and
               ray_directions (torch.Tensor) of shape (H*W, 3).
    """
    # Create a grid of pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i = i.transpose(1, 0).to(c2w.device) # (H, W)
    j = j.transpose(1, 0).to(c2w.device) # (H, W)

    # Normalize pixel coordinates to camera space
    # x = (u - cx) / fx, y = (v - cy) / fy
    fx = K['fx'].to(c2w.device)
    fy = K['fy'].to(c2w.device)
    cx = K['cx'].to(c2w.device)
    cy = K['cy'].to(c2w.device)

    # Direction vectors in camera coordinates (homogeneous)
    # Pinhole camera model assumes principal point is (cx, cy) and focal lengths fx, fy
    # d_cam = [(i - cx) / fx, (j - cy) / fy, 1]
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # Original NeRF convention often uses -z forward

    # Transform ray directions from camera space to world space
    # Directions are rotated, not translated
    ray_directions = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1) # (H, W, 3)

    # Ray origins are all the same: camera position in world space
    ray_origins = c2w[:3, -1].expand(dirs.shape)

    return ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)

def render_rays(nerf_mlp, generated_nerf_latent, ray_origins, ray_directions, N_samples, L_pos, L_dir):
    """
    Performs volume rendering for a batch of rays.

    Args:
        nerf_mlp (nn.Module): The NeRF MLP to query for density and color.
        generated_nerf_latent (torch.Tensor): The latent code representing the 3D scene (batch_size, latent_dim_neRF).
        ray_origins (torch.Tensor): Origins of the rays (num_rays, 3).
        ray_directions (torch.Tensor): Directions of the rays (num_rays, 3).
        N_samples (int): Number of samples to take along each ray.
        L_pos (int): Number of frequencies for positional encoding of points.
        L_dir (int): Number of frequencies for positional encoding of directions.

    Returns:
        torch.Tensor: Rendered image (num_rays, 3).
        torch.Tensor: Depth map (num_rays, 1).
    """
    num_rays = ray_origins.shape[0]
    # Ensure generated_nerf_latent is broadcastable to all sample points
    # It needs to be (num_rays * N_samples, latent_dim_neRF)
    latent_expanded = generated_nerf_latent.unsqueeze(1).repeat(1, N_samples, 1)
    latent_expanded = latent_expanded.view(-1, generated_nerf_latent.shape[-1])

    # Sample points along each ray
    t_vals = torch.linspace(0., 1., N_samples, device=ray_origins.device)
    # Introduce a depth range if needed, for simplicity we'll use a fixed range or scale t_vals
    # For a unit sphere scene, t_vals can be scaled by a max_depth
    max_depth = 5.0 # Example max depth
    z_vals = t_vals * max_depth # Scale t_vals to scene depth
    z_vals = z_vals.expand([num_rays, N_samples])

    # Add noise to samples for anti-aliasing
    mid_points = (z_vals[..., 1:] + z_vals[..., :-1]) / 2.
    upper = torch.cat([mid_points, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mid_points], -1)
    perturb_idx = torch.rand(z_vals.shape, device=ray_origins.device) # Random uniformly [0, 1)
    z_vals = lower + (upper - lower) * perturb_idx

    # Calculate sample points in 3D space
    # r(t) = o + t * d
    pts = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
    pts_flat = pts.reshape(-1, 3)

    # Get normalized ray directions for each sample point (repeated)
    dirs_flat = ray_directions[..., None, :].expand(-1, N_samples, -1).reshape(-1, 3)

    # Positional encoding for points and directions
    positions_encoded = positional_encoding(pts_flat, L_pos)
    directions_encoded = positional_encoding(dirs_flat, L_dir)

    # Query NeRF MLP for density and color
    # The NeRF MLP expects (num_rays * N_samples, encoded_dim)
    density, colors = nerf_mlp(latent_expanded, positions_encoded, directions_encoded)

    density = density.view(num_rays, N_samples)
    colors = colors.view(num_rays, N_samples, 3)

    # Volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.tensor([1e10], device=ray_origins.device).expand(z_vals[..., :1].shape)], -1)
    alpha = 1.0 - torch.exp(-F.relu(density) * dists) # alpha = 1 - exp(-sigma * delta_t)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((num_rays, 1), device=ray_origins.device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]

    rendered_image = (weights[..., None] * colors).sum(dim=-2) # Sum over samples
    depth_map = (weights * z_vals).sum(dim=-1).unsqueeze(-1)

    return rendered_image, depth_map
