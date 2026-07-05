import numpy as np

class EgoVehicleState:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0,
                 vx=0.0, vy=0.0, vz=0.0, ax=0.0, ay=0.0, az=0.0,
                 steering_angle=0.0, throttle=0.0, brake=0.0):
      self.x=x
      self.y=y
      self.z=z
      self.yaw=yaw
      self.pitch=pitch
      self.roll=roll
      self.vx = vx  # Velocity in X (m/s)
      self.vy = vy  # Velocity in Y (m/s)
      self.vz = vz  # Velocity in Z (m/s)
      self.ax = ax  # Acceleration in X (m/s^2)
      self.ay = ay  # Acceleration in Y (m/s^2)
      self.az = az  # Acceleration in Z (m/s^2)
      self.steering_angle=steering_angle
      self.throttle=throttle
      self.brake=brake
    def __repr__(self):
        return (f"EgoVehicleState(pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), "
                f"yaw={np.degrees(self.yaw):.2f} deg, vel=({self.vx:.2f}, {self.vy:.2f}), "
                f"ctrl=(steer={np.degrees(self.steering_angle):.2f}, throt={self.throttle:.2f}, brake={self.brake:.2f}))")
    
class DynamicAgent:
    def __init__(self, agent_id, agent_type, x, y, z, yaw, vx, vy, vz,
                 length=4.5, width=1.8, height=1.5, intention=None):
        self.agent_id = agent_id
        self.agent_type = agent_type  # e.g., 'vehicle', 'pedestrian', 'cyclist'
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw # Heading (radians)
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.length = length
        self.width = width
        self.height = height
        self.intention = intention # e.g., 'straight', 'turn_left', 'stop'
        self.history = [] # To store past states for prediction GT
    def get_corners_2d(self):
      half_l=self.length/2
      half_w=self.width/2
      corners=np.array([
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w]
      ])
      rotation_matrix=np.array([
          [np.cos(self.yaw),-np.sin(self.yaw)],
          [np.sin(self.yaw),np.cos(self.yaw)]
      ])
      rotated_corners=corners @ rotation_matrix.T
      return rotated_corners+np.array([self.x,self.y])
    def __repr__(self):
        return (f"DynamicAgent(id='{self.agent_id}', type='{self.agent_type}', "
                f"pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), "
                f"yaw={np.degrees(self.yaw):.2f} deg, vel=({self.vx:.2f}, {self.vy:.2f}), "
                f"intent='{self.intention}')")
                
class Lane:
  def __init__(self,lane_id,waypoints,lane_type='driving',speed_limit=None):
    self.lane_id=lane_id
    self.waypoints=waypoints
    self.lane_type=lane_type
    self.speed_limit=speed_limit
  def __repr__(self):
    return (f"Lane(id='{self.lane_id}', type='{self.lane_type}', "
                f"waypoints={len(self.waypoints)} points, speed_limit={self.speed_limit} m/s)")

class TrafficLight:
  def __init__(self,light_id,x,y,z,orientation_yaw,state='red',junction_id=None):
        self.light_id = light_id
        self.x = x
        self.y = y
        self.z = z
        self.orientation_yaw = orientation_yaw # Direction the light faces (radians)
        self.state = state                     # e.g., 'red', 'yellow', 'green', 'off'
        self.junction_id = junction_id         # ID of the intersection it belongs to
  def __repr__(self):
        return (f"TrafficLight(id='{self.light_id}', pos=({self.x:.2f}, {self.y:.2f}), "
                f"state='{self.state}', orient={np.degrees(self.orientation_yaw):.2f} deg)")

class RoadMap:
  def __init__(self):
    self.lanes={}
    self.traffic_lights={}
    self.traffic_signs={}
  def add_lane(self,lane:Lane):
    self.lanes[lane.lane_id]=lane
  def add_traffic_light(self,traffic_light:TrafficLight):
    self.traffic_lights[traffic_light.light_id]=traffic_light
  # def add_traffic_signs(self,sign:TrafficSign):
  #   self.traffic_signs[sign:sign_id]=sign
  def __repr__(self):
      return (f"RoadMap(lanes={len(self.lanes)}, "
                f"traffic_lights={len(self.traffic_lights)}, "
                f"traffic_signs={len(self.traffic_signs)})")

class CameraSensor:
  def __init__(self,sensor_id,mount_x,mount_y,mount_z,mount_yaw,fov_h,fov_v,image_width,image_height):
        self.sensor_id = sensor_id
        self.mount_x = mount_x # Position relative to ego-vehicle (m)
        self.mount_y = mount_y
        self.mount_z = mount_z
        self.mount_yaw = mount_yaw # Orientation relative to ego-vehicle (radians)
        self.fov_h = fov_h       # Horizontal Field of View (radians)
        self.fov_v = fov_v       # Vertical Field of View (radians)
        self.image_width = image_width
        self.image_height = image_height
  def  get_relative_tranform(self):
    rot_z=np.array(
            [[np.cos(self.mount_yaw), -np.sin(self.mount_yaw), self.mount_x],
            [np.sin(self.mount_yaw),  np.cos(self.mount_yaw), self.mount_y],
            [0, 0, 1]]
    )
    return rot_z
  def simulate(self,ego_state:EgoVehicleState,agents:list[DynamicAgent],roadmap:RoadMap):
    image=np.zeros((self.image_height,self.image_width,3),dtype=np.uint8)
    image[self.image_height-20:self.image_height-5,self.image_width//2-10:self.image_width//2+10]=[0,0,255]
    for agent in agents:
      dx=agent.x-ego_state.x
      dy=agent.y-ego_state.y
      cos_yaw=np.cos(-ego_state.yaw)
      sin_yaw=np.sin(-ego_state.yaw)
      rel_x=dx*cos_yaw-dy*sin_yaw
      rel_y=dx*sin_yaw+dy*cos_yaw
      if rel_x> 0 and abs(rel_y)<10:
        img_x=int(self.image_width//2+rel_y*(self.image_width/20))
        img_y=int(self.image_height-(rel_x*(self.image_height/30)))
        img_x=np.clip(img_x,0,self.image_width-1)
        img_y=np.clip(img_y,0,self.image_height-1)
        color=[255,0,0] if agent.agent_type=="vehicle"else [0,255,0]
        size=5
        y1=max(0,img_y-size)
        y2=min(self.image_height,img_y+size)
        x1=max(0,img_x-size)
        x3=min(self.image_width,img_x+size)
        image[y1:y2,x1:x3]=color
    return image
  def __repr__(self):
        return (f"CameraSensor(id='{self.sensor_id}', mount_pos=({self.mount_x:.2f}, {self.mount_y:.2f}, {self.mount_z:.2f}), "
                f"FOV={np.degrees(self.fov_h):.1f}x{np.degrees(self.fov_v):.1f} deg, resolution={self.image_width}x{self.image_height})")
  
class LiDARSensor:
  def __init__(self,sensor_id,mount_x,mount_y,mount_z,mount_yaw,max_range,num_points):
        self.sensor_id = sensor_id
        self.mount_x = mount_x # Position relative to ego-vehicle (m)
        self.mount_y = mount_y
        self.mount_z = mount_z
        self.mount_yaw = mount_yaw # Orientation relative to ego-vehicle (radians)
        self.max_range = max_range
        self.num_points = num_points
  def simulate(self,ego_state:EgoVehicleState,agents:list[DynamicAgent],roadmap:RoadMap):
    points=[]
    for agent in agents:
      dx=agent.x-ego_state.x
      dy=agent.y-ego_state.y
      dz=agent.z-ego_state.z # Fixed: agtent.z -> agent.z
      cos_yaw_ego=np.cos(-ego_state.yaw)
      sin_yaw_ego=np.sin(-ego_state.yaw) # Fixed: sin_yaw_eog -> sin_yaw_ego
      agent_rel_x=dx*cos_yaw_ego-dy*sin_yaw_ego # Fixed: sin_yaw -> sin_yaw_ego
      agent_rel_y=dx*sin_yaw_ego+dy*cos_yaw_ego # Fixed: sin_yaw_eog -> sin_yaw_ego
      agent_rel_z=dz
      for _ in range(self.num_points//len(agents)):
        px=agent_rel_x+(np.random.rand()-0.5)*agent.length
        py=agent_rel_y+(np.random.rand()-0.5)*agent.width
        pz=agent_rel_z+(np.random.rand()-0.5)*agent.height
        if np.sqrt(px**2+py**2)<self.max_range:
          points.append([px,py,pz,1.0])
    for _ in range(self.num_points//5):
      px=np.random.uniform(0,self.max_range)
      py=np.random.uniform(-self.max_range/2,self.max_range/2)
      pz=-0.5
      points.append([px,py,pz,0.5])
    return np.array(points)
  def __repr__(self):
        return (f"LiDARSensor(id='{self.sensor_id}', mount_pos=({self.mount_x:.2f}, {self.mount_y:.2f}, {self.mount_z:.2f}), "
                f"max_range={self.max_range} m, num_points={self.num_points})")
  
class RadarSensor:
    """Simulates a Radar sensor, generating conceptual detections for dynamic objects."""
    def __init__(self, sensor_id, mount_x, mount_y, mount_z, mount_yaw, max_range, fov_h):
        self.sensor_id = sensor_id
        self.mount_x = mount_x # Position relative to ego-vehicle (m)
        self.mount_y = mount_y
        self.mount_z = mount_z
        self.mount_yaw = mount_yaw # Orientation relative to ego-vehicle (radians)
        self.max_range = max_range
        self.fov_h = fov_h       # Horizontal Field of View (radians)

    def simulate(self, ego_state: EgoVehicleState, agents: list[DynamicAgent]):
        """Generates conceptual radar detections.

        Each detection includes range, radial velocity, and angle for detected agents.
        """
        detections = []
        for agent in agents:
            # Calculate relative position and velocity (simplified for 2D)
            dx = agent.x - ego_state.x
            dy = agent.y - ego_state.y
            dvx = agent.vx - ego_state.vx
            dvy = agent.vy - ego_state.vy

            distance = np.sqrt(dx**2 + dy**2)

            # Check if agent is within max_range
            if distance < self.max_range and distance > 0:
                # Relative bearing from ego-vehicle's front
                angle_from_ego = np.arctan2(dy, dx) - ego_state.yaw
                # Normalize angle to [-pi, pi]
                angle_from_ego = (angle_from_ego + np.pi) % (2 * np.pi) - np.pi

                # Check if within FoV
                if abs(angle_from_ego) < self.fov_h / 2:
                    # Calculate radial velocity
                    radial_vel = (dx * dvx + dy * dvy) / distance

                    detections.append({
                        'agent_id': agent.agent_id,
                        'range': distance,
                        'radial_velocity': radial_vel,
                        'azimuth_angle': angle_from_ego # Angle relative to ego-forward
                    })
        return detections

    def __repr__(self):
        return (f"RadarSensor(id='{self.sensor_id}', mount_pos=({self.mount_x:.2f}, {self.mount_y:.2f}, {self.mount_z:.2f}), "
                f"max_range={self.max_range} m, FOV={np.degrees(self.fov_h):.1f} deg)")

class GroundTruthGenerator:
    """Generates ground truth labels for perception, prediction, and planning."""
    def __init__(self, simulation_horizon_seconds=5.0, prediction_steps=20, dt=0.25, pose_dim=3):
        self.simulation_horizon_seconds = simulation_horizon_seconds
        self.prediction_steps = prediction_steps
        self.dt = dt # Time step for prediction trajectories
        self.pose_dim = pose_dim # Add pose_dim attribute here

    def generate_perception_gt(self, ego_state: EgoVehicleState, agents: list[DynamicAgent], roadmap: RoadMap):
        """Generates ground truth for perception tasks.

        Includes 3D bounding boxes, semantic segmentation (conceptual), and lane info.
        """
        perception_gt = {
            '3d_boxes': [],
            'semantic_map': None, # Conceptual BEV semantic map
            'lane_boundaries': [],
            'traffic_lights_state': []
        }

        # 3D Bounding Boxes for Agents
        for agent in agents:
            perception_gt['3d_boxes'].append({
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type,
                'x': agent.x, 'y': agent.y, 'z': agent.z,
                'yaw': agent.yaw,
                'length': agent.length, 'width': agent.width, 'height': agent.height,
                'vx': agent.vx, 'vy': agent.vy, 'vz': agent.vz
            })

        # Conceptual Semantic Map (Bird's Eye View Grid)
        # For simplicity, let's create a small grid and mark agent/lane presence.
        grid_size = 50 # 50x50 meter grid
        resolution = 0.5 # 0.5 meters per pixel
        bev_width = int(grid_size / resolution)
        bev_height = int(grid_size / resolution)
        # Change semantic_bev to single channel for class IDs
        semantic_bev = np.zeros((bev_height, bev_width), dtype=np.int64) # Use int64 for class indices

        # Map ego-centric coordinates to BEV grid indices
        def world_to_bev(world_x, world_y, ego_x, ego_y, ego_yaw):
            # Translate relative to ego-vehicle
            rel_x = world_x - ego_x
            rel_y = world_y - ego_y

            # Rotate into ego-vehicle's frame
            cos_yaw = np.cos(-ego_yaw)
            sin_yaw = np.sin(-ego_yaw)
            rot_x = rel_x * cos_yaw - rel_y * sin_yaw
            rot_y = rel_x * sin_yaw + rel_y * cos_yaw

            # Map to BEV grid (center of grid is ego-vehicle)
            # NOTE: For BEV visualization and semantic map, typically X in world is up/down in BEV, Y is left/right
            # So world_y correlates to bev_x, world_x to bev_y.
            grid_x = int(bev_width // 2 - rot_y / resolution) # Y in world -> X in BEV (column index)
            grid_y = int(bev_height // 2 - rot_x / resolution) # X in world -> Y in BEV (row index)
            return grid_x, grid_y

        # Draw lanes (Class ID: 1)
        for lane_id, lane in roadmap.lanes.items():
            lane_pts = []
            for wp in lane.waypoints:
                # Consider only x, y for 2D BEV map
                grid_x, grid_y = world_to_bev(wp[0], wp[1], ego_state.x, ego_state.y, ego_state.yaw)
                lane_pts.append([grid_x, grid_y])
            if len(lane_pts) > 1:
                # Draw line (simplified - fill points along line)
                for i in range(len(lane_pts) - 1):
                    x0, y0 = lane_pts[i]
                    x1, y1 = lane_pts[i+1]
                    num_segments = max(abs(x1 - x0), abs(y1 - y0)) + 1
                    for j in range(int(num_segments)):
                        px = int(x0 + (x1 - x0) * j / num_segments)
                        py = int(y0 + (y1 - y0) * j / num_segments)
                        if 0 <= px < bev_width and 0 <= py < bev_height:
                            semantic_bev[py, px] = 1 # Class ID for lanes
            perception_gt['lane_boundaries'].append({
                'lane_id': lane_id,
                'waypoints': lane.waypoints # Store original world coordinates
            })

        # Draw agents on BEV (Class IDs: 2 for vehicle, 3 for pedestrian)
        for agent in agents:
            corners = agent.get_corners_2d() # Get world corners
            bev_corners = []
            for wc_x, wc_y in corners:
                gc_x, gc_y = world_to_bev(wc_x, wc_y, ego_state.x, ego_state.y, ego_state.yaw)
                bev_corners.append([gc_x, gc_y])

            # Simplified: draw a rectangle for agent on BEV
            min_x = int(np.min([c[0] for c in bev_corners]))
            max_x = int(np.max([c[0] for c in bev_corners]))
            min_y = int(np.min([c[1] for c in bev_corners]))
            max_y = int(np.max([c[1] for c in bev_corners]))

            min_x = np.clip(min_x, 0, bev_width - 1)
            max_x = np.clip(max_x, 0, bev_width - 1)
            min_y = np.clip(min_y, 0, bev_height - 1)
            max_y = np.clip(max_y, 0, bev_height - 1)

            class_id = 0 # Default background
            if agent.agent_type == 'vehicle': class_id = 2 # Vehicle class ID
            elif agent.agent_type == 'pedestrian': class_id = 3 # Pedestrian class ID
            # Add other agent types as needed

            if min_x < max_x and min_y < max_y: # Ensure valid rectangle
                semantic_bev[min_y:max_y, min_x:max_x] = class_id

        # Draw traffic lights (Class ID: 4 for green, 5 for red, etc. - simplified for one ID for now)
        for light_id, light in roadmap.traffic_lights.items():
            # Simplified: just mark its position conceptually
            grid_x, grid_y = world_to_bev(light.x, light.y, ego_state.x, ego_state.y, ego_state.yaw)
            if 0 <= grid_x < bev_width and 0 <= grid_y < bev_height:
                semantic_bev[grid_y, grid_x] = 4 # Conceptual Traffic Light class ID
            perception_gt['traffic_lights_state'].append({
                'light_id': light_id,
                'state': light.state,
                'position': [light.x, light.y, light.z]
            })

        perception_gt['semantic_map'] = semantic_bev

        return perception_gt

    def generate_prediction_gt(self, ego_state: EgoVehicleState, agents: list[DynamicAgent]):
        """Generates ground truth future trajectories for dynamic agents.

        This assumes a simple motion model for agents for the prediction horizon.
        """
        prediction_gt = {
            'future_trajectories': [],
            'agent_intentions': []
        }

        for agent in agents:
            future_trajectory = []
            current_x, current_y, current_yaw = agent.x, agent.y, agent.yaw
            current_vx, current_vy = agent.vx, agent.vy

            for i in range(self.prediction_steps):
                # Simple constant velocity model for prediction GT
                next_x = current_x + current_vx * self.dt
                next_y = current_y + current_vy * self.dt

                # For more complex scenarios, this would involve behavior models
                # or pre-defined paths from the scenario generator.
                future_trajectory.append([next_x, next_y, current_yaw]) # Assuming constant yaw for simplicity

                current_x, current_y = next_x, next_y

            prediction_gt['future_trajectories'].append({
                'agent_id': agent.agent_id,
                'trajectory': future_trajectory # List of [x, y, yaw] for each step
            })
            prediction_gt['agent_intentions'].append({
                'agent_id': agent.agent_id,
                'intention': agent.intention
            })

        return prediction_gt

    def generate_planning_gt(self, ego_state: EgoVehicleState, roadmap: RoadMap, current_route_waypoints=None):
        """Generates ground truth optimal trajectory/actions for the ego-vehicle.

        In a real simulator, this would come from an expert planner or pre-defined optimal paths.
        For this conceptual sim, we generate a simple 'follow lane' trajectory.
        """
        planning_gt = {
            'optimal_trajectory': [],
            'optimal_controls': []
        }

        # Simple 'follow current lane' logic
        if roadmap.lanes:
            # Find the closest lane waypoint ahead of the ego-vehicle
            closest_waypoint_idx = -1
            min_dist_to_lane = float('inf')

            target_lane = list(roadmap.lanes.values())[0] # Just take the first lane for simplicity

            for i, wp in enumerate(target_lane.waypoints):
                dist = np.sqrt((wp[0] - ego_state.x)**2 + (wp[1] - ego_state.y)**2)
                if dist < min_dist_to_lane: # Only consider waypoints ahead of ego
                    # Check if waypoint is roughly ahead of ego (simplified to x-coordinate for now)
                    if (wp[0] > ego_state.x and ego_state.yaw < np.pi/4) or \
                       (wp[0] < ego_state.x and ego_state.yaw > 3*np.pi/4) or \
                       (wp[1] > ego_state.y and ego_state.yaw > np.pi/4 and ego_state.yaw < 3*np.pi/4) or \
                       (wp[1] < ego_state.y and ego_state.yaw > -3*np.pi/4 and ego_state.yaw < -np.pi/4):
                        min_dist_to_lane = dist
                        closest_waypoint_idx = i

            if closest_waypoint_idx != -1:
                # Generate a trajectory by following the lane waypoints
                for i in range(closest_waypoint_idx, min(len(target_lane.waypoints), closest_waypoint_idx + self.prediction_steps)):
                    wp = target_lane.waypoints[i]
                    planning_gt['optimal_trajectory'].append([wp[0], wp[1], ego_state.yaw]) # Assume constant yaw towards waypoint
                    planning_gt['optimal_controls'].append({
                        'steering_angle': 0.0, # Placeholder
                        'throttle': 0.5,     # Placeholder
                        'brake': 0.0         # Placeholder
                    }) # Conceptual controls
            else:
                # If no lane found ahead, try to stay straight
                for i in range(self.prediction_steps):
                    planning_gt['optimal_trajectory'].append([ego_state.x + ego_state.vx * i * self.dt,
                                                           ego_state.y + ego_state.vy * i * self.dt,
                                                           ego_state.yaw])
                    planning_gt['optimal_controls'].append({'steering_angle': 0.0, 'throttle': 0.5, 'brake': 0.0})

        return planning_gt