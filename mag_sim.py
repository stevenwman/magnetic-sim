import mujoco
import mujoco.viewer
import time 
import numpy as np

file = "scene.xml"

model = mujoco.MjModel.from_xml_path(file)
data = mujoco.MjData(model)

t_max = 100.0  # Maximum simulation time in seconds
start_time = time.time()

MU_0 = 4 * np.pi * 1e-7

def calculate_magnetic_force(
    distance: float,
    magnet_remanence: float,
    magnet_volume: float
) -> float:
    """
    Calculates the attractive force between a permanent magnet and a large
    ferromagnetic plate using the method of images and a dipole approximation.

    This model is a good approximation when the distance is not extremely small
    compared to the magnet's dimensions.

    Args:
        distance (float): The shortest distance from the face of the magnet
                          to the ferromagnetic surface (in meters). Must be > 0.
        magnet_remanence (float): The remanence (or residual flux density, Br)
                                  of the permanent magnet material (in Tesla).
                                  A typical N42 neodymium magnet is ~1.3 T.
        magnet_volume (float): The volume of the magnet (in cubic meters).

    Returns:
        float: The calculated attractive force (in Newtons).
    """
    if distance <= 0:
        # Return a very large force for contact, or handle as you see fit.
        # Returning 0 prevents crashes but isn't physically perfect.
        return 0

    # 1. Calculate the magnetic dipole moment (m) of the magnet.
    # m = (B_r * V) / μ₀
    magnetic_moment = (magnet_remanence * magnet_volume) / MU_0

    # 2. Apply the method of images.
    # The distance between the centers of the real and image dipoles is 2 * distance.
    z = 2 * distance

    # 3. Calculate the force between two aligned magnetic dipoles.
    # The force between two aligned dipoles is F = (3 * μ₀ * m₁ * m₂) / (2 * π * z⁴).
    force = (3 * MU_0 * magnetic_moment**2) / (2 * np.pi * z**4)

    return force

def add_visual_arrow(scene, from_point, to_point, radius=0.01, rgba=(0, 0, 1, 1)):
    """
    Adds a single visual arrow to the mjvScene.
    This is a visual-only object and does not affect the physics.

    Args:
        scene (mjvScene): The scene to add the arrow to.
        from_point (np.ndarray): The starting point of the arrow.
        to_point (np.ndarray): The ending point of the arrow.
        radius (float): The radius of the arrow's shaft.
        rgba (tuple): The color and alpha of the arrow.
    """
    if scene.ngeom >= scene.maxgeom:
        print("Warning: Maximum number of geoms reached. Cannot add arrow.")
        return

    # Get a reference to the next available geom in the scene
    geom = scene.geoms[scene.ngeom]
    
    # Set the properties of the arrow
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(), # Will be updated by mjv_connector
        rgba=np.array(rgba, dtype=np.float32)
    )
    
    # Use MuJoCo's built-in function to correctly position and orient the arrow.
    # This version uses mjv_connector and passes the full numpy arrays.
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        from_point,
        to_point
    )

    # Increment the number of geoms in the scene
    scene.ngeom += 1
    print(scene.ngeom, "geoms in scene")


with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set the simulation timestep
    model.opt.timestep = 0.01  # 100 Hz simulation frequency

    # Get the simulation timestep
    timestep = model.opt.timestep

    # Main simulation loop
    while viewer.is_running():
        mag_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mag_center")
        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "metal_box")
        mag_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "magnet_body")

        fromto = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        distance = mujoco.mj_geomDistance(model, data, mag_id, box_id, 0.5, fromto)
        vec = fromto[3:6] - fromto[0:3]

        print(f"dx: {distance:.2f}, fromto: {np.round(fromto,2)}")
        viewer.user_scn.ngeom = 0
        add_visual_arrow(viewer.user_scn, fromto[0:3], fromto[3:6]+vec, rgba=(0, 0, 1, 1))

        if distance > 1e-5: # Add a small threshold to prevent division by zero
            # 1. Define magnet properties (you would get these from your model or constants)
            magnet_remanence = 1.3  # Example: N42 grade neodymium
            magnet_volume = (0.02**3) # Example: 2cm cube magnet

            # 2. Calculate the force magnitude (using a function like we made before)
            # This is a placeholder for your actual force function
            force_magnitude = calculate_magnetic_force(distance, magnet_remanence, magnet_volume)

            # 3. Calculate the normalized direction vector for attraction
            from_point = fromto[0:3]
            to_point = fromto[3:6]
            direction_vector = to_point - from_point
            normalized_direction = direction_vector / np.linalg.norm(direction_vector)

            # 4. Calculate the final 3D force vector
            force_vector = force_magnitude * normalized_direction

            # 5. Apply the force to the magnet's body
            # data.xfrc_applied is a [nbody x 6] array (force, torque)
            # We add our force to the correct body's linear force component.
            data.xfrc_applied[mag_body_id, 0:3] += force_vector

        step_start_time = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = timestep - (time.time() - step_start_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if time.time() - start_time >= t_max:
            print("Simulation time exceeded maximum limit. Exiting.")
            break