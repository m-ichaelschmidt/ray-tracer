# CSC 305 - Fall 2023 - Assignment 3 - Ray Tracing
# Michael Schmidt
# V00967578
# December 4, 2023

import sys
import numpy as np

def parse_input(file_lines):
    scene_data = {
        "near": None,
        "left": None,
        "right": None,
        "top": None,
        "bottom": None,
        "resolution": None,
        "spheres": [],
        "lights": [],
        "background_colour": None,
        "ambient_intensity": None,
        "output_file_name": None
    }

    for line in file_lines:
        parts = line.strip().split()
        if len(parts) > 0:
            keyword = parts[0].upper()

            if keyword in ["NEAR", "LEFT", "RIGHT", "TOP", "BOTTOM"]:
                scene_data[keyword.lower()] = float(parts[1])
            elif keyword == "RES":
                scene_data["resolution"] = (int(parts[1]), int(parts[2]))
            elif keyword == "SPHERE":
                # getting sphere data
                name = parts[1]
                position = tuple(map(float, parts[2:5]))
                scaling = tuple(map(float, parts[5:8]))
                color = tuple(map(float, parts[8:11]))
                ka, kd, ks, kr = map(float, parts[11:15])
                n = int(parts[15])

                inv_scaling_matrix = np.diag([1/s if s != 0 else 1 for s in scaling] + [1])
                translation_matrix = np.identity(4)
                translation_matrix[:3, 3] = -np.array(position)

                # inverse transformation matrix
                inv_transform = np.dot(inv_scaling_matrix, translation_matrix)

                sphere = {
                    "name": name,
                    "position": position,
                    "scaling": scaling,
                    "color": color,
                    "ka": ka,
                    "kd": kd,
                    "ks": ks,
                    "kr": kr,
                    "n": n,
                    "trans_matrix" : inv_transform,
                    "inv_scaling_matrix" : inv_scaling_matrix[:3, :3]
                }
                scene_data["spheres"].append(sphere)

            elif keyword == "LIGHT":
                # getting light data
                name = parts[1]
                position = tuple(map(float, parts[2:5]))
                intensity = tuple(map(float, parts[5:8]))
                light = {
                    "name": name,
                    "position": position,
                    "intensity": intensity
                }
                scene_data["lights"].append(light)
            elif keyword == "BACK":
                scene_data["background_colour"] = tuple(map(float, parts[1:]))
            elif keyword == "AMBIENT":
                scene_data["ambient_intensity"] = tuple(map(float, parts[1:]))
            elif keyword == "OUTPUT":
                # Ensuring output file name is not more than 20 characters
                scene_data["output_file_name"] = parts[1][:20]

    return scene_data

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: # avoid division by 0
        return vector
    return vector / norm

def ellipsoid_intersect(inv_transform, ray_origin, ray_direction):
    norm_ray_dir = normalize(ray_direction)

    # transform ray into object space
    transformed_ray_origin = np.dot(inv_transform, np.append(ray_origin, 1))[:3]
    transformed_ray_direction = np.dot(inv_transform, np.append(norm_ray_dir, 0))[:3]

    A = np.dot(transformed_ray_direction, transformed_ray_direction)
    B = 2 * np.dot(transformed_ray_direction, transformed_ray_origin)
    C = np.dot(transformed_ray_origin, transformed_ray_origin) - 1

    # solve quadratic equation for t
    delta = B**2 - 4 * A * C
    if delta < 0: # no intersection
        return None  
    sqrt_delta = np.sqrt(delta)
    t1 = (-B + sqrt_delta) / (2 * A)
    t2 = (-B - sqrt_delta) / (2 * A)

    # near plane clipping
    valid_t1 = t1 > 0 and (ray_origin + t1 * norm_ray_dir)[2] < -1
    valid_t2 = t2 > 0 and (ray_origin + t2 * norm_ray_dir)[2] < -1

    if valid_t1 and valid_t2:
        return min(t1, t2)
    elif valid_t1:
        return t1
    elif valid_t2:
        return t2

    return None

# find nearest object(sphere) ray intersects
def nearest_intersected_object(spheres, ray_origin, ray_direction):
    distances = [ellipsoid_intersect(sphr["trans_matrix"], ray_origin, ray_direction) for sphr in spheres]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = spheres[index]
    return nearest_object, min_distance

def calculate_normal_at_intersection(sphere, intersection_point):
    translated_intersection = intersection_point - np.array(sphere["position"])

    inv_scaling_matrix = sphere["inv_scaling_matrix"]

    # apply inverse scaling
    unscaled_normal = normalize(np.dot(inv_scaling_matrix, translated_intersection))

    # get normal back to the world space
    transformed_normal = normalize(np.dot(inv_scaling_matrix.T, unscaled_normal))

    return transformed_normal

def trace_ray(ray_origin, ray_direction, spheres, lights, ambient_intensity, background_colour, camera, depth=0):
    if depth > 3:
        return np.array([0, 0, 0])

    nearest_object, min_distance = nearest_intersected_object(spheres, ray_origin, ray_direction)
    if nearest_object is None and depth > 0:
        return np.array([0, 0, 0])
    elif nearest_object is None and depth == 0:
        return np.clip(background_colour, 0, 1)

    intersection = ray_origin + min_distance * ray_direction
    normal_to_surface = calculate_normal_at_intersection(nearest_object, intersection)
    offset_intersection = intersection + 1e-7 * normal_to_surface

    # ambient
    illumination = np.array(nearest_object["color"]) * np.array(ambient_intensity) * nearest_object["ka"]

    # indentifying spheres bisected by near plane
    bisected_by_near_plane = (nearest_object["position"][2] - nearest_object["scaling"][2] < -1 and nearest_object["position"][2] + nearest_object["scaling"][2] > -1)

    for light in lights:
        intersection_to_light = normalize(light["position"] - offset_intersection)
        light_in_sphere = False

        # check if light is coming from inside sphere
        if np.dot(intersection_to_light, normal_to_surface) < 0:
            light_in_sphere = True
            normal_to_surface = -normal_to_surface  # invert normal
            offset_intersection = intersection + 1e-7 * normal_to_surface # recalculate intersection with new normal

        _, min_distance = nearest_intersected_object(spheres, offset_intersection, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light["position"] - offset_intersection)
        is_shadowed = min_distance < intersection_to_light_distance

        if bisected_by_near_plane and not light_in_sphere:
            is_shadowed = True

        if is_shadowed:
            if light_in_sphere:
                normal_to_surface = -normal_to_surface
                offset_intersection = intersection + 1e-7 * normal_to_surface
            continue

        # diffuse
        illumination += nearest_object["kd"] * np.array(light["intensity"]) * np.dot(intersection_to_light, normal_to_surface) * np.array(nearest_object["color"])

        # specular - Phong 
        if nearest_object["ks"] > 0:
            view_direction = normalize(camera - intersection)
            light_reflect_direction = reflect(-intersection_to_light, normal_to_surface)
            R_dot_V = max(0, np.dot(light_reflect_direction, view_direction))
            specular_intensity = nearest_object["ks"] * np.array(light["intensity"]) * (R_dot_V ** nearest_object["n"])
            illumination += specular_intensity

        # reset normal if it was inverted
        if light_in_sphere:
            light_in_sphere = False
            normal_to_surface = -normal_to_surface
            offset_intersection = intersection + 1e-7 * normal_to_surface

    # reflections
    if nearest_object["kr"] > 0:
        reflected_ray_direction = reflect(ray_direction, normal_to_surface)
        reflected_color = trace_ray(offset_intersection, reflected_ray_direction, spheres, lights, ambient_intensity, background_colour, camera, depth + 1)
        illumination += nearest_object["kr"] * reflected_color

    return np.clip(illumination, 0, 1)

def reflect(direction, normal):
    return direction - 2 * np.dot(direction, normal) * normal

def main():
    # ensure correct usage
    if len(sys.argv) < 2:
        print("Usage: python RayTracer.py <file_name>")
        sys.exit(1)

    file_name = sys.argv[1]

    with open(file_name, 'r') as file:
        file_contents = file.readlines()
    
    # set up scene
    scene = parse_input(file_contents)
    width = scene["resolution"][0]
    height = scene["resolution"][1]
    camera = np.array([0, 0, 0])
    ratio = float(width) / height
    screen = (scene["left"], scene["top"] / ratio, scene["right"], scene["bottom"] / ratio)
    image = np.zeros((int(height), int(width), 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], int(height))):
            
        for j, x in enumerate(np.linspace(screen[0], screen[2], int(width))):

            pixel = np.array([x, y, -scene["near"]])
            origin = camera
            direction = normalize(pixel - origin)
            illumination = trace_ray(origin, direction, scene["spheres"], scene["lights"], scene["ambient_intensity"], scene["background_colour"], camera, 0)
            image[i, j] = np.clip(illumination, 0, 1)

    save_as_ppm(image, scene["output_file_name"])

# uses P3 format
def save_as_ppm(image_array, file_name):
    height, width, _ = image_array.shape
    max_val = 255

    with open(file_name, 'w') as file:
        # header
        file.write("P3\n")
        file.write("{} {}\n".format(width, height))
        file.write("{}\n".format(max_val))

        # pixel data
        for row in image_array:
            for pixel in row:
                # ensure pixel values in [0, 255]
                r, g, b = (int(max(0, min(max_val, val))) for val in pixel * max_val)
                file.write("{} {} {} ".format(r, g, b))
            file.write("\n")

if __name__ == "__main__":
    main()