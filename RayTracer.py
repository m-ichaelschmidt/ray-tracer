# Ray Tracing - Michael Schmidt

import sys
import numpy as np
import matplotlib.pyplot as plt

# Updated function to parse the file with corrected light source parsing
def parse_file(file_lines):
    scene_data = {
        "near": None,
        "left": None,
        "right": None,
        "top": None,
        "bottom": None,
        "resolution": None,
        "spheres": [],
        "lights": [],
        "background_color": None,
        "ambient_intensity": None,
        "output_file_name": None
    }

    for line in file_lines:
        parts = line.strip().split()
        keyword = parts[0].upper()

        if keyword in ["NEAR", "LEFT", "RIGHT", "TOP", "BOTTOM"]:
            scene_data[keyword.lower()] = float(parts[1])
        elif keyword == "RES":
            scene_data["resolution"] = (int(parts[1]), int(parts[2]))
        elif keyword == "SPHERE":
            # Parsing sphere data
            name = parts[1]
            position = tuple(map(float, parts[2:5]))
            scaling = tuple(map(float, parts[5:8]))
            color = tuple(map(float, parts[8:11]))
            ka, kd, ks, kr = map(float, parts[11:15])
            n = int(parts[15])
            sphere = {
                "name": name,
                "position": position,
                "scaling": scaling,
                "color": color,
                "ka": ka,
                "kd": kd,
                "ks": ks,
                "kr": kr,
                "n": n
            }
            scene_data["spheres"].append(sphere)
        elif keyword == "LIGHT":
            # Parsing light source data with corrected format
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
            scene_data["background_color"] = tuple(map(float, parts[1:]))
        elif keyword == "AMBIENT":
            scene_data["ambient_intensity"] = tuple(map(float, parts[1:]))
        elif keyword == "OUTPUT":
            # Ensuring output file name is not more than 20 characters
            scene_data["output_file_name"] = parts[1][:20]

    return scene_data

def normalize(vector):

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # or you could raise an exception, return None, or handle it another way
    return vector / norm

def ellipsoid_intersect(center, scaling, ray_origin, ray_direction):
    # # Normalize the ray direction
    # # norm_ray_dir = normalize(ray_direction)

    # # Adjust ray origin and direction according to the ellipsoid's center and scaling
    # adjusted_origin = (ray_origin - center) / scaling
    # adjusted_direction = ray_direction / scaling

    # # The ellipsoid equation coefficients
    # A = np.dot(adjusted_direction, adjusted_direction)
    # B = 2 * np.dot(adjusted_origin, adjusted_direction)
    # C = np.dot(adjusted_origin, adjusted_origin) - 1


    # inverse T matrix method
    # Normalize the ray direction
    norm_ray_dir = normalize(ray_direction)

    # Inverse scaling matrix for non-uniform scaling
    inv_scaling_matrix = np.diag([1/s if s != 0 else 1 for s in scaling] + [1])

    # Transformation matrix for moving the ellipsoid center to the origin
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = -np.array(center)

    # Combined inverse transformation matrix
    inv_transform = np.dot(inv_scaling_matrix, translation_matrix)

    # Transform the ray into the object space
    transformed_ray_origin = np.dot(inv_transform, np.append(ray_origin, 1))[:3]
    transformed_ray_direction = np.dot(inv_transform, np.append(norm_ray_dir, 0))[:3]

    # Perform intersection test with the unit sphere in the transformed space
    A = np.dot(transformed_ray_direction, transformed_ray_direction)
    B = 2 * np.dot(transformed_ray_direction, transformed_ray_origin)
    C = np.dot(transformed_ray_origin, transformed_ray_origin) - 1


    # Solving the quadratic equation for t
    delta = B**2 - 4 * A * C
    if delta < 0:
        return None  # No intersection

    sqrt_delta = np.sqrt(delta)
    t1 = (-B + sqrt_delta) / (2 * A)
    t2 = (-B - sqrt_delta) / (2 * A)

    # Find the smallest positive t value
    if t1 > 0 and t2 > 0:
        return min(t1, t2)
    elif t1 > 0:
        return t1
    elif t2 > 0:
        return t2

    return None


# from medium article
def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def find_intersection(sphere, ray_origin, ray_direction):
    scaling = sphere["scaling"]
    center = sphere["position"]
    return ellipsoid_intersect(center, scaling, ray_origin, ray_direction)


def nearest_intersected_object(spheres, ray_origin, ray_direction):
    distances = [find_intersection(sphr, ray_origin, ray_direction) for sphr in spheres]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = spheres[index]
    return nearest_object, min_distance


def main():
    
    # Ensure correct usage
    if len(sys.argv) < 2:
        print("Usage: python RayTracer.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]

    with open(fileName, 'r') as file:
        fileContents = file.readlines()
    
    scene = parse_file(fileContents)

    # Begin 'Ray Tracing From Scratch in Python'
    width = scene["resolution"][0]
    height = scene["resolution"][1]
    camera = np.array([0, 0, 0])
    ratio = float(width) / height
    screen = (scene["left"], scene["top"] / ratio, scene["right"], scene["bottom"] / ratio)

    image = np.zeros((int(height), int(width), 3))
    print(scene["spheres"][0]["ks"])
    for i, y in enumerate(np.linspace(screen[1], screen[3], int(height))):
        for j, x in enumerate(np.linspace(screen[0], screen[2], int(width))):
            # image[i, j] = ...
            print("progress: %d/%d" % (i + 1, int(height)))

            pixel = np.array([x, y, -scene["near"]])
            origin = camera
            direction = normalize(pixel - origin)

            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(scene["spheres"], origin, direction)
            if nearest_object is None:
                image[i, j] = np.clip(scene["background_color"], 0, 1)
                continue

            # compute intersection point between ray and nearest object
            intersection = origin + min_distance * direction
            normal_to_surface = normalize(intersection - nearest_object["position"])
            shifted_point = intersection + 1e-7 * normal_to_surface
            intersection_to_light = normalize(scene["lights"][0]["position"] - shifted_point)

            _, min_distance = nearest_intersected_object(scene["spheres"], shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(scene["lights"][0]["position"] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            # if is_shadowed:
            #     continue

            if is_shadowed:
                break

            # RGB
            illumination = np.array(nearest_object["color"]) * np.array(scene["ambient_intensity"]) * nearest_object["ka"]

            # # ambient
            # illumination += nearest_object["ka"] * np.array(scene["ambient_intensity"])
            # illumination *= np.array(scene["ambient_intensity"])
   
            # diffuse
            # illumination += nearest_object["kd"] * np.array(scene["lights"][0]["intensity"]) * np.dot(intersection_to_light, normal_to_surface)
            illumination += nearest_object["kd"] * np.dot(intersection_to_light, normal_to_surface)
            # illumination += nearest_object["kd"] * normalize(np.dot(intersection_to_light, normal_to_surface))

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            # illumination += nearest_object["ks"] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
            # breakpoint()
            test = np.dot(normal_to_surface, H)
            if nearest_object["ks"] == 0.0:
                pass
            else:
                illumination += nearest_object["ks"] * 1 * test ** (nearest_object["n"] / 4)
            # breakpoint()

            image[i, j] = np.clip(illumination, 0, 1)

            print("progress: %d/%d" % (i + 1, int(height)))

    plt.imsave('RECENT.png', image)
    plt.imsave(scene["output_file_name"], image)

    # pixel = np.array([0, 0, -scene["near"]])
    # origin = camera
    # direction = normalize(pixel - origin)

    # print(scene["spheres"][0])
    # print(find_intersection(scene["spheres"][0], origin, direction))

    # print(scene)

if __name__ == "__main__":
    main()