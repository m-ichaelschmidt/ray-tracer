# Ray Tracing - Michael Schmidt

import sys
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime as time

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
                # Parsing sphere data
                name = parts[1]
                position = tuple(map(float, parts[2:5]))
                scaling = tuple(map(float, parts[5:8]))
                color = tuple(map(float, parts[8:11]))
                ka, kd, ks, kr = map(float, parts[11:15])
                n = int(parts[15])

                # Inverse scaling matrix for non-uniform scaling
                inv_scaling_matrix = np.diag([1/s if s != 0 else 1 for s in scaling] + [1])

                # Transformation matrix for moving the ellipsoid center to the origin
                translation_matrix = np.identity(4)
                translation_matrix[:3, 3] = -np.array(position)

                # Combined inverse transformation matrix
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
                scene_data["background_colour"] = tuple(map(float, parts[1:]))
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

def ellipsoid_intersect(inv_transform, ray_origin, ray_direction):
    # inverse T matrix method
    # Normalize the ray direction
    norm_ray_dir = normalize(ray_direction)

    # Inverse scaling matrix for non-uniform scaling
    # inv_scaling_matrix = np.diag([1/s if s != 0 else 1 for s in scaling] + [1])

    # # Transformation matrix for moving the ellipsoid center to the origin
    # translation_matrix = np.identity(4)
    # translation_matrix[:3, 3] = -np.array(center)

    # # Combined inverse transformation matrix
    # inv_transform = np.dot(inv_scaling_matrix, translation_matrix)

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

    # Check if intersections are behind the near plane
    valid_t1 = t1 > 0 and (ray_origin + t1 * norm_ray_dir)[2] < -1
    valid_t2 = t2 > 0 and (ray_origin + t2 * norm_ray_dir)[2] < -1

    # Return the valid t value
    if valid_t1 and valid_t2:
        return min(t1, t2) #if min(t1, t2) > 0 else max(t1, t2)
    elif valid_t1:
        return t1
    elif valid_t2:
        return t2

    return None

def nearest_intersected_object(spheres, ray_origin, ray_direction):
    # distances = [find_intersection(sphr, ray_origin, ray_direction) for sphr in spheres]
    distances = [ellipsoid_intersect(sphr["trans_matrix"], ray_origin, ray_direction) for sphr in spheres]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = spheres[index]
    return nearest_object, min_distance

def calculate_normal_at_intersection(sphere, intersection_point):
    # Calculate the normal for the intersection point as if the ellipsoid were a unit sphere at the origin
    # This involves first translating the intersection point by the negative of the sphere's position
    translated_intersection = intersection_point - np.array(sphere["position"])

    inv_scaling_matrix = sphere["inv_scaling_matrix"]

    # Then apply the inverse scaling to this point
    # Since normals are direction vectors, we don't translate them, only scale
    # inv_scaling_matrix = np.diag([1/s if s != 0 else 1 for s in sphere["scaling"]])
    unscaled_normal = normalize(np.dot(inv_scaling_matrix, translated_intersection))

    # Transform the normal back to the world space
    # For normals, we use the transpose of the inverse scaling matrix
    transformed_normal = normalize(np.dot(inv_scaling_matrix.T, unscaled_normal))

    return transformed_normal

def trace_ray(ray_origin, ray_direction, spheres, lights, ambient_intensity, background_colour, camera, depth=0):
    if depth > 3:  # Maximum recursion depth
        return np.array([0, 0, 0])  # Return black

    nearest_object, min_distance = nearest_intersected_object(spheres, ray_origin, ray_direction)
    if nearest_object is None and depth > 0:
        return np.array([0, 0, 0])  # Return black
    elif nearest_object is None and depth == 0:
        return np.clip(background_colour, 0, 1)

    # Compute the intersection point and normal
    intersection = ray_origin + min_distance * ray_direction
    normal_to_surface = calculate_normal_at_intersection(nearest_object, intersection)
    shifted_point = intersection + 1e-7 * normal_to_surface

    # Start with ambient lighting
    illumination = np.array(nearest_object["color"]) * np.array(ambient_intensity) * nearest_object["ka"]

    bisected_by_near_plane = (nearest_object["position"][2] - nearest_object["scaling"][2] < -1 and nearest_object["position"][2] + nearest_object["scaling"][2] > -1)

    # Add diffuse and specular components for each light source
    for light in lights:
        intersection_to_light = normalize(light["position"] - shifted_point)
        lightInSphere = False

        # Check if light is coming from inside a sphere
        if np.dot(intersection_to_light, normal_to_surface) < 0:
            lightInSphere = True
            normal_to_surface = -normal_to_surface  # Invert normal
            shifted_point = intersection + 1e-7 * normal_to_surface

        _, min_distance = nearest_intersected_object(spheres, shifted_point, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light["position"] - intersection)
        is_shadowed = min_distance < intersection_to_light_distance

        if bisected_by_near_plane and not lightInSphere:
            is_shadowed = True

        if is_shadowed:
            # return np.clip(illumination, 0, 1)
            if lightInSphere:
                # lightInSphere = False
                normal_to_surface = -normal_to_surface
                shifted_point = intersection + 1e-7 * normal_to_surface
            continue

        # diffuse
        illumination += nearest_object["kd"] * np.array(light["intensity"]) * np.dot(intersection_to_light, normal_to_surface) * np.array(nearest_object["color"])

        # specular - Phong lighting model
        if nearest_object["ks"] > 0:
            view_direction = normalize(camera - intersection)
            light_reflect_direction = reflect(-intersection_to_light, normal_to_surface)
            RdotV = max(0, np.dot(light_reflect_direction, view_direction))
            specular_intensity = nearest_object["ks"] * np.array(light["intensity"]) * (RdotV ** nearest_object["n"])
            illumination += specular_intensity

        # Reset normal if it was inverted
        if lightInSphere:
            lightInSphere = False
            normal_to_surface = -normal_to_surface
            shifted_point = intersection + 1e-7 * normal_to_surface

    # Handle reflections
    if nearest_object["kr"] > 0:
        reflected_ray_direction = reflect(ray_direction, normal_to_surface)
        reflected_color = trace_ray(shifted_point, reflected_ray_direction, spheres, lights, ambient_intensity, background_colour, camera, depth + 1)
        illumination += nearest_object["kr"] * reflected_color

    return np.clip(illumination, 0, 1)

# Reflection function
def reflect(direction, normal):
    return direction - 2 * np.dot(direction, normal) * normal

def main():
    
    startTime = time.now()

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
    # print(scene["spheres"][0]["ks"])
    for i, y in enumerate(np.linspace(screen[1], screen[3], int(height))):

        progress = i / height * 100
        if progress % 10 == 0:
            # print(f"Current progress: {int(progress):3d}%")
            print("Current progress: {:3d}%".format(int(progress)))

            
        for j, x in enumerate(np.linspace(screen[0], screen[2], int(width))):

            pixel = np.array([x, y, -scene["near"]])
            origin = camera
            direction = normalize(pixel - origin)

            illumination = trace_ray(origin, direction, scene["spheres"], scene["lights"], scene["ambient_intensity"], scene["background_colour"], camera, 0)

            image[i, j] = np.clip(illumination, 0, 1)

    print("Current progress: 100%") # remove
    # plt.imsave(scene["output_file_name"], image)
    save_as_ppm(image, scene["output_file_name"])

    finishTime = time.now()
    print(scene["output_file_name"][:-4] + ": " + str(finishTime - startTime)[3:9])

def save_as_ppm(image_array, filename):
    height, width, _ = image_array.shape
    max_val = 255

    with open(filename, 'w') as file:
        # Write the header
        file.write("P3\n")
        file.write("{} {}\n".format(width, height))
        file.write("{}\n".format(max_val))

        # Write the pixel data
        for row in image_array:
            for pixel in row:
                # Ensure pixel values are integers in the range [0, 255]
                r, g, b = (int(max(0, min(max_val, val))) for val in pixel * max_val)
                file.write("{} {} {} ".format(r, g, b))
            file.write("\n")

if __name__ == "__main__":
    main()

# ------------TEST RESULTS--------------
# testAmbient         -    57.6  -  PASS
# testBackground      -  1:32.0  -  PASS *
# testBehind          -    48.6  -  PASS 
# testDiffuse         -    58.2  -  PASS 
# testIllum           -  1:42.4  -  PASS * 
# testImgPlane        -    27.4  -  PASS 
# testIntersection    -  1:25.7  -  PASS *
# testParsing         -  1:08.7  -  PASS 
# testReflection      -  1:08.5  -  PASS 
# testSample          -  1:00.8  -  PASS 
# testShadow          -    57.4  -  PASS - shadow SLIGHTLY off on green sphere
# testSpecular        -  1:02.9  -  PASS 