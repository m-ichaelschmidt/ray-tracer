# Ray Tracer

This is a simple implementation of a Ray Tracer using Python. It renders PPM format images of 3D scenes composed of spheres and light sources.

## Installation

1. Download RayTracer.py.
2. Install all necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python RayTracer.py input.txt
```
with ``input.txt`` as the path to your input file.

## Input Files
This is the scene configuration - it should follow the following format:
```
NEAR <distance>
LEFT <left>
TOP <top>
RIGHT <right>
BOTTOM <bottom>
RES <width> <height>
SPHERE <name> <center_x> <center_y> <center_z> <scale_x> <scale_y> <scale_z> <color_r> <color_g> <color_b> <ka> <kd> <ks> <kr> <shininess>
LIGHT <name> <position_x> <position_y> <position_z> <color_r> <color_g> <color_b>
AMBIENT <r> <g> <b>
BACK <r> <g> <b>
OUTPUT <output_file.ppm>
```

- ``NEAR``: Sets the z-distance of the near plane from the camera.
- ``LEFT, TOP, RIGHT, BOTTOM``: Image boundaries.
- `RES`: Resolution of the rendered image in pixels (width x height).
- `SPHERE`: Defines a sphere with a name, center coordinates, scale factors, color, and material properties (ambient, diffuse, specular, reflective, shininess). Can define multiple per file.
- `LIGHT`: Defines a light source with a name, position, and color. Can define multiple per file.
- `AMBIENT`: Ambient intensity of each colour for ambient scene lighting.
- `BACK`: Background color of the scene.
- `OUTPUT`: File name for the rendered output image.

## Example:
### testSample.txt
```
NEAR 1
LEFT -1
RIGHT 1
BOTTOM -1
TOP 1
RES 600 600
SPHERE s1 0 0 -10 2 4 2 0.5 0 0 1 1 0.9 0 50
SPHERE s2 4 4 -10 1 2 1 0 0.5 0 1 1 0.9 0 50
SPHERE s3 -4 2 -10 1 2 1 0 0 0.5 1 1 0.9 0 50
LIGHT l1 0 0 0 0.9 0.9 0.9
LIGHT l2 10 10 -10 0.9 0.9 0
LIGHT l3 -10 5 -5 0 0 0.9
BACK 1 1 1
AMBIENT 0.2 0.2 0.2
OUTPUT testSample.ppm
```

The result is a scene with the camera at (0, 0, 1) containing 3 spheres and 3 light sources. It is output as ``testSample.ppm`` and looks like this:
# Ray Tracer

This is a simple implementation of a Ray Tracer using Python. It renders PPM format images of 3D scenes composed of spheres and light sources.

## Installation

1. Download RayTracer.py.
2. Install all necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python RayTracer.py input.txt
```
with ``input.txt`` as the path to your input file.

## Input Files
This is the scene configuration - it should follow the following format:
```
NEAR <distance>
LEFT <left>
TOP <top>
RIGHT <right>
BOTTOM <bottom>
RES <width> <height>
SPHERE <name> <center_x> <center_y> <center_z> <scale_x> <scale_y> <scale_z> <color_r> <color_g> <color_b> <ka> <kd> <ks> <kr> <shininess>
LIGHT <name> <position_x> <position_y> <position_z> <color_r> <color_g> <color_b>
AMBIENT <r> <g> <b>
BACK <r> <g> <b>
OUTPUT <output_file.ppm>
```

- ``NEAR``: Sets the z-distance of the near plane from the camera.
- ``LEFT, TOP, RIGHT, BOTTOM``: Image boundaries.
- `RES`: Resolution of the rendered image in pixels (width x height).
- `SPHERE`: Defines a sphere with a name, center coordinates, scale factors, color, and material properties (ambient, diffuse, specular, reflective, shininess). Can define multiple per file.
- `LIGHT`: Defines a light source with a name, position, and color. Can define multiple per file.
- `AMBIENT`: Ambient intensity of each colour for ambient scene lighting.
- `BACK`: Background color of the scene.
- `OUTPUT`: File name for the rendered output image.

## Example:
### testSample.txt
```
NEAR 1
LEFT -1
RIGHT 1
BOTTOM -1
TOP 1
RES 600 600
SPHERE s1 0 0 -10 2 4 2 0.5 0 0 1 1 0.9 0 50
SPHERE s2 4 4 -10 1 2 1 0 0.5 0 1 1 0.9 0 50
SPHERE s3 -4 2 -10 1 2 1 0 0 0.5 1 1 0.9 0 50
LIGHT l1 0 0 0 0.9 0.9 0.9
LIGHT l2 10 10 -10 0.9 0.9 0
LIGHT l3 -10 5 -5 0 0 0.9
BACK 1 1 1
AMBIENT 0.2 0.2 0.2
OUTPUT testSample.ppm
```

The result is a scene with the camera at (0, 0, 1) containing 3 spheres and 3 light sources. The image is output as ``testSample.ppm``.

