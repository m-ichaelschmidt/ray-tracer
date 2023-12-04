import subprocess

def run_tests():
    test_names = [
        "testAmbient",
        "testBackground",
        "testBehind",
        "testDiffuse",
        "testIllum",
        "testImgPlane",
        "testIntersection",
        "testParsing",
        "testReflection",
        "testSample",
        "testShadow",
        "testSpecular"
    ]

    for test in test_names:
        print(f"Running {test}...")
        subprocess.run(["python", "RayTracer.py", f"{test}.txt"])

if __name__ == "__main__":
    run_tests()