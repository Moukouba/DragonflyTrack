from setuptools import setup, find_packages

setup(
    name="dragonflytrack",
    version="1.3.0",
    description="DragonflyTrack-X: Multi-Object Tracker with IMM Kalman, sparse identity memory, and GMC",
    author="DragonflyTrack Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
    ],
    extras_require={
        "yolo": ["ultralytics", "pillow"],
        "torch": ["torch", "torchvision"],
    },
)
