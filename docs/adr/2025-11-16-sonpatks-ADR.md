# ADR-001: Use OSU COE High Performance Computing Cluster for Yolo Model Training

**Status:** Pending

**Context:**
Training the YOLOv11 model on a hammer-only dataset currently takes over 6 hours on local machines. We need faster. more scalable compute resources to support continued model experimentation and dataset expansion. Requirements include support for higher batch sizes, consistent Python/CUDA environment, and capability to run multiple training jobs without hindering local development.

**Decision:**
We will use the OSU COE High Performance Computing Cluster for training the object detection (YOLO) model.

**Options Considered:**
- Don't adjust model; allow training to run for 6+ hrs
- Adjusting model to smaller variant (YOLOv11 to YOLOv5 or YOLOv8)
- Use faster hardware (use more powerful GPU to increase maximum batch size)
- Optimize hyperparameters (increase batch size and/or decrease size of input image)
- Use multi-scale training (training on images of differents sizes to improve detection of objects of various scales and distances)
- Caching (storing preprocessed images in memory to avoid increased loading times)

**Rationale:**
The capabilities of the HPC Cluster would greatly reduce training time, allow for larger and more complex models/datasets, and provide a greater scalability than the operating systems on our personal computers.

**Consequences:**
- Team members need to requrest COE HPC accounts.
- We will need to attend an Intro to HPC training session with COE HPC Manager.
- We will use the Lmod Environment Modules to access CUDA and Python for faster training.
- Future training (especially completed by future DAM Robotics members) may require careful planning.

**References:**
- [COE HPC Cluster documentation](https://it.engineering.oregonstate.edu/hpc)
- Project Board Issue (https://github.com/OSURoboticsClub/Rover_2023_2024/issues/39#issue-3603889956)
- Team discussion notes on YOLO training (2025-11-09-MarsRoverAutonomy-Sprint3-ProgressReport), see:
    - Progress vs Plan: Object Detection
    - Risk and Quality: Risk 3