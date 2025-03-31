import subprocess

# Define the YOLO command
yolo_command = "yolo task=detect mode=train model=yolov8s.pt data=config.yaml epochs=25 imgsz=224 project='/Users/deepshikhar/Documents/My_Projects/SLR/training_results' name='exp_1' plots=True"

def run_yolo(command):
    try:
        # Execute the YOLO command
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running YOLO:", e)

if __name__ == "__main__":
    # Run the YOLO command
    run_yolo(yolo_command)


