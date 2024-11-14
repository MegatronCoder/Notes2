pip3 install ultralytics 
pip3 install numpy pandas scikit-learn 
pip3 install torch torchvision torchaudio


pip install wandb
import wandb

wandb.login()

  


kaggle_custom_yaml = '''
    

path: /kaggle/input/scratch-yolo/partitioned_data

train: /kaggle/input/scratch-yolo/partitioned_data/images/train
val: /kaggle/input/scratch-yolo/partitioned_data/images/val


names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign


nc: 10

'''

with open('kaggle_custom.yaml', 'w') as file:
    file.write(kaggle_custom_yaml)





# command
yolo train model=yolov8n.pt data=/kaggle/working/kaggle_custom.yaml epochs=10 imgsz=640



