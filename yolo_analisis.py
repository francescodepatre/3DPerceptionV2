from ultralytics import YOLO

class YoloAnalisis:
    def  __init__(self, model_path, class_path):
        self.model = YOLO(model_path)

        with open(class_path,'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.framescounter = 0

    def analyze(rgb_image,dept_image):
        self.framescounter +=1

        if self.framescounter % 2 == 0:
            results = self.model(rgb_image)  # Passa il frame direttamente al modello
        else:
            return
        
        
    