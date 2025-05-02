
import cv2
import numpy as np
TARGET_COLORS = ['green','blue','orange','red','white','yellow']
class ColorDetector:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.current_color_idx = 0
        self.sampled = {c: [] for c in TARGET_COLORS}
        self.show_help = True
        self.running = True
        cv2.namedWindow("Color Detector")

    def start_camera(self, idx=0):
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            print("Error: cannot open camera")
            return False
        cv2.setMouseCallback("Color Detector", self.on_click)
        return True
    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.frame is None: return
        rgb = self.frame[y, x]           
        hsv = cv2.cvtColor(
            np.uint8([[rgb]]), cv2.COLOR_BGR2HSV
        )[0][0]
        cname = TARGET_COLORS[self.current_color_idx]
        self.sampled[cname].append((x,y, list(rgb), list(hsv)))
        print(f"Sampled {cname}: RGB={rgb[::-1]} HSV={hsv}")

    def draw_help(self, frame):
        txt = [
            f"Sampling color: {TARGET_COLORS[self.current_color_idx].upper()}",
            "Left‐click to sample that color",
            "- N: next color",
            "- S: save & exit",
            "- ?: toggle help"
        ]
        for i, line in enumerate(txt):
            cv2.putText(frame, line, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    def run(self):
        if not self.start_camera(): 
            return
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret: break
            for c, pts in self.sampled.items():
                color_bgr = {
                  'green':(0,255,0),
                  'blue':(255,0,0),
                  'orange':(0,165,255),
                  'red':(0,0,255),
                  'white':(255,255,255),
                  'yellow':(0,255,255)}[c]
                for x,y,_,_ in pts:
                    cv2.drawMarker(self.frame, (x,y), color_bgr,
                                   markerType=cv2.MARKER_CROSS, 
                                   markerSize=10, thickness=2)

            if self.show_help:
                self.draw_help(self.frame)
            cv2.imshow("Color Detector", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('?'):
                self.show_help = not self.show_help
            elif key == ord('n'):
                if self.current_color_idx < len(TARGET_COLORS)-1:
                    self.current_color_idx += 1
                else:
                    print("Already at last color—press 's' to save.")
            elif key == ord('s'):
                self.save_colors()
                break
            elif key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def save_colors(self):
        with open("cube_colors.txt","w") as f:
            f.write("#Sample#,Color, X, Y, R, G, B, H, S, V\n")
            idx = 1
            for cname in TARGET_COLORS:
                for (x,y,rgb,hsv) in self.sampled[cname]:
                    r,g,b = rgb[::-1]   
                    h,s,v = hsv
                    f.write(f"{idx},{cname},{x},{y},{r},{g},{b},{h},{s},{v}\n")
                    idx += 1
        print("Saved cube_colors.txt with named samples.")
if __name__ == "__main__":
    print("Rubik's Cube Color Detector\n")
    detector = ColorDetector()
    detector.run()
