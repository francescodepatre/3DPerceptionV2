import freenect
import cv2
import numpy as np
from errorMessages import ErrorManager

class Kinect:

    def __init__(self):
        self.manager = ErrorManager()

    def get_realtime_video(self):
        try:
            rgb, _ = freenect.sync_get_video()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return rgb
        except Exception as e:
            self.manager.log_exception()
            self.manager.message_error("Impossibile comunicare con il sensore.")
            return None

    def get_realtime_depth(self):
        try:
            depth, _ = freenect.sync_get_depth()
            depth = depth.astype(np.uint8)
            return depth
        except Exception as e:
            self.manager.log_exception(e)
            self.manager.message_error(e)
            return None

    def calibra_kinect(self, pattern_size=(9, 6), square_size=0.025, save_file="kinect_calibration.npz"):
        # Prepara i punti del mondo reale in base alla dimensione della scacchiera
        object_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        object_points *= square_size

        # Array per memorizzare i punti trovati
        obj_points = []  # Punti 3D nel mondo reale
        img_points_rgb = []  # Punti 2D nell'immagine RGB
        img_points_depth = []  # Punti 2D nell'immagine di profondità

        print("Posiziona la scacchiera in diverse posizioni davanti al Kinect e premi 's' per catturare.")
        print("Premi 'q' per terminare l'acquisizione e avviare la calibrazione.")

        while True:
            # Acquisisce i frame RGB e di profondità
            rgb_frame, _ = freenect.sync_get_video()
            depth_frame, _ = freenect.sync_get_depth()

            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            depth_frame = depth_frame.astype(np.uint8)

            # Trova i punti della scacchiera nell'immagine RGB
            ret_rgb, corners_rgb = cv2.findChessboardCorners(gray_rgb, pattern_size)

            if ret_rgb:
                cv2.drawChessboardCorners(rgb_frame, pattern_size, corners_rgb, ret_rgb)
                cv2.imshow("RGB Frame", rgb_frame)

            cv2.imshow("Depth Frame", depth_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and ret_rgb:
                # Memorizza i punti se la scacchiera è stata trovata
                obj_points.append(object_points)
                img_points_rgb.append(corners_rgb)
                img_points_depth.append(corners_rgb)
                print("Immagine acquisita")

            elif key == ord('q'):
                print("Acquisizione terminata")
                break

        cv2.destroyAllWindows()

        # Calibrazione della fotocamera RGB e di profondità
        ret_rgb, mtx_rgb, dist_rgb, rvecs_rgb, tvecs_rgb = cv2.calibrateCamera(
            obj_points, img_points_rgb, gray_rgb.shape[::-1], None, None
        )
        ret_depth, mtx_depth, dist_depth, rvecs_depth, tvecs_depth = cv2.calibrateCamera(
            obj_points, img_points_depth, depth_frame.shape[::-1], None, None
        )

        # Stampa i risultati della calibrazione
        print("Calibrazione RGB:")
        print("Matrice intrinseca:", mtx_rgb)
        print("Coefficiente di distorsione:", dist_rgb)

        print("Calibrazione Depth:")
        print("Matrice intrinseca:", mtx_depth)
        print("Coefficiente di distorsione:", dist_depth)

        # Salva i parametri su file per utilizzo successivo
        np.savez(save_file, mtx_rgb=mtx_rgb, dist_rgb=dist_rgb, mtx_depth=mtx_depth, dist_depth=dist_depth)
        print(f"Parametri di calibrazione salvati in '{save_file}'.")
