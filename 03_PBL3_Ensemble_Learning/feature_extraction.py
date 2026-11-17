import cv2
import numpy as np

def extract_pixel_features(img_rgb):
    """
    Extracts 18 spatial, spectral, edge and texture features for each pixel:
    - Normalized RGB (3)
    - Normalized HSV (3)
    - Normalized LAB (3)
    - Normalized Grayscale (1)
    - Sobel gradients: dx, dy, and gradient magnitude (3)
    - Laplacian edge operator (1)
    - Multi-scale local mean blurs (5x5, 11x11) (2)
    - Multi-scale local standard deviation textures (5x5, 11x11) (2)
    Total: 18 features
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    rgb = img_rgb.astype(np.float32) / 255.0

    # Sobel Gradients
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Laplacian Edge Response
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    # Multi-scale neighborhood statistics
    blur5 = cv2.blur(gray, (5, 5))
    blur11 = cv2.blur(gray, (11, 11))
    std5 = np.sqrt(np.maximum(0, cv2.blur(gray**2, (5, 5)) - blur5**2))
    std11 = np.sqrt(np.maximum(0, cv2.blur(gray**2, (11, 11)) - blur11**2))

    features = [
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        lab[:, :, 0], lab[:, :, 1], lab[:, :, 2],
        gray,
        sobelx, sobely, grad_mag,
        laplacian,
        blur5, blur11,
        std5, std11
    ]

    feature_matrix = np.stack(features, axis=-1).reshape(-1, len(features))
    return feature_matrix, img_rgb.shape[:2]

