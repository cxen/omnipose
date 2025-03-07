import numpy as np

# for cursor
from PySide6.QtWidgets import QGraphicsPathItem
from PySide6.QtGui import QPen, QBrush, QPainterPath, QTransform
from PySide6.QtGui import QCursor
from PySide6.QtCore import QPointF
import pyqtgraph as pg

# cursor highlight, maybe more of an annotation thing 
def update_highlight(self, pos=None):
    # Check if drawing is enabled and required attributes are available
    if not hasattr(self, 'SCheckBox') or not self.SCheckBox.isChecked():
        return

    # More thorough checks for image
    if not hasattr(self, 'img') or self.img is None:
        return
    if not hasattr(self.img, 'image') or self.img.image is None:
        return
    if not hasattr(self.img.image, 'shape'):
        return
        
    if pos is None:
        # Get the current global mouse position
        mouse_pos = QCursor.pos()  # Get the cursor position in global coordinates
        scene_mouse_pos = self.p0.scene().views()[0].mapFromGlobal(mouse_pos)  # Map to scene coordinates
        # Convert scene_mouse_pos to QPointF
        pos = QPointF(scene_mouse_pos.x(), scene_mouse_pos.y())
    
    # Map the cursor position from scene to view coordinates
    view_pos = self.p0.mapSceneToView(pos)
    
    # Get cursor position in image data coordinates
    x, y = int(view_pos.x()), int(view_pos.y())
    
    # Ensure the position is within the image bounds
    if x < 0 or y < 0 or x >= self.img.image.shape[1] or y >= self.img.image.shape[0]:
        if hasattr(self, 'highlight_rect'):
            self.highlight_rect.hide()
        return
    
    # Initialize highlight_rect if it doesn't exist
    if not hasattr(self, 'highlight_rect'):
        self.highlight_rect = QGraphicsPathItem()
        self.p0.addItem(self.highlight_rect)
        
    self.highlight_rect.show()
    
    # Get the kernel and its dimensions
    px = np.ones((1, 1))
    
    # Check if layer exists and has kernel attribute
    if not hasattr(self, 'layer'):
        kernel = px
    else:
        kernel = getattr(self.layer, '_kernel', px) if self.SCheckBox.isChecked() else px
    
    # Recompute the path if needed
    if not hasattr(self, 'highlight_path') or not hasattr(self, '_last_kernel_shape') or self._last_kernel_shape != kernel.shape:
        self.compute_kernel_path(kernel)
        self._last_kernel_shape = kernel.shape
        
    # Position the cached path relative to the cursor
    transform = QTransform()
    transform.translate(x - kernel.shape[1] // 2, y - kernel.shape[0] // 2)
    transformed_path = transform.map(self.highlight_path)  # Apply transformation
    self.highlight_rect.setPath(transformed_path)
    
    base_hex = "#FFF"  
    pen_color = pg.mkColor(base_hex)
    pen_color.setAlpha(100)  
    self.highlight_rect.setBrush(QBrush(pen_color))  # Semi-transparent fill
    pen_color.setAlpha(255)  
    
    self.highlight_rect.setPen(pg.mkPen(color=pen_color, width=1))

def compute_kernel_path(self, kernel):
    """
    Create and cache a QPainterPath representing the kernel shape for the cursor highlight.
    """
    path = QPainterPath()
    h, w = kernel.shape
    
    # Create a path that outlines the kernel
    for y in range(h):
        for x in range(w):
            if kernel[y, x]:
                path.addRect(x, y, 1, 1)
    
    self.highlight_path = path


