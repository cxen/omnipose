"""
Utility functions for graceful application shutdown.
"""
import logging
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

class ShutdownHandler(QObject):
    """
    Handles graceful shutdown of application components.
    Connect to the shutdown_requested signal to perform cleanup.
    """
    shutdown_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self._handlers_called = False
    
    def request_shutdown(self):
        """
        Request application shutdown by emitting the shutdown_requested signal.
        This allows components to clean up before the application exits.
        """
        if not self._handlers_called:
            logger.info("Initiating graceful shutdown sequence")
            self.shutdown_requested.emit()
            self._handlers_called = True
            logger.info("Shutdown handlers executed")
        
    def connect_to_app(self, app):
        """
        Connect this handler to application's aboutToQuit signal.
        
        Parameters:
            app (QApplication): The Qt application instance
        """
        app.aboutToQuit.connect(self.request_shutdown)
        logger.info("Shutdown handler connected to application")

# Singleton instance to be used across the application
handler = ShutdownHandler()
