from . import cgt_core_panel
from . import cgt_hand_tracking

classes = [
    cgt_core_panel
]


def register():
    from ..cgt_utils import cgt_logging
    cgt_logging.init()
    cgt_core_panel.register()
    cgt_hand_tracking.register()


def unregister():
    for cls in classes:
        cls.unregister()
    cgt_hand_tracking.unregister()
