from .services.pred_services import MultiModal

multimodal_service = MultiModal()
def get_multimodal_service() -> MultiModal:
    return multimodal_service