import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class CLIPScorer:
    """
    Оценивает соответствие текст→изображение с помощью CLIP.
    Поддерживает единичную и пакетную обработку, автоматический выбор устройства,
    JIT-трейсинг и логирование.

    Пример:
        scorer = CLIPScorer(device="cuda", use_jit=False, half_precision=True)
        score = scorer.score("path/to/image.png", "A photo of a cat")
        batch = scorer.score_batch(["img1.png", "img2.png"], ["prompt1", "prompt2"])
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        use_jit: bool = False,
        half_precision: bool = False
    ):
        # Выбираем устройство
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Загружаем модель и процессор
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        # Переводим в половинную точность, если запрошено
        if half_precision and self.device.startswith("cuda"):
            self.model.half()
        # Опциональный JIT-трейсинг для ускорения
        if use_jit:
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                self.model = torch.jit.trace(self.model, (dummy, ))

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Не удалось загрузить изображение {image_path}: {e}")
            raise

    def score(self, image_path: str, prompt: str) -> float:
        """
        Считает CLIP-скор между одним изображением и одним текстом.
        """
        img = self._load_image(image_path)
        inputs = self.processor(text=[prompt], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits_per_image: [1, 1]
            score = outputs.logits_per_image.squeeze().item()
        return score

    def score_batch(
        self,
        image_paths: List[str],
        prompts: List[str]
    ) -> List[float]:
        """
        Пакетная оценка: сопоставляет каждый prompt с соответствующим image_paths[i].
        Если len(prompts) == 1, то одна подсказка ко всем изображениям.
        """
        images = [self._load_image(p) for p in image_paths]
        # Если одна подсказка — продублируем её под все изображения
        if len(prompts) == 1 and len(image_paths) > 1:
            prompts = prompts * len(image_paths)
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # [N, N] или [N, 1] если один prompt
            if logits.shape[0] == logits.shape[1]:
                # диагональные элементы: птичка i→i
                scores = logits.diag().tolist()
            else:
                # каждый логит в столбце 0
                scores = logits[:, 0].tolist()
        return scores

    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Возвращает предварительно обработанное (normalized) изображение (pixel_values).
        """
        img = self._load_image(image_path)
        inputs = self.processor(images=img, return_tensors="pt", padding=True)
        return inputs["pixel_values"].to(self.device)