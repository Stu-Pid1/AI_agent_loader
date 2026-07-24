from typing import Dict, List, Optional, Type

from runners.base import BaseRunner


class TaskRegistry:

    def __init__(self):
        self._registry: Dict[str, Type[BaseRunner]] = {}
        self._register_defaults()

    def register(self, pipeline_tag: str, runner_class: Type[BaseRunner]) -> None:
        self._registry[pipeline_tag] = runner_class

    def get_runner_class(self, pipeline_tag: str) -> Optional[Type[BaseRunner]]:
        return self._registry.get(pipeline_tag)

    def supported_tags(self) -> List[str]:
        return list(self._registry.keys())

    def is_supported(self, pipeline_tag: str) -> bool:
        return pipeline_tag in self._registry

    def _register_defaults(self) -> None:
        from runners.text_generation import TextGenerationRunner
        from runners.image_generation import ImageGenerationRunner
        from runners.text_classification import TextClassificationRunner
        from runners.token_classification import TokenClassificationRunner
        from runners.speech_to_text import SpeechToTextRunner
        from runners.text_to_speech import TextToSpeechRunner
        from runners.image_classification import ImageClassificationRunner
        from runners.object_detection import ObjectDetectionRunner
        from runners.summarization import SummarizationRunner
        from runners.translation import TranslationRunner

        self.register("text-generation", TextGenerationRunner)
        self.register("text2text-generation", TextGenerationRunner)
        self.register("text-to-image", ImageGenerationRunner)
        self.register("text-classification", TextClassificationRunner)
        self.register("sentiment-analysis", TextClassificationRunner)
        self.register("zero-shot-classification", TextClassificationRunner)
        self.register("token-classification", TokenClassificationRunner)
        self.register("ner", TokenClassificationRunner)
        self.register("automatic-speech-recognition", SpeechToTextRunner)
        self.register("text-to-speech", TextToSpeechRunner)
        self.register("text-to-audio", TextToSpeechRunner)
        self.register("image-classification", ImageClassificationRunner)
        self.register("object-detection", ObjectDetectionRunner)
        self.register("summarization", SummarizationRunner)
        self.register("translation", TranslationRunner)

    def register_runner(
        self, pipeline_tags: List[str], runner_class: Type[BaseRunner]
    ) -> None:
        for tag in pipeline_tags:
            self.register(tag, runner_class)
