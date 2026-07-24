class AIAgentLoaderError(Exception):
    pass


class ModelLoadError(AIAgentLoaderError):
    pass


class VRAMError(AIAgentLoaderError):
    pass


class ModelNotLoadedError(AIAgentLoaderError):
    pass


class InferenceError(AIAgentLoaderError):
    pass


class DownloadError(AIAgentLoaderError):
    pass
