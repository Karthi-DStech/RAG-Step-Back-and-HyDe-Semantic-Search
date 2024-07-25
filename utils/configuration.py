class Configuration:
    """
    This class is responsible for setting up the 
    environment and installing dependencies.
    
    parameters
    ----------
    opt: Namespace
        The Namespace object.
        
        
    methods
    -------
    __init__(opt: Namespace)
        Initializes the Configuration object.
        
    _setup_environment()
        Sets up the environment variables.
    """
    def __init__(self, opt):
        self._opt = opt
        self._setup_environment()

    def _setup_environment(self):
        """
        This method sets up the environment variables using 
        the Keys provided in the Namespace object.
        
        parameters
        ----------
        None
        """
        import os

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = self._opt.LangChainAPIKey
        os.environ["OPENAI_API_KEY"] = self._opt.OpenAPIKey

    def install_dependencies(self):
        """
        This method installs the required dependencies.
        
        parameters
        ----------
        None
        """
        import subprocess

        subprocess.check_call(
            [
                "pip",
                "install",
                "langchain_community",
                "tiktoken",
                "langchain-openai",
                "langchainhub",
                "chromadb",
                "langchain",
            ]
        )
