class HyDERAG:
    """
    This class is responsible for generating documents using the HyDE model.
    
    parameters
    ----------
    method_name: str
        The name of the method.
        
    processor: DocumentProcessor
        The DocumentProcessor object.
        
    generate_docs_for_retrieval: Callable
        The function to generate documents for retrieval.
    
    methods
    -------
    generate_document(question: str) -> str:
        Generates a document for the given question using the HyDE model.
    """
    def __init__(self):
        self.method_name = "hyde"
        self.processor = None
        self.generate_docs_for_retrieval = None

    def generate_document(self, question):
        """
        This method generates a document for the given question
        using the HyDE technique.
        
        parameters
        ----------
        question: str
            The question for which the document is to be generated.
            
        returns
        -------
        str
            The generated document.
            
        raises
        ------
        ValueError
            If there is an error in the HyDERAG initialization 
            or in generating the document.
        """
        if self.processor is None or self.generate_docs_for_retrieval is None:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.output_parsers import StrOutputParser
                from utils.document_processor import DocumentProcessor

                self.processor = DocumentProcessor()
                self.processor.preprocess_hyde()

                self.generate_docs_for_retrieval = (
                    self.processor.prompt_hyde
                    | ChatOpenAI(temperature=self._opt.temperature)
                    | StrOutputParser()
                )
            except Exception as e:
                raise ValueError(f"Error in HyDERAG initialization: {e}")

        try:
            doc = self.generate_docs_for_retrieval.invoke({"question": question})
            return doc
        except Exception as e:
            raise ValueError(f"Error in HyDERAG generate_document: {e}")
