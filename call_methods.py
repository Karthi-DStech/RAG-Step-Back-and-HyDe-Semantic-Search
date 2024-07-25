class RAGMethodCaller:
    """
    This class is responsible for calling the RAG methods.
    
    parameters
    ----------
    retriever: Retriever
        The Retriever object.
        
    methods
    -------
    call_method(method_name: str, question: str)
        Calls the method with the given name for the given question.
    """
    def __init__(self, retriever):
        """
        Initializes the RAGMethodCaller object.
        
        parameters
        ----------
        retriever: Retriever
            The Retriever object.
            
        raises
        ------
        ValueError
            If the retriever is None.
        """
        if not retriever:
            raise ValueError("Retriever cannot be None")
        self.retriever = retriever

    def call_method(self, method_name, question):
        """
        This method calls the method with the given 
        name for the given question.
        
        parameters
        ----------
        method_name: str
            The name of the RAG method to be used.
            
        question: str
            The question (Prompt) for which the response is to be generated.
            
        raises
        ------
        ValueError
            If the method name is invalid.
        """
        method_name = method_name.lower()
        if method_name == "step_back":
            try:
                from models.step_back import StepBackRAG

                rag_method = StepBackRAG()
                result = rag_method.generate_response(question)
                print("Step Back RAG Result:", result)
            except ValueError as e:
                print(f"Error in StepBackRAG: {e}")

        elif method_name == "hyde":
            try:
                from models.hyde import HyDERAG

                rag_method = HyDERAG()
                result = rag_method.generate_document(question)
                print("HyDE RAG Result:", result)
            except ValueError as e:
                print(f"Error in HyDERAG: {e}")

        else:
            print(
                "Invalid method name provided. Please choose from 'step_back' or 'hyde'."
            )
