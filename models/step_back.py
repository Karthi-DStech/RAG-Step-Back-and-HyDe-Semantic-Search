class StepBackRAG:
    """
    This class is used to generate responses for the StepBackRAG model.

    parameters
    ----------
    method_name: str
        The name of the method.

    processor: DocumentProcessor
        The DocumentProcessor object.

    generate_queries_step_back: Callable
        The function to generate queries for the step back context.

    _opt: Configuration
        The Configuration object.

    methods
    -------
    generate_response(question: str) -> str:
        Generates a response for the given question
        using the StepBackRAG technique.

    """

    def __init__(self, opt):
        self.method_name = "step_back"
        self.processor = None
        self.generate_queries_step_back = None
        self._opt = opt

    def generate_response(self, question):
        """
        This method generates a response for the given question
        using the StepBackRAG technique.

        parameters
        ----------
        question: str
            The question for which the response is to be generated.

        returns
        -------
        str
            The generated response.

        raises
        ------
        ValueError
            If there is an error in the StepBackRAG initialization
            or in generating the response.
        """
        if self.processor is None or self.generate_queries_step_back is None:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.output_parsers import StrOutputParser
                from utils.document_processor import DocumentProcessor

                self.processor = DocumentProcessor()
                self.processor.preprocess_step_back()

                self.generate_queries_step_back = (
                    self.processor.prompt_step_back
                    | ChatOpenAI(temperature=self._opt.temperature)
                    | StrOutputParser()
                )
            except Exception as e:
                raise ValueError(f"Error in StepBackRAG initialization: {e}")

        try:
            from langchain_core.prompts import ChatPromptTemplate

            response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. \
                Your response should be comprehensive and not contradicted with the following context if they are relevant. \
                    Otherwise, ignore them if they are not relevant.

            # {normal_context}
            # {step_back_context}

            # Original Question: {question}
            # Answer:"""
            response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

            chain = (
                {
                    "normal_context": lambda x: x[
                        "question"
                    ],  # Dummy retriever placeholder
                    "step_back_context": self.generate_queries_step_back
                    | (lambda x: {"question": x["question"]}),
                    "question": lambda x: x["question"],
                }
                | response_prompt
                | ChatOpenAI(temperature=self._opt.temperature)
                | StrOutputParser()
            )

            result = chain.invoke({"question": question})
            return result
        except Exception as e:
            raise ValueError(f"Error in StepBackRAG generate_response: {e}")
