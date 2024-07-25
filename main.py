from options.train_options import TrainOptions
from utils.configuration import Configuration
from utils.document_processor import DocumentProcessor
from call_methods import RAGMethodCaller


def main():
    """
    Runs the RAG model.

    parameters
    ----------
    None

    Process
    -------
    1. Parse the training options.
    2. Setup configuration.
    3. Initialize Document Processor and load blog data.
    4. Initialize RAG Method Caller with the retriever.
    5. Prompt user for a question.
    6. Call the RAG method.

    Returns
    -------
    The generated response from the RAG model.
    """
    # Parse the training options
    opt = TrainOptions().parse()

    # Setup configuration
    config = Configuration()
    config.install_dependencies()

    # Initialize Document Processor and load blog data
    processor = DocumentProcessor()
    processor.load_and_process()

    # Initialize RAG Method Caller with the retriever
    caller = RAGMethodCaller(processor.retriever)

    # Prompt user for a question
    question = input("Please enter your question: ")

    # Call the RAG method
    caller.call_method(opt.model_name, question)


if __name__ == "__main__":
    main()
