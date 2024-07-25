class DocumentProcessor:
    """
    This class is responsible for processing the document, 
    loading the data and preparing templates for the RAG methods.
    
    parameters
    ----------
    opt: Namespace
        The Namespace object.
        
    methods
    -------
    __init__(opt: Namespace)
        Initializes the DocumentProcessor object.
        
    preprocess_step_back()
        Preprocesses the step back template.
        
    preprocess_hyde()
        Preprocesses the hyde template.
        
    load_and_process()
        Loads the blog data and processes it.
    """
    def __init__(self, opt):
        self._opt = opt

    def preprocess_step_back(self):
        """
        This method preprocesses the step back template and uses LangChain
        to augment the templates.
        
        parameters
        ----------
        None
        """
        from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        self.prompt_step_back = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
                ),
                few_shot_prompt,
                ("user", "{question}"),
            ]
        )

    def preprocess_hyde(self):
        """
        This method preprocesses the hyde template and uses LangChain
        to augment the templates.
        
        parameters
        ----------
        None
        """
        from langchain_core.prompts import ChatPromptTemplate

        template = """Please write a scientific paper passage to answer the question
        Question: {question}
        Passage:"""
        self.prompt_hyde = ChatPromptTemplate.from_template(template)

    def load_and_process(self):
        import bs4
        from langchain_community.document_loaders import WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma

        # Load blog
        loader = WebBaseLoader(
            web_paths=(self._opt.url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        blog_docs = loader.load()

        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, 
            chunk_overlap=50
        )

        # Make splits
        splits = text_splitter.split_documents(blog_docs)

        # Index
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()
        )

        self.retriever = vectorstore.as_retriever()
