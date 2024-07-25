from options.base_options import BaseOptions
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainOptions(BaseOptions):
    """
    This class defines the train options for the script.
    
    Parameters
    ----------
    None
    """
    
    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """
        Initialize train options
        """
        BaseOptions.initialize(self)
        
        self.parser.add_argument(
            "--chunk_size",
            type=int,
            default=300,
            help="Chunk size (Tokens) splitting for the document"
        )
        
        self.parser.add_argument(
            "--chunk_overlap",
            type=int,
            default=50,
            help="Chunk overlap (Tokens) splitting for the document"
        )
        
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help="Controls the randomness of the generated text from the model"
        )
        
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="hyde",
            choices=["hyde", "step_back"],
            help="Method to use for generating response"
        )
        
        
    
    