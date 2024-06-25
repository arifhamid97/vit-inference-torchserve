import logging
import io
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from ts.torch_handler.base_handler import BaseHandler
import sys


# logger.setLevel(logging.DEBUG)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__file__)

class ViTHandler(BaseHandler):
    """
    Vision Transformer handler class. This handler takes an image as input and returns the classification label.
        - example https://github.com/pytorch/serve/tree/master/ts/torch_handler
        - performance_guide https://github.com/pytorch/serve/blob/master/docs/performance_guide.md
    """
    def __init__(self):
        super(ViTHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):

        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps") # Using apple silicon to process our tensor

        # Initialize model and processor
        self.processor = ViTImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        logger.error(f"Transformer model from path {model_dir} loaded successfully")

        self.initialized = True

    def preprocess(self, data:list):
        """
        Preprocess the input data to be suitable for model inference.
        """
        list_of_PIL_images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body") or row.get("file")

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):

                buffer = io.BytesIO(image)
                pil_image = Image.open(buffer)
                list_of_PIL_images.append(pil_image)



        """
        The ViTImageProcessor accept list of PIL Image / Single PIL Image and return: {"pixel_format":[[Tensor]]}
        """
        list_of_input_tensor = self.processor(images=list_of_PIL_images, return_tensors="pt").to(self.device)
        return list_of_input_tensor


    def inference(self, inputs):
        """
        Perform inference on the preprocessed data using the loaded model.
        """
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits

        return logits.argmax(-1)

    def postprocess(self, inference_output:torch.Tensor) -> list:
        """
        Postprocess the inference output to be returned to the client.
        """
        list_of_predicted_class=[]

        for idx in range(inference_output.size(0)):
            predicted_class = self.model.config.id2label[inference_output[idx].item()]
            list_of_predicted_class.append(predicted_class)

            logger.info(f"Model predicted: {predicted_class}")
        
        return self._client_to_json_format(list_of_predicted_class)

    def _client_to_json_format(self,data:list):
        """
        Helper method to convert respond data to custom format
        """
        final_output = []
        for d in data:
            final_output.append({'data':d})

        return final_output
        

    def handle(self,data, context) -> list:
        """
        Entry point for TorchServe to handle inference requests. 
        
            - handle function will trigger when torchserve --start
            - torchserve handler always accept list as input and list as output
            - list(input) -> (preprocess) -> (inference) -> (postprocess) -> list(final output to client)
            - for big response tensor that might exceed max_response_size https://github.com/pytorch/serve/issues/2849

        """
        

        print(f"No of batch process---->{len(data)}")


        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None
        
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)

        """ handler function must return in list type """
        return data







