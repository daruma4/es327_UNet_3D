import random

class predictor:
    def __init__(self, model, image_array, mask_array):
        self.model = model
        self.image_array = image_array
        self.mask_array = mask_array
        
    def predict_single(self, img):
        """Returns the predicted mask using the model

        Args:
            model (_type_): _description_
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.model.predict(img[None,...])

    def random_predict(self):
        """Randomly selects an image from the object image_array and returns the predicted mask using the model

        Returns:
            _type_: _description_
        """
        idx = random.randint(0, len(self.image_array))
        random_image = self.image_array[idx]
        random_mask = self.mask_array[idx]
        random_predicted_mask = self.predict_single(random_image)
        
        return random_image, random_mask, random_predicted_mask
